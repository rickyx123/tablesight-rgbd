"""National RGB temporal filtering, migrated from preserved square7/round adaptive.

No provincial class branches. Missing depth is UNKNOWN, never FLAT.
"""
import math
import statistics
from collections import Counter
from dataclasses import asdict

from national_competition.config import TABLE_CLASS_ID, CLASS_CONF_OVERRIDES, CLASS_TRACK_CONF_OVERRIDES
from national_competition.fusion import iou
from national_competition.ui.competition_visualizer import UiDetection
from national_competition.runtime.depth_validation import consistent_reject, merge_burst_evidence, HARD_DEPTH_CLASSES, WATER_DEPTH_CLASSES, PRINT_DEPTH_CLASSES, water_confirmed
from national_competition.runtime.round_water import choose_capacity as choose_round_capacity,distinct_bottles

CAPACITY_CLASSES = frozenset((10, 11))
CAPACITY_MIN_FRAMES = 3
CAPACITY_SCORE_MARGIN = .35


def capacity_observations(rows):
    observations = {}
    for i, row in enumerate(rows):
        entries = row.get('capacity_observations')
        if entries is None:
            scores = row.get('capacity_scores')
            if scores is None:
                scores = (row['confidence'] if row['class_id']==10 else 0.,
                          row['confidence'] if row['class_id']==11 else 0.)
            entries = [(row.get('sequence', row.get('moment', i)), row['class_id'], *scores)]
        for sequence, cls, small, large in entries:
            observations[sequence] = (sequence, cls, small, large)
    return list(observations.values())


def capacity_choice(rows):
    observations = capacity_observations(rows)
    votes = Counter(obs[1] for obs in observations)
    scores = {10: sum(obs[2] for obs in observations),
              11: sum(obs[3] for obs in observations)}
    winner = max(scores, key=scores.get)
    total = sum(scores.values())
    margin = (scores[winner]-scores[21-winner])/total if total>0 else 0.
    accepted = votes[winner]>=CAPACITY_MIN_FRAMES and margin>=CAPACITY_SCORE_MARGIN
    return accepted, winner, {'capacity_votes': dict(votes), 'capacity_scores': scores,
        'capacity_score_margin': margin, 'capacity_min_frames': CAPACITY_MIN_FRAMES,
        'capacity_required_margin': CAPACITY_SCORE_MARGIN}

SQUARE_FRAMES = 7
SQUARE_INTERVAL = 0.18
SQUARE_TIMEOUT = 3.0
ROUND_INTERVAL = 0.75
ROUND_MIN_MOMENTS = 12
ROUND_MAX_MOMENTS = 27
ROUND_MAX_SECONDS = 20.0
ROUND_BURST = 2
ROUND_BURST_SECONDS = 0.55
ROUND_SCOPE_RADIUS = 1.0
ROUND_TABLE_VERIFY_GROWTH = 1.08
ROUND_TABLE_MAX_UNVERIFIED_GROWTH = 1.20


def wrap_angle(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def polygon_contains(point, polygon, margin=8.0):
    x, y = point
    inside = False
    previous = polygon[-1]
    distance = float('inf')
    for current in polygon:
        ax, ay = previous
        bx, by = current
        dx, dy = bx - ax, by - ay
        if (ay > y) != (by > y) and x < dx * (y - ay) / dy + ax:
            inside = not inside
        t = max(0., min(1., ((x-ax)*dx+(y-ay)*dy) / max(dx*dx+dy*dy, 1e-9)))
        distance = min(distance, math.hypot(x-ax-t*dx, y-ay-t*dy))
        previous = current
    return inside or distance <= margin


def scope(detections, mode, table_override=None, table_polygon=None, table_contacts=()):
    tables = [d for d in detections if d.class_id == TABLE_CLASS_ID and d.confidence >= .25]
    if not tables:
        return None, [], [asdict(d) for d in detections if d.class_id != TABLE_CLASS_ID]
    table = table_override or max(tables, key=lambda d: d.confidence)
    x1, y1, x2, y2 = table.xyxy
    w, h = x2-x1, y2-y1
    if w <= 1 or h <= 1:
        return None, [], []
    polygon = [(x1+.08*w, y1+.02*h), (x2-.08*w, y1+.02*h),
               (x2-.01*w, y2-.02*h), (x1+.05*w, y2-.02*h)]
    ellipse = ((x1+x2)/2, (y1+y2)/2, w/2, h/2)
    accepted, rejected = [], []
    for d in detections:
        if not 0 <= d.class_id < 16 or d.confidence < CLASS_CONF_OVERRIDES.get(d.class_id,.35):
            continue
        bx1, by1, bx2, by2 = d.xyxy
        px, py = (bx1+bx2)/2, by1+.94*(by2-by1)
        row = asdict(d)
        row['depth_state'] = 'UNKNOWN'
        if mode == 'square':
            within = polygon_contains((px, py), table_polygon or polygon,
                                      margin=0. if table_polygon is not None else 8.)
            if not within and table_polygon is not None:
                contact=next((r for r in table_contacts if r['class_id']==d.class_id and tuple(r['xyxy'])==tuple(d.xyxy)),None)
                if contact is not None:
                    within=True;row['table_membership']=contact['basis']
        else:
            cx, cy, rx, ry = ellipse
            py = by2-.08*(by2-by1)
            nx, ny = (px-cx)/rx, (py-cy)/ry
            radius = math.hypot(nx, ny)
            centre_radius = math.hypot(nx, ((by1+by2)/2-cy)/ry)
            # Floor objects can have a box centre projected onto the table;
            # only their lower contact point determines RGB table membership.
            within = radius <= ROUND_SCOPE_RADIUS
            row.update(nx=nx, ny=ny, radius=radius, angle=math.atan2(ny, nx))
        (accepted if within else rejected).append(row)
    return table, accepted, rejected


class RoundTableGeometry:
    """Short-history tracking: follow motion, reject isolated detector jumps.

    Each observation comes from this frame's YOLO proposal, optionally checked
    against its own RGB-D surface. No table geometry is locked for a round.
    """
    def __init__(self):
        self.observations = []
        self.table = None
        self.calibration = None
        self.jumps = 0
        self.missing = 0
        self.last_sequence = None
        self.last_result = None
        self.pending = None
        self.updates = 0
        self.moves = 0
        self.source = None
        self.reason = 'WAITING_TABLE'
        self.moved = False
        self.shape_reference = None
        self.shape_guard_size = None
        self.unverified_expansions = 0

    def unverified_expansion(self, box, limit=ROUND_TABLE_VERIFY_GROWTH):
        """Compare to a verified shape, so small false growth cannot accumulate."""
        if self.shape_reference is None:
            return False
        references=[self.shape_reference]
        if self.calibration is not None:references.append(self.calibration['xyxy'])
        # An occluded rim can make a depth fit see only the inner tabletop.
        # Do not tighten the growth gate below the initialized full outline.
        width=max(c-a for a,b,c,d in references)
        height=max(d-b for a,b,c,d in references)
        if self.shape_guard_size is not None:
            width=max(width,self.shape_guard_size[0]);height=max(height,self.shape_guard_size[1])
        return max((box[2]-box[0])/width, (box[3]-box[1])/height) > limit

    @staticmethod
    def difference(a,b):
        aw,ah=a[2]-a[0],a[3]-a[1];bw,bh=b[2]-b[0],b[3]-b[1]
        centre=max(abs((a[0]+a[2]-b[0]-b[2])/2)/bw,
                   abs((a[1]+a[3]-b[1]-b[3])/2)/bh)
        size=max(abs(aw/bw-1),abs(ah/bh-1))
        return centre,size

    def update(self, raw, sequence=None, surface=None):
        if sequence is not None:
            if sequence == self.last_sequence:
                return self.last_result
            self.last_sequence = sequence
        self.moved=False
        if raw is None:
            self.missing += 1
            self.pending=None
            self.last_result=None
            self.reason='TABLE_MISSING'
            return None
        coords=tuple(surface['xyxy']) if surface is not None else tuple(raw.xyxy)
        if not all(math.isfinite(v) for v in coords) or coords[2]-coords[0]<=1 or coords[3]-coords[1]<=1:
            self.last_result=None;self.reason='INVALID_TABLE';return None
        source='depth_rgb_temporal' if surface is not None else 'yolo_temporal'
        measured_coords=coords
        # Still request depth verification from 8% growth in the caller, but
        # tolerate up to 20% when that frame has no usable boundary measurement.
        if surface is None and self.unverified_expansion(coords,ROUND_TABLE_MAX_UNVERIFIED_GROWTH):
            self.unverified_expansions+=1
            self.pending=None
            self.last_result=None
            self.reason='UNVERIFIED_TABLE_EXPANSION'
            return None
        candidate={'xyxy':coords,'confidence':raw.confidence,'source':source}
        self.observations.append(candidate);self.observations=self.observations[-3:]
        self.last_result=None
        if self.table is None:
            self.reason='INITIAL_TABLE_CONFIRMATION'
            if len(self.observations)<3:return None
            groups=[[o for o in self.observations if iou(o['xyxy'],c['xyxy'])>=.65]
                    for c in self.observations]
            cluster=max(groups,key=lambda g:(len(g),sum(o['confidence'] for o in g)))
            if len(cluster)<2:return None
            coords=tuple(statistics.median(o['xyxy'][i] for o in cluster) for i in range(4))
            self.calibration={'xyxy':coords,'raw_xyxy':raw.xyxy,'source':source,
                              'tracking':'dynamic_yolo_rgbd','initial_frames':3}
            self.reason='TABLE_INITIALIZED'
        else:
            centre,size=self.difference(coords,self.table.xyxy)
            width,height=self.shape_guard_size or (self.calibration['xyxy'][2]-self.calibration['xyxy'][0],
                                                    self.calibration['xyxy'][3]-self.calibration['xyxy'][1])
            recovering_outline=(surface is None and centre<=.10 and
                abs((coords[2]-coords[0])/width-1)<=.20 and
                abs((coords[3]-coords[1])/height-1)<=.20)
            if centre>.18 or (size>.25 and not recovering_outline):
                if self.pending is None or iou(coords,self.pending['xyxy'])<.80:
                    self.pending=candidate;self.jumps+=1
                    self.reason='TABLE_MOVE_CONFIRMATION'
                    # Never apply the old ROI to a suspected new table position.
                    return None
                coords=tuple((coords[i]+self.pending['xyxy'][i])/2 for i in range(4))
                self.pending=None;self.moved=True;self.moves+=1
                self.reason='TABLE_RELOCALIZED'
            else:
                self.pending=None
                alpha=.70 if surface is not None else .50
                coords=tuple(alpha*coords[i]+(1-alpha)*self.table.xyxy[i] for i in range(4))
                self.moved=centre>.06
                self.reason='TABLE_FOLLOWING'
        self.table=UiDetection(TABLE_CLASS_ID,raw.confidence,coords)
        if surface is not None:
            self.shape_reference=measured_coords
        elif self.shape_reference is None:
            self.shape_reference=coords
        if self.shape_guard_size is None:
            self.shape_guard_size=(coords[2]-coords[0],coords[3]-coords[1])
        if surface is not None:
            self.shape_guard_size=(max(self.shape_guard_size[0],measured_coords[2]-measured_coords[0]),
                                   max(self.shape_guard_size[1],measured_coords[3]-measured_coords[1]))
        self.last_result=self.table;self.source=source;self.updates+=1
        return self.last_result

    def diagnostics(self):
        return {'calibration':self.calibration,'tracking':'dynamic_yolo_rgbd',
                'shape_policy':'verified_shape_growth_relaxed_20261016',
                'depth_check_growth_ratio':ROUND_TABLE_VERIFY_GROWTH,
                'max_unverified_growth_ratio':ROUND_TABLE_MAX_UNVERIFIED_GROWTH,
                'current_xyxy':self.table.xyxy if self.table else None,'source':self.source,
                'pending_table_jumps':self.jumps,'accepted_table_moves':self.moves,
                'unverified_expansions_blocked':self.unverified_expansions,
                'shape_reference_xyxy':self.shape_reference,
                'shape_guard_size':self.shape_guard_size,
                'geometry_updates':self.updates,'table_missing_samples':self.missing,'reason':self.reason}


def select_track(track, required, strict=False):
    rows = track['rows']
    if all(row['class_id'] in CAPACITY_CLASSES for row in rows):
        accepted, winner, detail = capacity_choice(rows)
        best = max(rows, key=lambda r:r['confidence'])
        score = detail['capacity_scores'][winner]/max(1, len(capacity_observations(rows)))
        return accepted, winner, UiDetection(winner, score, tuple(best['xyxy'])), {
            'class_id': winner, 'votes': detail['capacity_votes'],
            'score_margin': detail['capacity_score_margin'], 'accepted': accepted,
            'depth_state': 'UNKNOWN', 'observations': len(rows), **detail}
    votes = Counter(row['class_id'] for row in rows)
    scores = Counter()
    for row in rows:
        scores[row['class_id']] += row['confidence']
    winner = max(votes, key=lambda cls: (votes[cls], scores[cls]))
    other = max((v for cls, v in votes.items() if cls != winner), default=0)
    margin = (scores[winner] - sum(v for cls, v in scores.items() if cls != winner)) / sum(scores.values())
    accepted = votes[winner] >= required and margin >= .35
    mean_confidence = statistics.mean(r['confidence'] for r in rows if r['class_id'] == winner)
    minimum_confidence = CLASS_TRACK_CONF_OVERRIDES.get(winner, 0.)
    accepted = accepted and mean_confidence >= minimum_confidence
    if strict and other:
        accepted = accepted and votes[winner] >= 6 and other <= 1
    best = max((r for r in rows if r['class_id'] == winner), key=lambda r: r['confidence'])
    return accepted, winner, UiDetection(winner, statistics.mean(
        r['confidence'] for r in rows if r['class_id'] == winner), tuple(best['xyxy'])), {
        'class_id': winner, 'votes': dict(votes), 'score_margin': margin,
        'accepted': accepted, 'depth_state': 'UNKNOWN', 'observations': len(rows),
        'mean_confidence': mean_confidence, 'required_mean_confidence': minimum_confidence}


def square_fuse(frames, use_depth=True):
    tracks = []
    for index, frame in enumerate(frames):
        used = set()
        for row in sorted(frame['objects'], key=lambda r: r['confidence'], reverse=True):
            candidates = [(iou(t['last']['xyxy'], row['xyxy']), n)
                          for n, t in enumerate(tracks) if n not in used]
            score, n = max(candidates, default=(0., -1))
            if score < .20:
                tracks.append({'rows': [], 'last': row})
                n = len(tracks)-1
            track = tracks[n]
            track['rows'].append(dict(row, moment=index))
            track['last'] = row
            used.add(n)
    selected, decisions = [], []
    for track in tracks:
        accepted, _, detection, decision = select_track(track, 5, strict=True)
        depth_rejected = use_depth and accepted and consistent_reject(track['rows'], class_id=detection.class_id)
        water_unconfirmed = False
        if detection.class_id in CAPACITY_CLASSES:
            decision['water_depth_confirmed']=water_confirmed(track['rows'],identity_confidence=detection.confidence) if use_depth else None
            decision['water_unknown_retained']=use_depth and not decision['water_depth_confirmed'] and not depth_rejected
        if any('depth_evidence' in r for r in track['rows']):
            decision['depth_rejected'] = depth_rejected
        decisions.append(decision)
        if accepted and not depth_rejected and not water_unconfirmed:
            selected.append(detection)
    return selected, decisions


def burst_vote(samples):
    # Glasses can disappear with a small viewing-angle change: either frame
    # can support their quantity. Other classes still require two frames.
    voted = []
    for cls in range(16):
        if cls==11:
            continue
        rows = [sorted([r for r in s['objects'] if r['class_id'] == cls or
                       (cls==10 and r['class_id']==11)],
                       key=lambda r: r['confidence'], reverse=True) for s in samples]
        required_frames = 1 if cls == 0 else 2
        count = max((n for n in range(1, 6) if sum(len(r) >= n for r in rows) >= required_frames), default=0)
        if count:
            representative = max((r for r in rows if len(r) >= count),
                                 key=lambda r: sum(x['confidence'] for x in r[:count]))
            voted.extend(representative[:count])
    # Different classes voted from different samples must not become two
    # entities when their representative boxes are almost identical.
    conflicts = set()
    for i, left in enumerate(voted):
        for j, right in enumerate(voted[:i]):
            if left['class_id'] == right['class_id']:
                continue
            a, b = left['xyxy'], right['xyxy']
            aa, ab = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
            if iou(a, b) >= .70 and min(aa, ab)/max(aa, ab, 1.) >= .75:
                conflicts.update((i, j))
    kept = [dict(r) for i,r in enumerate(voted) if i not in conflicts]
    for row in kept:
        if row['class_id'] in CAPACITY_CLASSES:
            matched = []
            for sample in samples:
                candidates = [r for r in sample['objects'] if r['class_id'] in CAPACITY_CLASSES
                              and iou(r['xyxy'], row['xyxy']) >= .30]
                if candidates:
                    matched.append(dict(max(candidates,key=lambda r:iou(r['xyxy'],row['xyxy'])),
                                        sequence=sample['sequence']))
            observations = capacity_observations(matched)
            row['capacity_observations'] = observations
            _, winner, detail = capacity_choice([row])
            row['class_id'] = winner
            row['confidence'] = detail['capacity_scores'][winner]/max(1, len(observations))
        row['depth_evidence']=merge_burst_evidence(row,samples)
        row['depth_state']=row['depth_evidence']['state']
    return {'objects': kept,
            'burst_conflicts': [voted[i] for i in sorted(conflicts)],
            'sequences': [s['sequence'] for s in samples], 'sample_count': len(samples)}


def rotation_delta(previous, current):
    deltas = []
    for cls in range(16):
        a = [r for r in previous['objects'] if r['class_id'] == cls and r['radius'] >= .25 and r['confidence'] >= .5]
        b = [r for r in current['objects'] if r['class_id'] == cls and r['radius'] >= .25 and r['confidence'] >= .5]
        if len(a) == len(b) == 1:
            # Reject large apparent radius changes; rotating a fixed object
            # should not move it to a different concentric ring.
            if abs(a[0]['radius']-b[0]['radius']) > .15:
                continue
            delta = wrap_angle(b[0]['angle']-a[0]['angle'])
            if .01 <= abs(delta) <= 1.35:
                deltas.append(delta)
    if len(deltas) < 2:
        return None
    middle = statistics.median(deltas)
    consistent = [d for d in deltas if abs(d-middle) <= .35]
    return statistics.median(consistent) if len(consistent) >= 2 else None


def round_fuse(frames, use_depth=True, allow_partial=False):
    # Count levels require repeated moments, never sum each frame's detections.
    # Only valid two-frame tabletop moments enter frames. Missing/blocked
    # tabletop moments must not increase the denominator or confirmation gate.
    support_reference = len(frames)
    need_id = max(4, math.ceil(support_reference*.35))
    need_multi = max(6, math.ceil(support_reference*.42))
    caps = {}
    for cls in range(16):
        values = [sum(r['class_id'] == cls for r in f['objects']) for f in frames]
        caps[cls] = max((n for n in range(1, 6) if sum(v >= n for v in values) >=
                        (need_id if n == 1 else need_multi)), default=0)
    tracks = []
    for moment, frame in enumerate(frames):
        used = set()
        rotation = frame['signed_rotation']
        for row in sorted(frame['objects'], key=lambda r: r['confidence'], reverse=True):
            fixed_angle = wrap_angle(row['angle']-rotation)
            px, py = row['radius']*math.cos(fixed_angle), row['radius']*math.sin(fixed_angle)
            candidates = []
            for n, t in enumerate(tracks):
                if n in used or moment-t['last_moment'] > 2:
                    continue
                last = t['rows'][-1]
                bw, bh = row['xyxy'][2]-row['xyxy'][0], row['xyxy'][3]-row['xyxy'][1]
                tw, th = last['xyxy'][2]-last['xyxy'][0], last['xyxy'][3]-last['xyxy'][1]
                ratio = min(bw*bh, tw*th)/max(bw*bh, tw*th, 1.)
                aspect = min(bw/max(bh, 1.), tw/max(th, 1.))/max(bw/max(bh, 1.), tw/max(th, 1.), .01)
                distance = math.hypot(t['px']-px, t['py']-py)
                same_class = row['class_id'] == last['class_id']
                capacities = {row['class_id'], last['class_id']} <= {10, 11}
                if ratio < .45 or aspect < .5:
                    continue
                if not same_class and not capacities and not (distance <= .08 and ratio >= .8):
                    continue
                candidates.append((distance + (0. if same_class else .03), n))
            distance, n = min(candidates, default=(float('inf'), -1))
            if distance > .28:
                tracks.append({'rows': [], 'px': px, 'py': py, 'last_moment': moment})
                n = len(tracks)-1
            track = tracks[n]
            track['rows'].append(dict(row, moment=moment))
            track['px'], track['py'] = px, py
            track['last_moment'] = moment
            used.add(n)
    decisions = []
    ambiguous = set()
    depth_removed = set()
    capacity_labels = {}
    weak_identity = set()
    water_candidates=[]
    for track_id,track in enumerate(tracks):
        is_water=all(r['class_id'] in CAPACITY_CLASSES for r in track['rows'])
        if is_water:
            observations=capacity_observations(track['rows'])
            accepted,cls,detail=choose_round_capacity(track['rows'],observations)
            best=max(track['rows'],key=lambda row:(row['class_id']==cls,row['confidence']))
            matching=[obs for obs in observations if obs[1]==cls]
            score=sum(obs[2 if cls==10 else 3] for obs in matching)/max(1,len(matching))
            d=UiDetection(cls,score,tuple(best['xyxy']))
            decision=dict(detail,class_id=cls,votes=detail['capacity_votes'],accepted=accepted,
                          observations=len(track['rows']),depth_state='UNKNOWN')
        else:accepted, cls, d, decision = select_track(track, need_id)
        decision['track_id']=track_id
        # Preserve actual observations for OCR-to-object ownership checks.
        # These are provenance only; the recognition/counting gates stay intact.
        decision['trajectory_observations']=[{
            'moment':r['moment'],'class_id':r['class_id'],'xyxy':list(r['xyxy'])
            } for r in track['rows']]
        decisions.append(decision)
        votes = Counter(r['class_id'] for r in track['rows'])
        if cls in CLASS_TRACK_CONF_OVERRIDES and len(track['rows']) >= need_id and d.confidence < CLASS_TRACK_CONF_OVERRIDES[cls]:
            weak_identity.update((r['moment'], r['class_id'], tuple(r['xyxy'])) for r in track['rows'])
        rejected_classes = []
        if use_depth:
            for class_id in {r['class_id'] for r in track['rows']} & (HARD_DEPTH_CLASSES|PRINT_DEPTH_CLASSES):
                if consistent_reject(track['rows'], mode='round', class_id=class_id):
                    rejected_classes.append(class_id)
                    depth_removed.update((r['moment'], r['class_id'], tuple(r['xyxy']))
                                         for r in track['rows'] if r['class_id']==class_id)
        decision['depth_rejected_classes'] = rejected_classes
        if all(r['class_id'] in CAPACITY_CLASSES for r in track['rows']):
            if use_depth:
                decision['water_depth_confirmed']=water_confirmed(track['rows'],'round',identity_confidence=d.confidence)
                decision['water_unknown_retained']=not decision['water_depth_confirmed'] and not rejected_classes
            for r in track['rows']:
                capacity_labels[(r['moment'], r['class_id'], tuple(r['xyxy']))] = cls if accepted else None
            if accepted:
                water_candidates.append({'track_id':track_id,'rows':track['rows'],'detection':d,
                                         'depth_rejected':bool(rejected_classes)})
            continue
        # Only veto repeated, geometrically associated class ambiguity. A
        # short/fragmented trajectory must not erase a stable class count.
        if len(votes) >= 2 and sorted(votes.values(), reverse=True)[1] >= 4 and decision['score_margin'] < .35:
            ambiguous.update((r['moment'], r['class_id'], tuple(r['xyxy'])) for r in track['rows'])
    selected = []
    pure_counts = {}
    normalized_frames = []
    for moment, frame in enumerate(frames):
        normalized = []
        for row in frame['objects']:
            key = (moment, row['class_id'], tuple(row['xyxy']))
            label = capacity_labels.get(key, row['class_id'])
            if key in ambiguous or key in weak_identity or label is None:
                continue
            updated = dict(row, class_id=label, depth_track_rejected=key in depth_removed)
            if key in capacity_labels:
                observations = capacity_observations([row])
                updated['confidence'] = sum(obs[2 if label==10 else 3] for obs in observations)/max(1,len(observations))
            normalized.append(updated)
        normalized_frames.append(normalized)
    for cls in range(16):
        if cls in CAPACITY_CLASSES:continue
        rows_by_moment = [[r for r in objects if r['class_id']==cls] for objects in normalized_frames]
        pure_counts[cls] = max((n for n in range(1, 6) if sum(len(rows) >= n for rows in rows_by_moment) >=
                     (need_id if n == 1 else need_multi)), default=0)
        # Depth only vetoes candidates of an already stable YOLO quantity.
        if use_depth:
            rows_by_moment = [[r for r in rows if not r['depth_track_rejected']] for rows in rows_by_moment]
        count = max((n for n in range(1, pure_counts[cls]+1) if sum(len(rows) >= n for rows in rows_by_moment) >=
                     (need_id if n == 1 else need_multi)), default=0)
        if count:
            representative = max((rows for rows in rows_by_moment if len(rows) >= count),
                                 key=lambda rows: sum(r['confidence'] for r in sorted(rows, key=lambda r:r['confidence'], reverse=True)[:count]))
            selected.extend(UiDetection(cls, r['confidence'], tuple(r['xyxy'])) for r in
                            sorted(representative, key=lambda r:r['confidence'], reverse=True)[:count])
    pure_water,water_diagnostics=distinct_bottles(water_candidates)
    final_water=[item for item in pure_water if not (use_depth and item['depth_rejected'])]
    selected.extend(item['detection'] for item in final_water)
    for cls in CAPACITY_CLASSES:pure_counts[cls]=sum(item['detection'].class_id==cls for item in pure_water)
    water_diagnostics['depth_rejected_track_ids']=[item['track_id'] for item in pure_water if item['depth_rejected']]
    return selected, {'tracks': decisions, 'count_caps': caps,'water_confirmation':water_diagnostics,
                      'pure_counts': pure_counts,
                      'support_reference_moments': support_reference,
                      'partial_sampling':allow_partial,
                      'count_method': 'valid_table_moment_ratios',
                      'id_min_ratio': .35, 'multi_min_ratio': .42,
                      'ambiguous_observations_removed': len(ambiguous),
                      'depth_observations_removed': len(depth_removed),
                      'id_min_moments': need_id, 'multi_min_moments': need_multi}
