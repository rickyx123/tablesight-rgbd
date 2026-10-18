"""Confirmed text-object regions exclude competing YOLO identities.

Only accepted OCR target hits participate, never arbitrary package text.
Rotation compensation associates OCR counts only. Exclusion must be bound to
an actually observed object in the OCR source frame, never a projected mask.
"""
import math
from national_competition.runtime.ocr_logic import deduplicate


def box_iou(a,b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    aa=max(0.,a[2]-a[0])*max(0.,a[3]-a[1]);bb=max(0.,b[2]-b[0])*max(0.,b[3]-b[1])
    return inter/max(1.,aa+bb-inter)


def polygon_bounds(polygon):
    return [min(p[0] for p in polygon),min(p[1] for p in polygon),
            max(p[0] for p in polygon),max(p[1] for p in polygon)]


def motion_context(evidence,mode):
    contexts={}
    if mode!='round':return contexts
    for moment in evidence.get('frames',[]):
        for sample in moment.get('samples',[]):
            if sample.get('table'):
                contexts[sample['sequence']]={'table':sample['table']['xyxy'],'rotation':moment['signed_rotation']}
    return contexts


def transform(polygon,context,inverse=False):
    a,b,c,d=context['table'];cx,cy=(a+c)/2,(b+d)/2;rx,ry=(c-a)/2,(d-b)/2
    angle=context['rotation']*(1 if inverse else -1);co,si=math.cos(angle),math.sin(angle)
    points=[]
    for px,py in polygon:
        x,y=(px,py) if inverse else ((px-cx)/rx,(py-cy)/ry)
        nx,ny=co*x-si*y,si*x+co*y
        points.append([nx*rx+cx,ny*ry+cy] if inverse else [nx,ny])
    return points


def confirmed_regions(report,contexts=None):
    if not report.get('enabled',True):return []
    contexts=contexts or {};settings=report.get('config',{})
    required=settings.get('minimum_distinct_frames',2)
    names={row['name'] for row in report.get('results',[])}
    count_limits={row['name']:int(row.get('count',1)) for row in report.get('results',[])}
    tracks=[];seen=set()
    for frame in report.get('frames',[]):
        seq=frame.get('sequence')
        if seq in seen or frame.get('status')!='OK' or not frame.get('table_valid',True):continue
        seen.add(seq);used=set()
        for hit in deduplicate([h for h in frame.get('hits',[]) if h.get('box')]):
            if hit.get('name') not in names or hit.get('confidence',0)<settings.get('minimum_text_confidence',.78):continue
            polygon=hit.get('box')
            if not polygon:continue
            context=contexts.get(seq);fixed=transform(polygon,context) if context else polygon
            box=polygon_bounds(fixed);cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            candidates=[]
            for index,track in enumerate(tracks):
                if index in used or track['name']!=hit['name'] or track['rotating']!=(context is not None):continue
                prior=track['last'];distance=math.hypot(cx-(prior[0]+prior[2])/2,cy-(prior[1]+prior[3])/2)
                radius=.4*max(box[2]-box[0],box[3]-box[1],prior[2]-prior[0],prior[3]-prior[1])
                if box_iou(box,prior)>=.25 or distance<=radius:candidates.append((distance,index))
            if candidates:_,index=min(candidates)
            else:
                index=len(tracks);tracks.append({'name':hit['name'],'rotating':context is not None,'observations':[]})
            track=tracks[index];track['last']=box;track['observations'].append(
                {'sequence':seq,'box':polygon,'fixed_polygon':fixed,'confidence':hit['confidence'],
                 'appearance_signature':hit.get('appearance_signature')})
            used.add(index)
    regions=[]
    for track in tracks:
        observations=track['observations']
        if len({o['sequence'] for o in observations})<required:continue
        regions.append({'name':track['name'],'rotating':track['rotating'],'frame_support':len(observations),
                        'count_limit':count_limits[track['name']],
                        'observations':observations})
    return regions


def same_entity(yolo_box,text_polygon):
    a,b,c,d=polygon_bounds(text_polygon);text_area=max(1.,(c-a)*(d-b))
    padding=min(10.,max(4.,.10*max(c-a,d-b)))
    x1,y1,x2,y2=yolo_box;area=max(1.,(x2-x1)*(y2-y1))
    intersection=max(0.,min(x2,c)-max(x1,a))*max(0.,min(y2,d)-max(y1,b))
    expanded=max(0.,min(x2,c+padding)-max(x1,a-padding))*max(0.,min(y2,d+padding)-max(y1,b-padding))
    # A neighbouring object touching only a text corner is not the same item.
    return expanded/area>=.55 or (intersection/text_area>=.65 and area<=12*text_area)


def _same_kind(left,right):
    return left==right or {left,right}<={10,11}


def _label_owner(sample,polygon):
    """A unique physical box must contain most of the actual printed region."""
    import cv2
    import numpy as np
    try:
        points=np.asarray(polygon,dtype=np.float32)
        if points.shape!=(4,2) or not np.isfinite(points).all():return None
        area=abs(cv2.contourArea(points))
        if area<=1:return None
        owners=[]
        for row in sample.get('objects',[]):
            a,b,c,d=row['xyxy']
            if row['class_id']==16 or c<=a or d<=b:continue
            rectangle=np.float32([[a,b],[c,b],[c,d],[a,d]])
            overlap,_=cv2.intersectConvexConvex(rectangle,points)
            if overlap/area>=.65:owners.append(row)
        return owners[0] if len(owners)==1 else None
    except (ValueError,TypeError,cv2.error):
        return None


def _exclude_round(detections,regions,evidence):
    """Bind text to its observed object track; uncertainty never erases counts."""
    evidence=evidence or {};moments=evidence.get('frames',[])
    samples={s['sequence']:s for f in moments for s in f.get('samples',[])}
    tracks={};invalid=set();has_provenance=False
    for key in ('decisions','pure_yolo_decisions'):
        for track in evidence.get(key,{}).get('tracks',[]):
            rows=track.get('trajectory_observations')
            if rows is None:continue
            has_provenance=True
            tid=track['track_id']
            if tid in tracks and tracks[tid]!=rows:invalid.add(tid)
            tracks[tid]=rows
    tracks={tid:rows for tid,rows in tracks.items() if tid not in invalid}
    candidates=[]
    for detection in detections:
        cls=detection['class_id'] if isinstance(detection,dict) else detection.class_id
        box=detection['xyxy'] if isinstance(detection,dict) else detection.xyxy
        matching=[tid for tid,rows in tracks.items() if any(
            _same_kind(r['class_id'],cls) and box_iou(box,r['xyxy'])>=.98 for r in rows)]
        candidates.append((cls,box,matching))
    used=set();budgets={};bindings={}
    for region in regions:
        name=region['name'];limit=max(0,int(region.get('count_limit',1)))
        budgets[name]=min(budgets.get(name,limit),limit)
        options={}
        for observation in region['observations']:
            seq=observation['sequence'];sample=samples.get(seq)
            if sample is None:continue
            owner=_label_owner(sample,observation['box'])
            if owner is None:continue
            for index,(cls,box,matching) in enumerate(candidates):
                if index in used:continue
                if has_provenance:
                    # A representative box must identify exactly one existing
                    # trajectory. Matching coordinates alone across turns do not.
                    if len(matching)!=1:continue
                    tid=matching[0];actual=[]
                    for row in tracks[tid]:
                        moment=row['moment']
                        if not 0<=moment<len(moments):continue
                        if not any(s['sequence']==seq for s in moments[moment].get('samples',[])):continue
                        matches=sorted([(box_iou(row['xyxy'],r['xyxy']),r) for r in sample.get('objects',[])
                                        if _same_kind(row['class_id'],r['class_id'])],key=lambda entry:entry[0],reverse=True)
                        if not matches or matches[0][0]<.50:continue
                        if len(matches)>1 and matches[0][0]-matches[1][0]<.10:continue
                        actual.append(matches[0][1])
                    if any(r is owner for r in actual):
                        options[index]={'source_sequence':seq,'source_track_id':tid,'association':'observed_object_track'}
                elif _same_kind(cls,owner['class_id']) and box_iou(box,owner['xyxy'])>=.98:
                    # Historical evidence without provenance can only veto a
                    # box directly present in the very same OCR source frame.
                    occurrences={s['sequence'] for s in samples.values() if any(
                        _same_kind(cls,r['class_id']) and box_iou(box,r['xyxy'])>=.98 for r in s.get('objects',[]))}
                    if occurrences=={seq}:
                        options[index]={'source_sequence':seq,'source_track_id':None,'association':'same_source_frame'}
        # A broad/ambiguous text area cannot remove several physical objects.
        if len(options)!=1 or budgets[name]<=0:continue
        index,detail=next(iter(options.items()));used.add(index);budgets[name]-=1
        bindings[index]=(region,detail)
    kept=[];removed=[]
    for index,detection in enumerate(detections):
        if index not in bindings:kept.append(detection);continue
        region,detail=bindings[index];cls,box,_=candidates[index]
        removed.append({'class_id':cls,'xyxy':list(box),'ocr_name':region['name'],
                        'reason':'CONFIRMED_OCR_OBJECT_OWNERSHIP','frame_support':region['frame_support'],**detail})
    return kept,removed


def exclude_detections(detections,regions,contexts=None,evidence=None):
    if (contexts or (evidence is not None and 'frames' in evidence)
            or any(r.get('rotating') for r in regions)):
        return _exclude_round(detections,regions,evidence)
    contexts=contexts or {};kept=[];removed=[]
    for detection in detections:
        box=detection['xyxy'] if isinstance(detection,dict) else detection.xyxy
        cls=detection['class_id'] if isinstance(detection,dict) else detection.class_id
        match=None
        for region in regions:
            if not region['rotating']:
                polygon=region['observations'][-1]['box']
                if same_entity(box,polygon):match=region;break
        if match is None:kept.append(detection)
        else:removed.append({'class_id':cls,'xyxy':list(box),'ocr_name':match['name'],
                             'reason':'CONFIRMED_OCR_SAME_ENTITY','frame_support':match['frame_support']})
    return kept,removed


def appearance_signature(image,polygon):
    import cv2
    import numpy as np
    a,b,c,d=polygon_bounds(polygon);h,w=image.shape[:2]
    crop=image[max(0,int(b)):min(h,int(math.ceil(d))),max(0,int(a)):min(w,int(math.ceil(c)))]
    if crop.size==0:return None
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    patch=cv2.resize(gray,(16,16),interpolation=cv2.INTER_AREA).astype(np.float32)
    deviation=float(patch.std())
    if deviation<8:return None
    return np.round((patch-patch.mean())/deviation,3).reshape(-1).tolist()


def live_regions(image,regions):
    """Never hide a replacement object with an old static-table OCR mask."""
    verified=[]
    for region in regions:
        if region['rotating']:continue
        observation=region['observations'][-1];signature=observation.get('appearance_signature')
        current=appearance_signature(image,observation['box'])
        if signature is None or current is None:continue
        agreement=sum(a*b for a,b in zip(signature,current))/len(signature)
        if agreement>=.80:verified.append(region)
    return verified
