"""Round-only bottle identity, capacity preference, and distinct-track counting."""
from collections import Counter

MIN_WATER_FRAMES=8
MIN_WATER_MOMENTS=4
MIN_LARGE_FRAMES=4
MIN_LARGE_MOMENTS=2
LARGE_VOTE_SHARE=.45
MIN_LARGE_MEAN=.60

def choose_capacity(rows,observations):
    votes=Counter(obs[1] for obs in observations)
    scores={10:sum(obs[2] for obs in observations),11:sum(obs[3] for obs in observations)}
    moments={row['moment'] for row in rows}
    large_moments={row['moment'] for row in rows if any(obs[1]==11 for obs in row.get('capacity_observations',
        [(row.get('sequence',row['moment']),row['class_id'],0.,0.)]))}
    enough=len(observations)>=MIN_WATER_FRAMES and len(moments)>=MIN_WATER_MOMENTS
    large_mean=sum(obs[3] for obs in observations if obs[1]==11)/max(1,votes[11])
    large_share=votes[11]/max(1,len(observations))
    prefer_large=(votes[11]>=MIN_LARGE_FRAMES and len(large_moments)>=MIN_LARGE_MOMENTS
                  and large_share>=LARGE_VOTE_SHARE and large_mean>=MIN_LARGE_MEAN)
    winner=11 if prefer_large else max(scores,key=scores.get)
    margin=(scores[winner]-scores[21-winner])/max(1e-9,sum(scores.values()))
    accepted=enough and (prefer_large or (winner==10 and votes[10]>=4 and margin>=.35))
    reason=('WATER_SUPPORT_INSUFFICIENT' if not enough else 'ROUND_550_VOTE_PREFERENCE' if prefer_large
            else 'CLEAR_350_SCORE_LEAD' if accepted else 'CAPACITY_UNCONFIRMED')
    return accepted,winner,{'capacity_votes':dict(votes),'capacity_scores':scores,'capacity_score_margin':margin,
        'score_margin':margin,'capacity_min_frames':MIN_WATER_FRAMES,'capacity_required_margin':.35,
        'water_distinct_frames':len(observations),'water_distinct_moments':len(moments),
        'water_min_moments':MIN_WATER_MOMENTS,'large_vote_share':large_share,'large_required_share':LARGE_VOTE_SHARE,
        'large_mean_confidence':large_mean,'large_support_moments':len(large_moments),'capacity_reason':reason}

def independently_visible(left,right):
    """One voted keymoment implies two raw frames; boxes must be separate."""
    by_moment={row['moment']:row for row in right['rows']}
    for row in left['rows']:
        other=by_moment.get(row['moment'])
        if other is None:continue
        a,b,c,d=row['xyxy'];x,y,z,w=other['xyxy']
        overlap=max(0.,min(c,z)-max(a,x))*max(0.,min(d,w)-max(b,y))
        smaller=max(1.,min((c-a)*(d-b),(z-x)*(w-y)))
        if overlap/smaller<.30:return True
    return False

def distinct_bottles(candidates):
    """Choose a mutually distinct set; non-coexisting fragments cannot add count."""
    candidates=sorted(candidates,key=lambda item:len(item['rows']),reverse=True)[:24]
    edges={i:set() for i in range(len(candidates))}
    for i,left in enumerate(candidates):
        for j in range(i):
            if independently_visible(left,candidates[j]):edges[i].add(j);edges[j].add(i)
    best=[];best_support=0
    def search(chosen,remaining):
        nonlocal best,best_support
        support=sum(len(candidates[i]['rows']) for i in chosen)
        if len(chosen)>len(best) or (len(chosen)==len(best) and support>best_support):
            best=list(chosen);best_support=support
        while remaining:
            if len(chosen)+len(remaining)<len(best):return
            node=remaining[0];remaining=remaining[1:]
            search(chosen+[node],[i for i in remaining if i in edges[node]])
    search([],list(range(len(candidates))))
    return [candidates[i] for i in best],{'confirmed_track_candidates':len(candidates),
        'selected_track_ids':[candidates[i]['track_id'] for i in best],
        'distinct_track_pairs':[[candidates[i]['track_id'],candidates[j]['track_id']] for i in edges for j in edges[i] if j<i],
        'count_method':'confirmed_water_tracks_with_covisibility'}
