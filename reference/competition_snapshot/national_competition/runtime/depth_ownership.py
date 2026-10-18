"""Assign shared RGB-D void components before testing overlapping water boxes.

An IR void is UNKNOWN material evidence, never proof of a real bottle. This
only prevents a second loose box from borrowing the first object's same void.
No capacity, object count, scene coordinates, or expected quantities are used.
"""
import cv2
import numpy as np
import math


def foreign_water_voids(depth,plane,rows,gap_function):
    water=[row for row in rows if row['class_id'] in (10,11)]
    if len(water)<2:
        # No shared void exists, but retain the same geometric masking path.
        # None would incorrectly restore the legacy whole-box occlusion rule.
        return {id(row):np.zeros(depth.shape,bool) for row in water},[]
    h,w=depth.shape
    left=max(0,math.ceil(min(row['xyxy'][0] for row in water)))
    top=max(0,math.ceil(min(row['xyxy'][1] for row in water)))
    right=min(w,math.ceil(max(row['xyxy'][2] for row in water)))
    bottom=min(h,math.ceil(max(row['xyxy'][3] for row in water)))
    foreign={id(row):np.zeros((h,w),bool) for row in water}
    if right<=left or bottom<=top:return foreign,[]
    yy,xx=np.mgrid[top:bottom,left:right];union=np.zeros(xx.shape,bool)
    for row in water:
        a,b,c,d=row['xyxy']
        x1,x2=max(left,math.ceil(a))-left,min(right,math.ceil(c))-left
        y1,y2=max(top,math.ceil(b))-top,min(bottom,math.ceil(d))-top
        if x2>x1 and y2>y1:union[y1:y2,x1:x2]=True
    cropped=depth[top:bottom,left:right]
    valid=(cropped>250)&(cropped<5000)&np.isfinite(cropped)
    behind=gap_function(cropped,plane,xx,yy)<-30
    void=union&(~valid|behind)
    candidates_info=[]
    for row in water:
        a,b,c,d=row['xyxy']
        x1,x2=max(0,math.ceil(a+.14*(c-a))),min(w,math.ceil(c-.14*(c-a)))
        y1,y2=max(0,math.ceil(b+.10*(d-b))),min(h,math.ceil(d-.10*(d-b)))
        region=void[max(0,y1-top):max(0,y2-top),max(0,x1-left):max(0,x2-left)]
        purity=float(region.mean()) if region.size else 0.
        candidates_info.append((row,a,b,c,d,purity))
    number,labels,stats,_=cv2.connectedComponentsWithStats(void.astype(np.uint8),8)
    assigned=[]
    for component in range(1,number):
        area=int(stats[component,cv2.CC_STAT_AREA])
        if area<32:continue
        comp_left,comp_top,width,height=stats[component,:4]
        local=labels[comp_top:comp_top+height,comp_left:comp_left+width]
        ys,xs=np.where(local==component)
        ys+=top+comp_top;xs+=left+comp_left;candidates=[]
        for row,a,b,c,d,purity in candidates_info:
            inside=(xs>=a)&(xs<c)&(ys>=b)&(ys<d)
            coverage=float(inside.sum()/area)
            score=coverage*(.4+.6*purity)
            candidates.append((score,coverage,purity,id(row)))
        candidates.sort(reverse=True)
        score,coverage,purity,owner=candidates[0]
        margin=score-candidates[1][0]
        if coverage<.75 or purity<.25 or margin<.12:continue
        mask=local==component
        for row in water:
            if id(row)!=owner:
                foreign[id(row)][top+comp_top:top+comp_top+height,left+comp_left:left+comp_left+width] |= mask
        owner_row=next(row for row in water if id(row)==owner)
        assigned.append({'component_points':area,'owner_box':owner_row['xyxy'],
                         'coverage':coverage,'core_void_fraction':purity,'score_margin':margin})
    return foreign,assigned
