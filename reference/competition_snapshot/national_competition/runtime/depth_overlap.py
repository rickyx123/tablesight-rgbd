"""Local depth masks follow neighbouring foreground beyond a jittering box."""
import cv2
import numpy as np


def foreign_edge_mask(depth,plane,box,others,gap_function):
    h,w=depth.shape;a,b,c,d=box
    near=[other for other in others if other[2]>=a-24 and other[0]<=c+24 and other[3]>=b-24 and other[1]<=d+24]
    mask=np.zeros((h,w),bool)
    # An occluder can disappear from YOLO for a frame. A depth component
    # predominantly outside this box still cannot provide its own PASS.
    x1,x2=max(0,int(a)-24),min(w,int(c)+25)
    y1,y2=max(0,int(b)-24),min(h,int(d)+25)
    yy,xx=np.mgrid[y1:y2,x1:x2];z=depth[y1:y2,x1:x2]
    if z.size:
        delta=gap_function(z,plane,xx,yy)
        raised=(z>250)&(z<5000)&np.isfinite(z)&(delta>max(8.,2*plane['p90_error_mm']))
        _,labels,stats,centres=cv2.connectedComponentsWithStats(raised.astype(np.uint8),8)
        inside=(xx>=a)&(xx<c)&(yy>=b)&(yy<d)
        points=np.bincount(labels[inside],minlength=len(stats))
        outside=stats[:,cv2.CC_STAT_AREA]-points
        cx=centres[:,0]+x1;cy=centres[:,1]+y1
        selected=(points>=12)&(outside>=2*points)&~((cx>=a)&(cx<c)&(cy>=b)&(cy<d))
        selected[0]=False
        mask[y1:y2,x1:x2] |= selected[labels]
    for oa,ob,oc,od in near:
        # Similar duplicate boxes cannot claim each other's complete object.
        inter=max(0.,min(c,oc)-max(a,oa))*max(0.,min(d,od)-max(b,ob))
        area=max(1.,(c-a)*(d-b));other_area=max(1.,(oc-oa)*(od-ob))
        if inter/max(1.,area+other_area-inter)>.65:continue
        margin=min(24,max(8,int(.45*min(oc-oa,od-ob))))
        x1,x2=max(0,int(oa)-margin),min(w,int(oc)+margin+1)
        y1,y2=max(0,int(ob)-margin),min(h,int(od)+margin+1)
        yy,xx=np.mgrid[y1:y2,x1:x2];z=depth[y1:y2,x1:x2]
        if z.size==0:continue
        delta=gap_function(z,plane,xx,yy)
        valid=(z>250)&(z<5000)&np.isfinite(z)
        raised=valid&(delta>max(8.,2*plane['p90_error_mm']))
        core=(xx>=oa+.20*(oc-oa))&(xx<oc-.20*(oc-oa))&(yy>=ob+.20*(od-ob))&(yy<od-.20*(od-ob))
        seed=core&raised
        if seed.sum()<max(12,int(core.sum()*.25)):continue
        centre_height=float(np.median(delta[seed]))
        same_surface=raised&(np.abs(delta-centre_height)<=max(12.,2*plane['p90_error_mm']))
        count,labels,stats,_=cv2.connectedComponentsWithStats(same_surface.astype(np.uint8),8)
        component_ids=np.unique(labels[seed&same_surface]);component_ids=component_ids[component_ids>0]
        if not len(component_ids):continue
        belonging=np.isin(labels,component_ids)
        mask[y1:y2,x1:x2] |= belonging
    return mask


def has_interior_structure(raised,xx,yy,box,scale,water=False):
    a,b,c,d=box;bw,bh=c-a,d-b
    core=(xx>=a+.25*bw)&(xx<c-.25*bw)&(yy>=b+.25*bh)&(yy<d-.25*bh)
    if (raised&core).sum()>=max(4*scale,.15*core.sum()):return True
    # An upright transparent bottle may have only its own cap visible.
    cap=(xx>=a+.20*bw)&(xx<c-.20*bw)&(yy>=b)&(yy<b+.20*bh)
    return bool(water and bh>=1.2*bw and (raised&cap).sum()>=16*scale)
