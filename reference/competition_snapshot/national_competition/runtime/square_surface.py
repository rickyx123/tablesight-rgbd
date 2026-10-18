"""Locate a perspective tabletop from its own RGB-D support, not box ratios."""
import math
import struct
import time
from itertools import combinations

import cv2
import numpy as np

from national_competition.runtime.depth_validation import fit_plane, gaps


def _localization_plane(depth,box,boxes):
    """Allow measured far-range noise only for locating the rim, not item depth."""
    x1,y1,x2,y2=box;cx,cy=(x1+x2)/2,(y1+y2)/2;rx,ry=(x2-x1)/2,(y2-y1)/2
    yy,xx=np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2];z=depth[::2,::2]
    valid=(xx>x1+.12*(x2-x1))&(xx<x2-.12*(x2-x1))&(yy>y1+.12*(y2-y1))&(yy<y2-.12*(y2-y1))
    valid&=(z>250)&(z<5000)&np.isfinite(z)
    for a,b,c,d in boxes:valid&=~((xx>=a-3)&(xx<=c+3)&(yy>=b-3)&(yy<=d+3))
    x,y,z=xx[valid],yy[valid],z[valid]
    if len(z)<120:return None
    if len(z)>900:
        indices=np.linspace(0,len(z)-1,900).astype(int);x,y,z=x[indices],y[indices],z[indices]
    matrix=np.column_stack(((x-cx)/rx,(y-cy)/ry,np.ones(len(z))))
    rng=np.random.default_rng(17);chosen=np.array([rng.choice(len(z),3,replace=False) for _ in range(60)])
    try:coefficients=np.linalg.solve(matrix[chosen],(1/z)[chosen][...,None])[...,0]
    except np.linalg.LinAlgError:
        coefficients=np.full((60,3),np.nan)
        for i,indices in enumerate(chosen):
            try:coefficients[i]=np.linalg.solve(matrix[indices],1/z[indices])
            except np.linalg.LinAlgError:pass
    prediction=matrix@coefficients.T
    inliers=(prediction>0)&(np.abs(1/np.maximum(prediction,1e-6)-z[:,None])<=12.)
    best=inliers[:,np.argmax(inliers.sum(axis=0))]
    if best.sum()<100 or best.mean()<.65:return None
    quadrants=set(zip(x[best]>=cx,y[best]>=cy))
    if len(quadrants)<3 or np.ptp(x[best])<.4*(x2-x1) or np.ptp(y[best])<.3*(y2-y1):return None
    upper=y<cy
    if upper.sum()<30 or best[upper].mean()<.5:return None
    coeff=np.linalg.lstsq(matrix[best],1/z[best],rcond=None)[0]
    p90=float(np.percentile(np.abs(1/(matrix[best]@coeff)-z[best]),90))
    if p90>12.:return None
    return {'coeff':coeff.tolist(),'cx':cx,'cy':cy,'rx':rx,'ry':ry,
            'inliers':int(best.sum()),'fraction':float(best.mean()),'p90_error_mm':p90}


def ordered_quad(points):
    points=np.asarray(points,dtype=np.float32).reshape(4,2)
    center=points.mean(axis=0)
    angles=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])
    points=points[np.argsort(angles)]
    return np.roll(points,-int(np.argmin(points.sum(axis=1))),axis=0)


def _mask(shape,quad):
    mask=np.zeros(shape,np.uint8)
    cv2.fillConvexPoly(mask,np.rint(quad).astype(np.int32),255)
    return mask>0


def _quad_from_hull(hull):
    perimeter=cv2.arcLength(hull,True);area=cv2.contourArea(hull)
    outline=None
    for epsilon in (.015,.02,.025,.03,.04):
        approximation=cv2.approxPolyDP(hull,epsilon*perimeter,True)
        if len(approximation)==4 and cv2.contourArea(approximation)>=.88*area:
            return ordered_quad(approximation[:,0,:])
        if outline is None and 5<=len(approximation)<=8:outline=approximation[:,0,:].astype(float)
    if outline is None:return None
    # A bottle can hide a corner. Extend measured adjacent straight sides
    # across that gap, with a strict area bound; never replace it by a box.
    candidates=[]
    for indices in combinations(range(len(outline)),4):
        lines=[(outline[i],outline[(i+1)%len(outline)]-outline[i]) for i in indices]
        points=[]
        for i in range(4):
            a,u=lines[i-1];b,v=lines[i]
            try:t=np.linalg.solve(np.column_stack((u,-v)),b-a)[0]
            except np.linalg.LinAlgError:break
            points.append(a+t*u)
        if len(points)!=4:continue
        quad=ordered_quad(points)
        if not np.isfinite(quad).all() or not cv2.isContourConvex(quad.reshape(-1,1,2)):continue
        new_area=cv2.contourArea(quad)
        if not 1.<=new_area/max(1.,area)<=1.25:continue
        if any(cv2.pointPolygonTest(quad,tuple(map(float,p)),True)<-8 for p in hull[:,0,:]):continue
        edges=np.roll(quad,-1,axis=0)-quad
        lengths=np.linalg.norm(edges,axis=1)
        if lengths.min()<20:continue
        units=edges/lengths[:,None]
        if min(abs(np.dot(units[0],units[2])),abs(np.dot(units[1],units[3])))<math.cos(math.radians(35)):continue
        candidates.append((new_area,quad))
    return min(candidates,key=lambda item:item[0])[1] if candidates else None


def _rim_contacts(depth,plane,quad,objects):
    """Keep a standing item's actual base at a noisy/occluded rim.

    Only side/front registration errors may receive this allowance, backed
    by the item's own raised pixels. The rear edge never receives it.
    """
    contacts=[]
    for obj in objects:
        if obj.class_id==16:continue
        a,b,c,d=map(float,obj.xyxy);px=(a+c)/2
        if cv2.pointPolygonTest(quad,(px,b+.94*(d-b)),True)>=-2:continue
        # Rear membership is a physical footprint boundary. A raised or
        # stacked object behind it cannot borrow another support surface.
        edge=quad[1]-quad[0]
        signed=(edge[0]*(b+.94*(d-b)-quad[0,1])-edge[1]*(px-quad[0,0]))/max(1.,float(np.linalg.norm(edge)))
        if signed<0:continue
        if cv2.pointPolygonTest(quad,(px,d),True)<-6:continue
        yy,xx=np.mgrid[max(0,int(b+.15*(d-b))):min(depth.shape[0],int(d-.15*(d-b))+1),
                       max(0,int(a+.15*(c-a))):min(depth.shape[1],int(c-.15*(c-a))+1)]
        if xx.size==0:continue
        z=depth[yy,xx];valid=(z>250)&(z<5000)&np.isfinite(z)
        for other in objects:
            if other is obj or other.class_id==16:continue
            oa,ob,oc,od=other.xyxy
            valid&=~((xx>=oa)&(xx<=oc)&(yy>=ob)&(yy<=od))
        if valid.sum()>=30 and valid.mean()>=.60:
            delta=gaps(z,plane,xx,yy)[valid]
            if np.median(delta)<20 or (delta>15).mean()<.55:continue
            basis='measured_object_at_rim'
        else:continue
        contacts.append({'class_id':obj.class_id,'xyxy':list(obj.xyxy),'basis':basis})
    return contacts


def _rear_edge(rgb,depth,plane,box,objects,quad):
    """Fit the visible rear table edge, excluding every detected object.

    Coplanar tops of stacked background items are not tabletop boundary
    evidence. A rear line also needs table support on its near side and a
    measured depth or image transition on its far side.
    """
    x1,y1,x2,y2=box;bw,bh=x2-x1,y2-y1
    a,b=max(0,int(x1+.06*bw)),max(0,int(y1-.05*bh))
    c,d=min(depth.shape[1],int(x2-.06*bw)+1),min(depth.shape[0],int(y1+.65*bh)+1)
    yy,xx=np.mgrid[b:d,a:c];z=depth[b:d,a:c]
    tol=min(18.,max(12.,1.5*plane['p90_error_mm']))
    flat=(z>250)&(z<5000)&(np.abs(gaps(z,plane,xx,yy))<=tol)
    measured_flat=flat.copy()
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY) if rgb is not None else None
    if gray is not None and flat.sum()>=200:
        level=float(np.percentile(gray[b:d,a:c][flat],75))-30.
        if level>=60:
            near=cv2.dilate(flat.astype(np.uint8),np.ones((15,15),np.uint8))>0
            flat=near&(gray[b:d,a:c]>=level)
    for obj in objects:
        if obj.class_id==16:continue
        oa,ob,oc,od=obj.xyxy
        flat&=~((xx>=oa)&(xx<=oc)&(yy>=ob)&(yy<=od))
    runs=cv2.boxFilter(flat.astype(np.float32),-1,(1,3),normalize=False,borderType=cv2.BORDER_CONSTANT)>=2
    points=[]
    for ix in range(0,c-a,2):
        valid=np.flatnonzero(runs[:,ix])
        if len(valid):points.append((a+ix,b+int(valid[0])))
    if len(points)<12:return quad,False
    points=np.asarray(points,float);xs,ys=points[:,0],points[:,1]
    rng=np.random.default_rng(23);choices=[]
    for _ in range(90):
        i,j=rng.choice(len(points),2,replace=False)
        if abs(xs[j]-xs[i])<.35*bw:continue
        slope=(ys[j]-ys[i])/(xs[j]-xs[i]);intercept=ys[i]-slope*xs[i]
        if abs(slope)>.35:continue
        good=np.abs(ys-slope*xs-intercept)<=2.
        if good.sum()<max(6,.06*bw) or np.ptp(xs[good])<.45*bw:continue
        slope,intercept=np.linalg.lstsq(np.column_stack((xs[good],np.ones(good.sum()))),ys[good],rcond=None)[0]
        behind=measured_flat&(yy<slope*xx+intercept-3.)
        behind_y,behind_x=np.nonzero(behind)
        # An occluder's front boundary is not the table's back boundary:
        # a broad visible part of the same tabletop cannot lie behind it.
        if len(behind_x)>=max(20,.03*bw*bh) and np.ptp(behind_x)>.3*bw:continue
        sx=np.arange(int(xs[good].min()),int(xs[good].max())+1,2)
        sy=np.rint(slope*sx+intercept).astype(int)
        near_y,far_y=sy+2,sy-2
        allowed=(far_y>=0)&(near_y<depth.shape[0])
        for obj in objects:
            if obj.class_id==16:continue
            oa,ob,oc,od=obj.xyxy
            allowed&=~((sx>=oa)&(sx<=oc)&(near_y>=ob)&(far_y<=od))
        sx=sx[allowed];near_y=near_y[allowed];far_y=far_y[allowed]
        if len(sx)<6:continue
        nz,fz=depth[near_y,sx],depth[far_y,sx]
        nf=(nz>250)&(nz<5000)&(np.abs(gaps(nz,plane,sx,near_y))<=tol)
        ff=(fz>250)&(fz<5000)&(np.abs(gaps(fz,plane,sx,far_y))<=tol)
        depth_jump=((fz>250)&(fz<5000)&(np.abs(gaps(fz,plane,sx,far_y))>35)).mean()
        color_jump=float(np.median(np.abs(gray[near_y,sx].astype(float)-gray[far_y,sx]))) if gray is not None else 0.
        if (nf.mean()<.5 and color_jump<30) or ff.mean()>.3:continue
        if depth_jump<.3 and color_jump<18:continue
        choices.append((good.sum()+.1*np.ptp(xs[good]),slope,intercept))
    if not choices:return quad,False
    _,slope,intercept=max(choices)
    new=quad.copy().astype(float)
    for index,other in ((0,3),(1,2)):
        point=quad[index];vector=quad[other]-point
        denominator=vector[1]-slope*vector[0]
        if abs(denominator)<1e-6:return quad,False
        t=(slope*point[0]+intercept-point[1])/denominator
        new[index]=point+t*vector
    new=new.astype(np.float32)
    if not cv2.isContourConvex(new.reshape(-1,1,2)):return quad,False
    if np.max(np.linalg.norm(new[:2]-quad[:2],axis=1))>max(20.,.5*bh):return quad,False
    return new,True


def _refine_rim(rgb,quad,support):
    """Refine visible straight edges; occluded edges keep their depth estimate."""
    edges=cv2.Canny(cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY),60,140)
    perimeter=cv2.arcLength(quad.reshape(-1,1,2),True)
    segments=cv2.HoughLinesP(edges,1,np.pi/180,25,
                           minLineLength=max(30,int(perimeter*.045)),maxLineGap=10)
    if segments is None:return quad,False
    lines=[]
    for a,b in zip(quad,np.roll(quad,-1,axis=0)):
        vector=b-a;length=float(np.linalg.norm(vector));direction=vector/length
        choices=[]
        for segment in segments[:,0]:
            c,d=segment[:2].astype(float),segment[2:].astype(float)
            delta=d-c;span=float(np.linalg.norm(delta))
            if span<max(30.,length*.25):continue
            alignment=abs(float(np.dot(direction,delta/span)))
            if alignment<math.cos(math.radians(12)):continue
            projection=np.dot(np.stack((c-a,d-a)),direction)
            overlap=max(0.,min(length,max(projection))-max(0.,min(projection)))
            if overlap<length*.25:continue
            normal=np.array([-direction[1],direction[0]])
            distance=float(np.abs(np.dot(np.stack((c-a,d-a)),normal)).mean())
            if distance>10.:continue
            # An object's printed line is not a rim: plane support must be
            # on its inward side and disappear on the outward side.
            inward=normal if np.dot(quad.mean(axis=0)-(c+d)/2,normal)>0 else -normal
            samples=c[None,:]+np.linspace(.1,.9,25)[:,None]*(d-c)[None,:]
            def supported(offset):
                points=np.rint(samples+offset*inward).astype(int)
                valid=(points[:,0]>=0)&(points[:,0]<support.shape[1])&(points[:,1]>=0)&(points[:,1]<support.shape[0])
                return float(support[points[valid,1],points[valid,0]].mean()) if valid.any() else 0.
            inside,outside=supported(8.),supported(-8.)
            if inside<.45 or outside>.35 or inside-outside<.35:continue
            choices.append((distance+20*(1-alignment)-.01*overlap,c,d))
        if choices:
            _,c,d=min(choices,key=lambda item:item[0]);lines.append((c,d-c))
        else:lines.append((a,vector))
    refined=[]
    for i in range(4):
        a,u=lines[i-1];b,v=lines[i]
        try:t=np.linalg.solve(np.column_stack((u,-v)),b-a)[0]
        except np.linalg.LinAlgError:return quad,False
        refined.append(a+t*u)
    refined=np.asarray(refined,np.float32)
    if not np.isfinite(refined).all() or not cv2.isContourConvex(refined.reshape(-1,1,2)):
        return quad,False
    if np.max(np.linalg.norm(refined-quad,axis=1))>18:return quad,False
    area=cv2.contourArea(quad);new_area=cv2.contourArea(refined)
    if not .88<=new_area/max(1.,area)<=1.16:return quad,False
    original=_mask(support.shape,quad)&support
    if (original&_mask(support.shape,refined)).sum()<.94*max(1,original.sum()):
        return quad,False
    return refined,True


def surface_from_arrays(rgb,depth,raw_table,objects,plane_hint=None):
    if raw_table is None or depth is None or depth.shape!=(480,640):return None
    x1,y1,x2,y2=map(float,raw_table.xyxy);bw,bh=x2-x1,y2-y1
    if min(bw,bh)<30:return None
    boxes=[d.xyxy for d in objects if d.class_id!=16]
    plane=fit_plane(depth,raw_table.xyxy,boxes,'square')
    if plane is None:plane=fit_plane(depth,raw_table.xyxy,boxes,'square_center')
    if plane is None:plane=_localization_plane(depth,raw_table.xyxy,boxes)
    if plane is None and plane_hint is not None:plane=plane_hint
    if plane is None:return None
    h,w=depth.shape
    a,b=max(0,int(x1-.2*bw)),max(0,int(y1-.2*bh))
    c,d=min(w,int(x2+.2*bw)+1),min(h,int(y2+.2*bh)+1)
    yy,xx=np.mgrid[b:d,a:c];z=depth[b:d,a:c]
    tolerance=min(18.,max(12.,1.5*plane['p90_error_mm']))
    support=(z>250)&(z<5000)&np.isfinite(z)&(np.abs(gaps(z,plane,xx,yy))<=tolerance)
    closed=cv2.morphologyEx(support.astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        if cv2.contourArea(contour)<.2*bw*bh:continue
        hull=cv2.convexHull(contour)
        hx,hy,hw,hh=cv2.boundingRect(hull)
        # A cut-off support patch must not be extrapolated into a bigger table.
        if hx<=1 or hy<=1 or hx+hw>=c-a-1 or hy+hh>=d-b-1:continue
        quad=_quad_from_hull(hull)
        if quad is None:continue
        area=cv2.contourArea(quad)
        if not .35<=area/(bw*bh)<=1.3:continue
        if area/max(1.,cv2.contourArea(hull))<.88:continue
        inside=_mask(support.shape,quad)
        coverage=float((inside&support).sum()/max(1,inside.sum()))
        if coverage<.30:continue
        full=quad+np.float32([a,b]);cx,cy=full.mean(axis=0)
        if abs(cx-(x1+x2)/2)>.2*bw or abs(cy-(y1+y2)/2)>.2*bh:continue
        full_support=np.zeros(depth.shape,bool);full_support[b:d,a:c]=support
        refined=False
        if rgb is not None and rgb.shape[:2]==depth.shape:
            full,refined=_refine_rim(rgb,full,full_support)
        full,rear_refined=_rear_edge(rgb,depth,plane,raw_table.xyxy,objects,full)
        # Occluded sides may be extended only within the detector's measured
        # envelope. Reject a diagonal object edge masquerading as a table side.
        if (full[:,0].min()<x1-max(6.,.05*bw) or full[:,0].max()>x2+max(6.,.05*bw)
                or full[:,1].min()<y1-max(6.,.05*bh) or full[:,1].max()>y2+max(6.,.05*bh)):
            continue
        if abs(float(full[0,1]-full[1,1]))>.30*bh:continue
        candidates.append({'polygon':full.astype(float).tolist(),
            'xyxy':[float(full[:,0].min()),float(full[:,1].min()),
                    float(full[:,0].max()),float(full[:,1].max())],
            'source':'same_frame_depth_rgb' if refined else 'same_frame_depth',
            'rear_edge_verified':rear_refined,
            'rim_contacts':_rim_contacts(depth,plane,full,objects),
            'support_fraction':coverage,'plane_inliers':plane['inliers'],
            'plane_p90_error_mm':plane['p90_error_mm'],'rgb_refined':refined,'_plane':plane})
    return max(candidates,key=lambda item:item['support_fraction']) if candidates else None


def _capture_arrays(captured,sequence):
    if not isinstance(captured,dict) or captured.get('source_status')!='OK':return None,None
    packet=captured.get('packet')
    if not isinstance(packet,bytes) or len(packet)<614444:return None,None
    magic,version,seq,stamp,w,h,scale,flags=struct.unpack('<4sIQQIIfI',packet[:40])
    if (magic,version,seq,w,h)!=(b'NDP1',2,sequence,640,480) or flags&3!=3:return None,None
    length=struct.unpack('<I',packet[614440:614444])[0]
    if len(packet)!=614444+length:return None,None
    depth=np.frombuffer(packet[40:614440],'<u2').reshape(h,w).astype(np.float32)*scale
    rgb=cv2.imdecode(np.frombuffer(packet[614444:],np.uint8),cv2.IMREAD_COLOR) if length else None
    return rgb,depth


def surface_from_capture(captured,raw_table,objects,sequence):
    rgb,depth=_capture_arrays(captured,sequence)
    return surface_from_arrays(rgb,depth,raw_table,objects)


class SquareTableGeometry:
    """Bridge a noisy rim using fresh depth and current YOLO motion, never a lock."""
    def __init__(self):
        self.surface=None;self.raw=None;self.verified_at=0.;self.sequence=None;self.result=None
        self.pending=None

    def locate(self,rgb,depth,table,objects,sequence):
        if table is None or depth is None:return None
        if sequence==self.sequence:return self.result
        self.sequence=sequence
        result=surface_from_arrays(rgb,depth,table,objects,
            plane_hint=self.surface['_plane'] if self.surface is not None else None)
        if result is not None and self.surface is not None and time.monotonic()-self.verified_at<=2.:
            a,b,c,d=self.raw;x1,y1,x2,y2=table.xyxy
            sx,sy=(x2-x1)/(c-a),(y2-y1)/(d-b)
            expected=(np.float32(self.surface['polygon'])-[a,b])*[sx,sy]+[x1,y1]
            measured=np.float32(result['polygon'])
            if np.max(np.linalg.norm(measured-expected,axis=1))>max(10.,.08*(c-a)):
                if self.pending is None or np.max(np.linalg.norm(measured-self.pending,axis=1))>5.:
                    self.pending=measured;result=None
                else:self.pending=None
            else:
                self.pending=None
                measured=.65*measured+.35*expected
                result=dict(result,polygon=measured.astype(float).tolist(),
                    xyxy=[float(measured[:,0].min()),float(measured[:,1].min()),float(measured[:,0].max()),float(measured[:,1].max())],
                    rim_contacts=_rim_contacts(depth,result['_plane'],measured.astype(np.float32),objects))
        if result is not None:
            self.surface=result;self.raw=tuple(table.xyxy);self.verified_at=time.monotonic()
        elif self.surface is not None and time.monotonic()-self.verified_at<=2.:
            a,b,c,d=self.raw;x1,y1,x2,y2=table.xyxy
            sx,sy=(x2-x1)/(c-a),(y2-y1)/(d-b)
            dx,dy=(x1+x2-a-c)/2,(y1+y2-b-d)/2
            if .85<=sx<=1.15 and .85<=sy<=1.15 and abs(dx)<.15*(c-a) and abs(dy)<.15*(d-b):
                quad=np.float32(self.surface['polygon'])
                quad=(quad-np.float32([a,b]))*np.float32([sx,sy])+np.float32([x1,y1])
                yy,xx=np.mgrid[:depth.shape[0],:depth.shape[1]]
                plane=self.surface['_plane'];inside=_mask(depth.shape,quad)
                support=inside&(depth>250)&(depth<5000)&(np.abs(gaps(depth,plane,xx,yy))<=18.)
                coverage=float(support.sum()/max(1,inside.sum()))
                ys,xs=np.nonzero(support);cx,cy=quad.mean(axis=0)
                quadrants=set(zip(xs>=cx,ys>=cy))
                if coverage>=.28 and len(quadrants)>=3:
                    refined_quad,rear_verified=_rear_edge(rgb,depth,plane,table.xyxy,objects,quad)
                    if np.max(np.linalg.norm(refined_quad[:2]-quad[:2],axis=1))<=max(4.,.05*(y2-y1)):
                        quad=refined_quad
                    else:rear_verified=False
                    result=dict(self.surface,polygon=quad.astype(float).tolist(),
                        xyxy=[float(quad[:,0].min()),float(quad[:,1].min()),float(quad[:,0].max()),float(quad[:,1].max())],
                        source='current_depth_verified_tracking',support_fraction=coverage,
                        rear_edge_verified=rear_verified or self.surface.get('rear_edge_verified',False),
                        rim_contacts=_rim_contacts(depth,plane,quad,objects))
                    self.verified_at=time.monotonic()
        self.result=result
        return result

    def capture(self,captured,table,objects,sequence):
        rgb,depth=_capture_arrays(captured,sequence)
        return self.locate(rgb,depth,table,objects,sequence)
