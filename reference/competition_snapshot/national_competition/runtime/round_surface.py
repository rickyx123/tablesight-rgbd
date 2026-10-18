"""Estimate a circular tabletop from a validated, same-sequence depth packet."""
import math
import struct

import cv2
import numpy as np

from national_competition.runtime.depth_validation import fit_plane,gaps


def surface_from_capture(captured,raw_table,objects,sequence,reference_box=None):
    if not isinstance(captured,dict) or captured.get('source_status')!='OK' or raw_table is None:
        return None
    packet=captured.get('packet')
    if not isinstance(packet,bytes) or len(packet)<614440:return None
    magic,version,seq,stamp,w,h,scale,flags=struct.unpack('<4sIQQIIfI',packet[:40])
    if (magic,version,seq,w,h)!=(b'NDP1',2,sequence,640,480) or flags&3!=3:return None
    depth=np.frombuffer(packet[40:614440],dtype='<u2').reshape(h,w).astype(np.float32)*scale
    seed=raw_table.xyxy if reference_box is None else reference_box
    surface=estimate_surface(depth,seed,[d.xyxy for d in objects if d.class_id!=16])
    if surface is None:return None
    if len(packet)>614444:
        rgb=cv2.imdecode(np.frombuffer(packet[614444:],np.uint8),cv2.IMREAD_COLOR)
        if rgb is not None and rgb.shape[:2]==depth.shape:
            surface=refine_rgb_surface(rgb,depth,surface)
    return dict(surface,reference_seed_used=reference_box is not None)


def surface_from_arrays(rgb,depth,raw_table,objects,reference_box=None):
    if raw_table is None or depth is None:return None
    seed=raw_table.xyxy if reference_box is None else reference_box
    surface=estimate_surface(depth,seed,[d.xyxy for d in objects if d.class_id!=16])
    if surface is not None and rgb is not None and rgb.shape[:2]==depth.shape:
        surface=refine_rgb_surface(rgb,depth,surface)
    return dict(surface,reference_seed_used=reference_box is not None) if surface is not None else None


def refine_rgb_surface(rgb,depth,surface):
    """Use the same image's rim only when it agrees with measured depth support."""
    x1,y1,x2,y2=surface['xyxy'];cx,cy=(x1+x2)/2,(y1+y2)/2
    rx,ry=(x2-x1)/2,(y2-y1)/2
    h,w=depth.shape;yy,xx=np.mgrid[0:h,0:w]
    gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    inside=((xx-cx)/rx)**2+((yy-cy)/ry)**2<.8**2
    plane=surface['_plane']
    flat=inside&(depth>250)&(depth<5000)&(np.abs(gaps(depth,plane,xx,yy))<=12.)
    if flat.sum()<200:return surface
    level=float(np.percentile(gray[flat],75))-30.
    if level<60:return surface
    a,b=max(0,int(cx-1.35*rx)),max(0,int(cy-1.35*ry))
    c,d=min(w,int(cx+1.35*rx)+1),min(h,int(cy+1.35*ry)+1)
    contours=[]
    # Printed patterns and glare can split/join the rim at one luminance level.
    # Every candidate still needs the same depth agreement and ellipse checks.
    for threshold in (level,level+5.,level+10.,level+15.):
        mask=(gray[b:d,a:c]>=threshold).astype(np.uint8)*255
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        found,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    candidates=[]
    for contour in contours:
        hull=cv2.convexHull(contour)
        if len(hull)<8 or cv2.contourArea(hull)<.7*math.pi*rx*ry:continue
        hx,hy,hw,hh=cv2.boundingRect(hull)
        if hx<=1 or hy<=1 or hx+hw>=c-a-1 or hy+hh>=d-b-1:continue
        (mx,my),(ew,eh),angle=cv2.fitEllipse(hull);mx+=a;my+=b
        theta=math.radians(angle)
        ax=math.sqrt((ew/2*math.cos(theta))**2+(eh/2*math.sin(theta))**2)
        ay=math.sqrt((ew/2*math.sin(theta))**2+(eh/2*math.cos(theta))**2)
        if not (.92<=ax/rx<=1.12 and .92<=ay/ry<=1.12):continue
        if abs(mx-cx)>.10*rx or abs(my-cy)>.15*ry:continue
        points=hull[:,0,:].astype(float)
        radii=np.sqrt(((points[:,0]+a-mx)/ax)**2+((points[:,1]+b-my)/ay)**2)
        error=float(np.percentile(np.abs(radii-1.),90))
        fill=cv2.contourArea(hull)/(math.pi*ax*ay)
        if error>.08 or not .88<=fill<=1.10:continue
        candidates.append((ax*ay,error,(mx-ax,my-ay,mx+ax,my+ay)))
    if candidates:
        _,error,box=max(candidates)
        return dict(surface,xyxy=box,depth_xyxy=surface['xyxy'],rgb_refined=True,rgb_boundary_p90_error=error)
    return dict(surface,rgb_refined=False)


def estimate_surface(depth,raw_box,object_boxes=()):
    """Depth determines table support/boundary, never bottle capacity."""
    plane=fit_plane(depth,raw_box,object_boxes,'round')
    if plane is None:return None
    x1,y1,x2,y2=raw_box;bw,bh=x2-x1,y2-y1
    height,width=depth.shape
    a,b=max(0,int(x1-.35*bw)),max(0,int(y1-.35*bh))
    c,d=min(width,int(x2+.35*bw)+1),min(height,int(y2+.35*bh)+1)
    yy,xx=np.mgrid[b:d,a:c];z=depth[b:d,a:c]
    support=((z>250)&(z<5000)&(np.abs(gaps(z,plane,xx,yy))<=12.)).astype(np.uint8)*255
    closed=cv2.morphologyEx(support,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(closed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for contour in contours:
        if cv2.contourArea(contour)<.25*bw*bh:continue
        hull=cv2.convexHull(contour)
        if len(hull)<8:continue
        (cx,cy),(ew,eh),angle=cv2.fitEllipse(hull)
        cx+=a;cy+=b
        theta=math.radians(angle)
        rx=math.sqrt((ew/2*math.cos(theta))**2+(eh/2*math.sin(theta))**2)
        ry=math.sqrt((ew/2*math.sin(theta))**2+(eh/2*math.cos(theta))**2)
        if not (.65*bw<=2*rx<=1.6*bw and .65*bh<=2*ry<=1.6*bh):continue
        if abs(cx-(x1+x2)/2)>.25*bw or abs(cy-(y1+y2)/2)>.25*bh:continue
        if not .30<=ry/rx<=1.1:continue
        # Reject a truncated region at the search boundary or image edge.
        hx,hy,hw,hh=cv2.boundingRect(hull)
        if hx<=1 or hy<=1 or hx+hw>=c-a-1 or hy+hh>=d-b-1:continue
        points=hull[:,0,:].astype(float)
        radii=np.sqrt(((points[:,0]+a-cx)/rx)**2+((points[:,1]+b-cy)/ry)**2)
        error=float(np.percentile(np.abs(radii-1.),90))
        quadrants={(int(p[0]+a>=cx),int(p[1]+b>=cy)) for p in points}
        if error>.10 or len(quadrants)<4:continue
        ellipse=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1.
        coverage=float(((support>0)&ellipse).sum()/max(1,ellipse.sum()))
        hull_fill=float(cv2.contourArea(hull)/(math.pi*rx*ry))
        if coverage<.45 or not .78<=hull_fill<=1.10:continue
        candidates.append({'xyxy':(cx-rx,cy-ry,cx+rx,cy+ry),
            'support_fraction':coverage,'boundary_p90_error':error,'hull_fill':hull_fill,
            'plane_inliers':plane['inliers'],'plane_p90_error_mm':plane['p90_error_mm'],'_plane':plane})
    return max(candidates,key=lambda c:c['support_fraction']) if candidates else None
