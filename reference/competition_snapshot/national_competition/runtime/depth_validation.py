"""Sequence-matched depth evidence. Only stable YOLO quantities may be vetoed.

Distances are camera-ray residuals from a fitted inverse-depth support plane,
not bottle capacity, object dimensions, or assumed vertical object heights.
"""
import json
import struct
import time
import os
from pathlib import Path
from collections import Counter

import numpy as np

from national_competition.config import NAMES
from national_competition.fusion import iou
from national_competition.runtime.depth_ownership import foreign_water_voids
from national_competition.runtime.depth_overlap import foreign_edge_mask,has_interior_structure

HARD_DEPTH_CLASSES = frozenset((2, 3, 5, 8, 12, 13, 15))
WATER_DEPTH_CLASSES = frozenset((10, 11))
PRINT_DEPTH_CLASSES = frozenset(range(1,16))


def quadrant_count(xx,yy,mask,cx,cy):
    codes=(xx[mask]>=cx).astype(np.uint8)+2*(yy[mask]>=cy).astype(np.uint8)
    return int(np.count_nonzero(np.bincount(codes,minlength=4)))


def transparent_bottle_support(depth,plane,row,other_boxes,profile):
    import cv2
    a,b,c,d=row['xyxy'];h,w=depth.shape
    x1,x2=max(0,int(a)+1),min(w,int(c)-1);y1,y2=max(0,int(b)+1),min(h,int(d)-1)
    if x2-x1<12 or y2-y1<12:return None
    yy,xx=np.mgrid[y1:y2:2,x1:x2:2];z=depth[y1:y2:2,x1:x2:2]
    allowed=np.ones(z.shape,dtype=bool)
    for oa,ob,oc,od in row.get('positive_exclusion_boxes',other_boxes):
        allowed &= ~((xx>=oa)&(xx<=oc)&(yy>=ob)&(yy<=od))
    if allowed.mean()<.20 or allowed.sum()<80:return None
    missing=allowed&~((z>250)&(z<5000)&np.isfinite(z))
    hole_fraction=float(missing.sum()/allowed.sum())
    count,_,stats,_=cv2.connectedComponentsWithStats(missing.astype(np.uint8),8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0
    if hole_fraction<.25 or largest<max(30,int(allowed.sum()*.20)):return None
    pad=max(12,int(.15*max(c-a,d-b)))
    rx1,rx2=max(0,int(a)-pad),min(w,int(c)+pad);ry1,ry2=max(0,int(b)-pad),min(h,int(d)+pad)
    ry,rx=np.mgrid[ry1:ry2:2,rx1:rx2:2];rz=depth[ry1:ry2:2,rx1:rx2:2]
    ring=(rx<a)|(rx>=c)|(ry<b)|(ry>=d)
    ring &= ((rx-plane['cx'])/plane['rx'])**2+((ry-plane['cy'])/plane['ry'])**2<.95**2
    for oa,ob,oc,od in other_boxes:ring &=~((rx>=oa)&(rx<=oc)&(ry>=ob)&(ry<=od))
    valid=ring&(rz>250)&(rz<5000)&np.isfinite(rz)
    ring_valid=float(valid.sum()/max(1,ring.sum()))
    good=valid&(np.abs(gaps(rz,plane,rx,ry))<=max(10.,2*plane['p90_error_mm']))
    ring_agreement=float(good.sum()/max(1,valid.sum()))
    quadrants=quadrant_count(rx,ry,good,(a+c)/2,(b+d)/2)
    if good.sum()<40 or ring_valid<.75 or ring_agreement<.65 or quadrants<2:return None
    if hole_fraction-(1-ring_valid)<.20:return None
    return {'state':'UNKNOWN','reason':'TRANSPARENT_BOTTLE_PATTERN','reject':False,
            'water_transparent_support':True,'hole_fraction':hole_fraction,
            'connected_hole_points':largest,'visible_fraction':float(allowed.mean()),
            'ring_valid_fraction':ring_valid,'ring_plane_agreement':ring_agreement}


def print_local_reference(surround,rx,ry,good,plane,box):
    """Fit only clean support pixels outside the target; never fit its centre."""
    a,b,c,d=box;cx,cy=(a+c)/2,(b+d)/2;srx,sry=(c-a)/2,(d-b)/2
    x,y,z=rx[good],ry[good],surround[good]
    if len(z)<60:return None
    if len(z)>600:
        selected=np.linspace(0,len(z)-1,600).astype(int);x,y,z=x[selected],y[selected],z[selected]
    design=np.column_stack(((x-cx)/srx,(y-cy)/sry,np.ones(len(z))))
    if np.linalg.matrix_rank(design)<3 or np.linalg.cond(design)>30:return None
    keep=np.ones(len(z),bool)
    for _ in range(3):
        if keep.sum()<60:return None
        coeff=np.linalg.lstsq(design[keep],1/z[keep],rcond=None)[0]
        predicted=design@coeff
        if np.any(predicted<=0):return None
        error=1/predicted-z
        centre=float(np.median(error[keep]));mad=float(np.median(np.abs(error[keep]-centre)))
        keep=np.abs(error-centre)<=max(3.,3.*1.4826*mad)
    if keep.sum()<60:return None
    coeff=np.linalg.lstsq(design[keep],1/z[keep],rcond=None)[0]
    error=1/(design[keep]@coeff)-z[keep]
    noise=float(np.percentile(np.abs(error),90))
    if noise>7.:return None
    local={'coeff':coeff.tolist(),'cx':cx,'cy':cy,'rx':srx,'ry':sry,
           'p90_error_mm':noise,'inliers':int(keep.sum())}
    xx,yy=np.meshgrid(np.linspace(a,c,3),np.linspace(b,d,3))
    spread=np.abs(gaps(np.zeros_like(xx),local,xx,yy)-gaps(np.zeros_like(xx),plane,xx,yy))
    if float(np.max(spread))>max(6.,1.5*plane['p90_error_mm']):return None
    return local


def visible_print_layout(xx,yy,valid,core,box):
    """Partial visibility needs broad interior evidence, not a exposed border."""
    a,b,c,d=box;bw,bh=c-a,d-b
    central=valid&core
    if central.sum()<max(30,.15*core.sum()):return False
    if np.ptp(xx[central])<.20*bw or np.ptp(yy[central])<.20*bh:return False
    gx=np.clip(((xx[valid]-a)/bw*3).astype(int),0,2)
    gy=np.clip(((yy[valid]-b)/bh*3).astype(int),0,2)
    cells=np.bincount(gy*3+gx,minlength=9)
    occupied=cells>=max(12,int(valid.sum()*.05))
    return occupied.sum()>=3 and occupied[4] and len(set(gx))>=2 and len(set(gy))>=2


def bottle_print_evidence(depth,plane,row,other_boxes,profile):
    """Strict dense-plane proof; sparse transparent/background returns cannot veto."""
    import cv2
    water_candidate=row.get('class_id') in WATER_DEPTH_CLASSES or profile.get('bottle_positive_only',False)
    bx1,by1,bx2,by2=row['xyxy']
    h,w=depth.shape
    x1,x2=max(0,int(np.floor(bx1))),min(w,int(np.ceil(bx2)))
    y1,y2=max(0,int(np.floor(by1))),min(h,int(np.ceil(by2)))
    if x2-x1<12 or y2-y1<12:return None
    stride=1 if (x2-x1)*(y2-y1)*.72*.80<1600 else 2
    area_scale=(2/stride)**2
    yy,xx=np.mgrid[y1:y2:stride,x1:x2:stride]
    z=depth[y1:y2:stride,x1:x2:stride]
    weights=np.maximum(0,np.minimum(xx+stride,bx2)-np.maximum(xx,bx1))*np.maximum(0,np.minimum(yy+stride,by2)-np.maximum(yy,by1))
    # Keep the complete RGB box; erode only negative depth measurements.
    inset_x=max(2.,.14*(bx2-bx1));inset_y=max(2.,.10*(by2-by1))
    inner=(xx>=bx1+inset_x)&(xx<bx2-inset_x)&(yy>=by1+inset_y)&(yy<by2-inset_y)
    allowed=inner.copy()
    delta=gaps(z,plane,xx,yy)
    foreign=row.get('foreign_depth_void_mask')
    edge_foreign=row.get('_foreign_edge_mask')
    if edge_foreign is None:edge_foreign=foreign_edge_mask(depth,plane,row['xyxy'],other_boxes,gaps)
    for a,b,c,d in other_boxes:
        covered=(xx>=a-3)&(xx<=c+3)&(yy>=b-3)&(yy<=d+3)
        # Two planar printed targets may have overlapping loose RGB boxes.
        # Opaque negative evidence excludes actual foreground, not flat pixels
        # belonging to the same table. Transparent targets retain box masks.
        foreground=(delta>max(10.,2*plane['p90_error_mm']))|~((z>250)&(z<5000)&np.isfinite(z))
        if water_candidate and foreign is None:allowed &= ~covered
        else:allowed &= ~(covered&(z>250)&(z<5000)&np.isfinite(z)&(delta>max(10.,2*plane['p90_error_mm'])))
        if not water_candidate:allowed &=~(covered&~((z>250)&(z<5000)&np.isfinite(z)))
    if foreign is not None:allowed &=~foreign[yy,xx]
    if edge_foreign is not None:allowed &=~edge_foreign[yy,xx]
    own_core=(xx>=bx1+.25*(bx2-bx1))&(xx<bx2-.25*(bx2-bx1))&(yy>=by1+.25*(by2-by1))&(yy<by2-.25*(by2-by1))
    raw_front=(z>250)&(z<5000)&np.isfinite(z)&(delta>max(8.,2*plane['p90_error_mm']))
    heavy_overlap=bool(edge_foreign is not None and
                   (raw_front&edge_foreign[yy,xx]).sum()*stride**2>=.40*(bx2-bx1)*(by2-by1))
    visible=float(allowed.sum()/max(1,inner.sum()))
    valid=allowed&(z>250)&(z<5000)&np.isfinite(z)
    density=float(valid.sum()/max(1,allowed.sum()))
    # Water caps may be the only trustworthy raised pixels. Keep that
    # protection across the whole bottle, but never borrow another object's
    # depth merely because its YOLO confidence is lower.
    positive_allowed=np.ones(z.shape,dtype=bool)
    for a,b,c,d in other_boxes:
        positive_allowed &= ~((xx>=a-3)&(xx<=c+3)&(yy>=b-3)&(yy<=d+3))
    if edge_foreign is not None:positive_allowed &=~edge_foreign[yy,xx]
    positive_valid=positive_allowed&(z>250)&(z<5000)&np.isfinite(z)
    raised=(positive_valid&(delta>profile['pass_mm'])).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(raised,connectivity=8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if count>1 else 0
    # The denominator never shrinks with occlusion or missing depth.
    bbox_area=max(1.,(bx2-bx1)*(by2-by1))
    positive_fraction=float((raised*weights).sum()/bbox_area)
    required_area=profile.get('pass_area_fraction',.12)
    component_areas=np.bincount(labels.reshape(-1),weights=(raised*weights).reshape(-1))
    connected_fraction=float(component_areas[1:].max()/bbox_area) if len(component_areas)>1 else 0.
    area_stats={'pass_height_mm':profile['pass_mm'],'pass_required_box_fraction':required_area,
                'raised_box_fraction':positive_fraction,'connected_box_fraction':connected_fraction,
                'valid_box_fraction':float((positive_valid*weights).sum()/bbox_area),
                'raw_valid_box_fraction':float((((z>250)&(z<5000)&np.isfinite(z))*weights).sum()/bbox_area),
                'raw_missing_box_fraction':float(((~((z>250)&(z<5000)&np.isfinite(z)))*weights).sum()/bbox_area),
                'pass_area_ready':positive_fraction>=required_area and connected_fraction>=.60*required_area}
    if '_height_area_gate' in row:row['_height_area_gate'].update(area_stats)
    if positive_fraction>=required_area and connected_fraction>=.60*required_area:
        return {'state':'PASS','reason':'BOTTLE_VISIBLE_STRUCTURE' if water_candidate else 'VISIBLE_3D_STRUCTURE','reject':False,
                'water_structure':water_candidate,
                'dense_valid_fraction':float(positive_valid.sum()/max(1,positive_allowed.sum())),
                'visible_fraction':float(positive_allowed.mean()),
                'positive_point_ownership':'all_other_boxes_excluded',
                'negative_depth_roi_inset_xy':(.14,.10),
                'raised_fraction':positive_fraction,
                'pass_height_mm':profile['pass_mm'],'pass_required_box_fraction':required_area,
                'raised_box_fraction':positive_fraction,'connected_box_fraction':connected_fraction,
                'pass_basis':'CLASS_HEIGHT_AND_WHOLE_YOLO_BOX_AREA',
                'raised_connected_points':largest}
    if connected_fraction>=.50*required_area:
        return {'state':'UNKNOWN','reason':'HEIGHT_POINTS_TOO_SMALL_FOR_OBJECT','reject':False,
                'pass_height_mm':profile['pass_mm'],'raised_box_fraction':positive_fraction,
                'pass_required_box_fraction':required_area,'raised_connected_points':largest}
    # A bottle's own cap or an interior fragment may be too small for PASS,
    # but still prevents a table-returning centre from proving a flat print.
    weak_height=max(profile.get('shallow_protect_mm',max(3.,profile['flat_mm'])),
                    2.*plane['p90_error_mm'],.5*profile['pass_mm'])
    weak_raised=positive_valid&(delta>weak_height)
    _,weak_labels,_,_=cv2.connectedComponentsWithStats(weak_raised.astype(np.uint8),8)
    weak_areas=np.bincount(weak_labels.reshape(-1),weights=(weak_raised*weights).reshape(-1))
    weak_connected=float(weak_areas[1:].max()/bbox_area) if len(weak_areas)>1 else 0.
    if weak_connected>=max(.015,.15*required_area) and has_interior_structure(
            weak_raised,xx,yy,row['xyxy'],1/(stride*stride),water_candidate):
        return {'state':'UNKNOWN','reason':'SMALL_OWN_3D_STRUCTURE','reject':False,
                'raised_box_fraction':positive_fraction,'connected_box_fraction':connected_fraction,
                'pass_height_mm':profile['pass_mm'],'pass_required_box_fraction':required_area,
                'weak_own_height_mm':weak_height,'weak_own_connected_box_fraction':weak_connected}
    # Opaque solid targets may be judged from a broad exposed half of the
    # eroded ROI. Transparent bottles and thin/hollow targets retain 60%.
    opaque_solid=row.get('class_id') in (2,3,4,5,6,7,8,12,13,15)
    min_visible=profile.get('print_visible_fraction',.50 if opaque_solid else .60)
    partial=(visible<min_visible or heavy_overlap)
    if partial:
        minimum=profile.get('print_partial_visible_fraction',min_visible)
        # Missing depth cannot masquerade as known foreground occlusion.
        known_foreground=(delta>max(8.,2*plane['p90_error_mm']))&(z>250)&(z<5000)
        hidden_known=(inner&~allowed&known_foreground).sum()/max(1,inner.sum())
        if visible<minimum or hidden_known<.15 or not visible_print_layout(xx,yy,valid,own_core,row['xyxy']):
            return {'state':'UNKNOWN','reason':'INSUFFICIENT_VISIBLE_PRINT_LAYOUT','reject':False,
                    'visible_fraction':visible,'known_foreground_fraction':float(hidden_known)} if heavy_overlap else None
    if density<profile.get('print_valid_fraction',.95) or valid.sum()<200:return None
    required_span=.30 if partial else .50
    if np.ptp(xx[valid])<required_span*(bx2-bx1-2*inset_x) or np.ptp(yy[valid])<required_span*(by2-by1-2*inset_y):return None
    noise=float(plane['p90_error_mm'])
    tolerance=min(profile.get('print_max_flat_mm',12.),max(6.,profile['flat_mm']+2*noise))
    residual=np.abs(delta[valid])
    if np.median(residual)>profile.get('print_median_mm',6.) or np.percentile(residual,90)>tolerance:return None
    # A partially obscured ring must still agree broadly with the SAME plane.
    pad=max(12,int(.15*max(bx2-bx1,by2-by1)))
    a,b=max(0,int(bx1)-pad),max(0,int(by1)-pad)
    c,d=min(w,int(bx2)+pad),min(h,int(by2)+pad)
    ry,rx=np.mgrid[b:d:2,a:c:2];surround=depth[b:d:2,a:c:2]
    ring=(rx<bx1)|(rx>=bx2)|(ry<by1)|(ry>=by2)
    ring &= ((rx-plane['cx'])/plane['rx'])**2+((ry-plane['cy'])/plane['ry'])**2<.95**2
    ring_gaps=gaps(surround,plane,rx,ry)
    foreground=ring_gaps>max(10.,2*noise)
    for ox1,oy1,ox2,oy2 in other_boxes:
        covered=(rx>=ox1-3)&(rx<=ox2+3)&(ry>=oy1-3)&(ry<=oy2+3)
        ring &= ~(covered if water_candidate and foreign is None else covered&foreground)
    if foreign is not None:ring &=~foreign[ry,rx]
    if edge_foreign is not None:ring &=~edge_foreign[ry,rx]
    ring &= (surround>250)&(surround<5000)&np.isfinite(surround)
    good=ring&(np.abs(ring_gaps)<=tolerance)
    quadrants=quadrant_count(rx,ry,good,(bx1+bx2)/2,(by1+by2)/2)
    ring_fraction=float(good.sum()/max(1,ring.sum()))
    if good.sum()<60 or ring_fraction<profile.get('print_ring_fraction',.65) or quadrants<3:return None
    local=print_local_reference(surround,rx,ry,good,plane,row['xyxy'])
    if partial and local is None:return None
    flat_delta=gaps(z,local,xx,yy) if local is not None else delta
    flat_residual=np.abs(flat_delta[valid])
    local_noise=local['p90_error_mm'] if local is not None else noise
    # Local refinement is an extra consistency check, never permission to
    # enlarge the original class-specific flat residual limits.
    if np.median(flat_residual)>profile.get('print_median_mm',6.) or np.percentile(flat_residual,90)>tolerance:return None
    core_valid=valid&own_core
    local_bias=float(np.median(ring_gaps[good]))
    protect=max(profile.get('shallow_protect_mm',max(3.,profile['flat_mm'])),
                profile.get('shallow_protect_noise_scale',0.)*max(noise,local_noise))
    reference_info={'plane_noise_mm':noise,'local_reference_p90_error_mm':local_noise,
                    'local_reference_refined':local is not None,'shallow_protect_mm':protect,
                    'visible_fraction':visible,'dense_valid_fraction':density,
                    'partial_visible_proof':bool(partial)}
    if core_valid.sum()>=30:
        core_delta=flat_delta[core_valid] if local is not None else delta[core_valid]-local_bias
        core_median=float(np.median(core_delta));core_lower=float(np.percentile(core_delta,25))
        reference_noise=max(noise,local_noise)
        # A well-measured shallow real surface remains protected even for a
        # large class. The class minimum is not permission to delete it.
        significant_shallow=(reference_noise<=2.5 and
            core_median>max(3.,2.5*reference_noise) and core_lower>max(1.,2.*reference_noise))
        if ((core_median>protect and core_lower>.5*profile['flat_mm']) or significant_shallow):
            return {'state':'UNKNOWN','reason':'SHALLOW_COHERENT_INTERIOR','reject':False,
                    'local_core_gap_median_mm':core_median,
                    'local_table_bias_mm':local_bias,'significant_shallow_structure':bool(significant_shallow),
                    **reference_info}
    return {'state':'FLAT','reason':'DENSE_PLANAR_BOTTLE_PRINT' if water_candidate else 'DENSE_PLANAR_PRINT','reject':True,
            'reliable_flat':True,'reliable_bottle_print':water_candidate,
            'reliable_planar_print':True,
            'dense_valid_fraction':density,'visible_fraction':visible,
            'whole_box_abs_gap_p90_mm':float(np.percentile(residual,90)),
            'whole_box_abs_gap_median_mm':float(np.median(residual)),
            'local_abs_gap_median_mm':float(np.median(flat_residual)),
            'local_abs_gap_p90_mm':float(np.percentile(flat_residual,90)),
            'effective_flat_mm':tolerance,'plane_noise_mm':noise,
            'ring_support_fraction':ring_fraction,'raised_connected_points':largest,
            'sample_stride_px':stride,'valid_sample_points':int(valid.sum()),
            'foreign_void_points_excluded':int((foreign[yy,xx]&inner).sum()) if foreign is not None else 0,
            'foreign_edge_points_excluded':int((edge_foreign[yy,xx]&inner).sum()) if edge_foreign is not None else 0,
            'negative_depth_roi_inset_xy':(.14,.10),**reference_info}


def unknown(reason):
    return {'state': 'UNKNOWN', 'reason': reason, 'reject': False}


def round_floor_evidence(depth,plane,row):
    """Strong table-relative proof that a round-table edge candidate is on the floor."""
    if row.get('radius',0.)<.75:return None
    h,w=depth.shape;a,b,c,d=row['xyxy']
    x1,x2=max(0,int(np.floor(a))),min(w,int(np.ceil(c)))
    y1,y2=max(0,int(np.floor(b))),min(h,int(np.ceil(d)))
    if x2-x1<5 or y2-y1<5:return None
    z=depth[y1:y2,x1:x2];yy,xx=np.mgrid[y1:y2,x1:x2]
    valid=(z>250)&(z<5000)&np.isfinite(z)
    valid_fraction=float(valid.mean())
    if valid_fraction<.45 or valid.sum()<80:return None
    delta=gaps(z,plane,xx,yy)[valid]
    median=float(np.median(delta));below_fraction=float((delta<-60).mean())
    if median>=-100 or below_fraction<.60:return None
    return {'state':'FLAT','reason':'RELIABLE_BELOW_TABLE','reject':True,
            'reliable_flat':True,'reliable_outside_table':True,
            'table_gap_median_mm':median,'below_table_fraction':below_fraction,
            'valid_fraction':valid_fraction,'round_radius':row['radius']}


def opaque_core_print_evidence(depth, plane, row, other_boxes, profile):
    """Opaque objects: dense flat centre plus a clean adjacent support segment.

    A neighbouring raised object or floor beyond the table edge cannot poison
    every support side. Sparse depth or a raised centre never proves a print.
    Fixed millimetre limits do not grow with noisy plane estimates.
    """
    if not profile.get('opaque_core_print_check') or plane['p90_error_mm'] > 7:
        return None
    a,b,c,d=row['xyxy'];h,w=depth.shape;bw,bh=c-a,d-b
    x1,x2=max(0,int(a+.30*bw)),min(w,int(c-.30*bw))
    y1,y2=max(0,int(b+.30*bh)),min(h,int(d-.30*bh))
    if min(x2-x1,y2-y1)<8:return None
    yy,xx=np.mgrid[y1:y2:2,x1:x2:2];z=depth[y1:y2:2,x1:x2:2]
    visible=np.ones(z.shape,dtype=bool)
    raw_gap=gaps(z,plane,xx,yy)
    for oa,ob,oc,od in other_boxes:
        covered=(xx>=oa)&(xx<=oc)&(yy>=ob)&(yy<=od)
        visible &= ~(covered&((raw_gap>max(10.,2*plane['p90_error_mm']))|~((z>250)&(z<5000))))
    valid=visible&(z>250)&(z<5000)&np.isfinite(z)
    if visible.mean()<.90 or valid.sum()<40 or valid.sum()/max(1,visible.sum())<.95:return None
    delta=raw_gap[valid];residual=np.abs(delta)
    if np.median(residual)>3 or np.percentile(residual,90)>6 or np.percentile(residual,95)>10:return None
    # A genuine opaque condiment has a raised centre even when lying down.
    if (delta>profile.get('opaque_protect_mm',12)).mean()>.03:return None
    pad=max(12,int(.20*max(bw,bh)))
    ry,rx=np.mgrid[max(0,int(b)-pad):min(h,int(d)+pad):2,max(0,int(a)-pad):min(w,int(c)+pad):2]
    rz=depth[ry,rx];rg=gaps(rz,plane,rx,ry)
    available=(rz>250)&(rz<5000)&np.isfinite(rz)
    for oa,ob,oc,od in other_boxes:
        available &= ~((rx>=oa-3)&(rx<=oc+3)&(ry>=ob-3)&(ry<=od+3)&(rg>max(10.,2*plane['p90_error_mm'])))
    sides=(('left',(rx<a)&(ry>=b+.30*bh)&(ry<d-.30*bh),ry,bh),
           ('right',(rx>=c)&(ry>=b+.30*bh)&(ry<d-.30*bh),ry,bh),
           ('top',(ry<b)&(rx>=a+.30*bw)&(rx<c-.30*bw),rx,bw),
           ('bottom',(ry>=d)&(rx>=a+.30*bw)&(rx<c-.30*bw),rx,bw))
    support=[]
    for name,mask,along,length in sides:
        area=int(mask.sum());mask &= available
        good=mask&(np.abs(rg)<=7)
        if mask.sum()<40 or mask.sum()/max(1,area)<.75 or good.sum()/max(1,mask.sum())<.70:continue
        if good.sum()<40 or np.ptp(along[good])<.30*length:continue
        support.append(name)
    if not support:return None
    return {'state':'FLAT','reason':'OPAQUE_CORE_PLANAR_PRINT','reject':True,
            'reliable_flat':True,'reliable_planar_print':True,
            'core_abs_gap_median_mm':float(np.median(residual)),
            'core_abs_gap_p90_mm':float(np.percentile(residual,90)),
            'core_abs_gap_p95_mm':float(np.percentile(residual,95)),
            'core_valid_fraction':float(valid.sum()/visible.sum()),
            'clean_support_sides':support,'plane_noise_mm':plane['p90_error_mm']}


def capture_active():
    try:
        pid=int(Path('/dev/shm/national_depth/active.pid').read_text().strip())
        os.kill(pid,0)
        return b'rgbd_snapshot_capture' in Path(f'/proc/{pid}/cmdline').read_bytes()
    except (OSError,ValueError):return False


def rgb_for_sequence(sequence):
    if not capture_active():return None
    try:
        data=(Path('/dev/shm/national_depth')/f'slot_{sequence%32}.bin').read_bytes()
        magic,version,seq,stamp,w,h,scale,flags=struct.unpack('<4sIQQIIfI',data[:40])
        if magic!=b'NDP1' or version!=2 or seq!=sequence or not 0<=time.monotonic()-stamp/1e6<=1.2:return None
        end=40+w*h*2
        length=struct.unpack('<I',data[end:end+4])[0]
        if len(data)!=end+4+length:return None
        import cv2
        return cv2.imdecode(np.frombuffer(data[end+4:],dtype=np.uint8),cv2.IMREAD_COLOR)
    except (OSError,ValueError,struct.error):return None


def decode_rgbd_packet(data,sequence,now):
    try:
        magic,version,seq,stamp,w,h,scale,flags=struct.unpack('<4sIQQIIfI',data[:40])
        if magic!=b'NDP1' or version!=2 or seq!=sequence:return None,None,'SEQUENCE_MISMATCH'
        if (w,h)!=(640,480):return None,None,'INVALID_DEPTH_PACKET'
        end=40+w*h*2
        length=struct.unpack('<I',data[end:end+4])[0]
        if len(data)!=end+4+length:return None,None,'INVALID_RGB_DEPTH_PACKET'
        if not 0<=now-stamp/1e6<=1.2:return None,None,'STALE_DEPTH'
        if flags&3!=3:return None,None,'REGISTRATION_OR_PAIR_TIMING_INVALID'
        import cv2
        rgb=cv2.imdecode(np.frombuffer(data[end+4:],dtype=np.uint8),cv2.IMREAD_COLOR)
        if rgb is None:return None,None,'INVALID_RGB_PACKET'
        if rgb.shape[:2]!=(h,w):return None,None,'RGB_DEPTH_SIZE_MISMATCH'
        depth=np.frombuffer(data[40:end],dtype='<u2').reshape(h,w).astype(np.float32)*scale
        return rgb,depth,'OK'
    except (ValueError,struct.error):return None,None,'INVALID_RGB_DEPTH_PACKET'


def read_rgbd(sequence):
    if not capture_active():return None,None,'DEPTH_CAPTURE_INACTIVE'
    try:
        data=(Path('/dev/shm/national_depth')/f'slot_{sequence%32}.bin').read_bytes()
        return decode_rgbd_packet(data,sequence,time.monotonic())
    except OSError:return None,None,'DEPTH_MISSING'


def read_depth(sequence):
    depth,reason,_,_=read_depth_packet(sequence)
    return depth,reason


def read_depth_packet(sequence):
    if not capture_active():return None,'DEPTH_CAPTURE_INACTIVE',None,{}
    try:
        data = (Path('/dev/shm/national_depth')/f'slot_{sequence%32}.bin').read_bytes()
        magic, version, seq, stamp, w, h, scale, flags = struct.unpack('<4sIQQIIfI', data[:40])
        if magic != b'NDP1' or version != 2 or seq != sequence:
            return None, 'SEQUENCE_MISMATCH', None, {}
        end=40+w*h*2
        if w != 640 or h != 480 or len(data)<end+4:
            return None, 'INVALID_DEPTH_PACKET', None, {}
        length=struct.unpack('<I',data[end:end+4])[0]
        if len(data)!=end+4+length:return None,'INVALID_RGB_DEPTH_PACKET',None,{}
        if not 0 <= time.monotonic()-stamp/1e6 <= 1.2:
            return None, 'STALE_DEPTH', None, {}
        timing={'sequence':sequence,'rgb_host_us':stamp,'timing_status':'LEGACY_PAIR_FLAGS_ONLY'}
        try:
            meta=json.loads((Path('/dev/shm/national_depth')/f'slot_{sequence%32}.timing.json').read_text())
            timing=meta
            if meta['sequence']!=sequence or meta.get('rgb_host_us')!=stamp:
                return None,'PAIR_METADATA_MISMATCH',data,timing
            if meta['skew_us']>70000:return None,'PAIR_TIMING_TOO_LARGE',data,timing
        except (OSError,ValueError,KeyError):pass
        if flags & 3 != 3:
            return None, 'REGISTRATION_OR_PAIR_TIMING_INVALID', data, timing
        return np.frombuffer(data[40:end], dtype='<u2').reshape(h,w).astype(np.float32)*scale, 'OK',data,timing
    except (OSError, ValueError, struct.error):
        return None, 'DEPTH_MISSING', None, {}


def fit_plane(depth, table, boxes, mode, min_points=180,min_inliers=150,min_fraction=.60):
    h,w = depth.shape
    yy,xx=np.mgrid[0:h:2,0:w:2]
    x1,y1,x2,y2=table
    cx,cy=(x1+x2)/2,(y1+y2)/2
    rx,ry=(x2-x1)/2,(y2-y1)/2
    if min(rx,ry)<12:return None
    if mode in ('round', 'square_center'):
        mask=((xx-cx)/rx)**2+((yy-cy)/ry)**2 <= .8**2
    else:
        mask=(xx>x1+.12*(x2-x1))&(xx<x2-.12*(x2-x1))&(yy>y1+.12*(y2-y1))&(yy<y2-.12*(y2-y1))
    for bx1,by1,bx2,by2 in boxes:
        mask &= ~((xx>=bx1-3)&(xx<=bx2+3)&(yy>=by1-3)&(yy<=by2+3))
    zz=depth[::2,::2]
    mask &= (zz>250)&(zz<5000)&np.isfinite(zz)
    x,y,z=xx[mask],yy[mask],zz[mask]
    if len(z)<min_points:return None
    if len(z)>900:
        indices=np.linspace(0,len(z)-1,900).astype(int)
        x,y,z=x[indices],y[indices],z[indices]
    a=np.column_stack(((x-cx)/rx,(y-cy)/ry,np.ones(len(z))))
    rng=np.random.default_rng(17)
    chosen=np.array([rng.choice(len(z),3,replace=False) for _ in range(45)])
    try:coefficients=np.linalg.solve(a[chosen],(1/z)[chosen])
    except np.linalg.LinAlgError:
        coefficients=np.full((45,3),np.nan)
        for index,points in enumerate(chosen):
            try:coefficients[index]=np.linalg.solve(a[points],1/z[points])
            except np.linalg.LinAlgError:pass
    prediction=a@coefficients.T
    residual=np.abs(1/np.maximum(prediction,1e-6)-z[:,None])
    inliers=(prediction>0)&(residual<=7.)
    best=inliers[:,np.argmax(inliers.sum(axis=0))]
    if best.mean()<min_fraction or best.sum()<min_inliers:return None
    quadrants=quadrant_count(x,y,best,cx,cy)
    if quadrants<3 or np.ptp(x[best])<.4*(x2-x1) or np.ptp(y[best])<.3*(y2-y1):return None
    upper=y<cy
    if upper.sum()<30 or best[upper].mean()<.5:return None
    coeff=np.linalg.lstsq(a[best],1/z[best],rcond=None)[0]
    residual=np.abs(1/(a[best]@coeff)-z[best])
    if np.percentile(residual,90)>7:return None
    return {'coeff':coeff.tolist(),'cx':cx,'cy':cy,'rx':rx,'ry':ry,
            'inliers':int(best.sum()),'fraction':float(best.mean()),'p90_error_mm':float(np.percentile(residual,90))}


def gaps(depth,plane,xx,yy):
    coeff=plane['coeff']
    inv=coeff[0]*(xx-plane['cx'])/plane['rx']+coeff[1]*(yy-plane['cy'])/plane['ry']+coeff[2]
    return 1/np.maximum(inv,1e-6)-depth


def local_plane_agrees(local,global_plane,box):
    """A clean wall/background fit is not a table support surface."""
    a,b,c,d=box
    xx,yy=np.meshgrid(np.linspace(a,c,3),np.linspace(b,d,3))
    error=np.abs(gaps(np.zeros_like(xx),local,xx,yy)-gaps(np.zeros_like(xx),global_plane,xx,yy))
    return float(np.percentile(error,90))<=max(10.,2*global_plane['p90_error_mm'])


def rear_structure_evidence(depth,plane,row,other_boxes,profile):
    if not profile.get('rear_structure_check'):return None
    a,b,c,d=row['xyxy'];rear=plane['cy']-plane['ry'];table_h=2*plane['ry']
    if b>=rear or not rear-.10*table_h<=d<=rear+.45*table_h:return None
    h,w=depth.shape
    yy,xx=np.mgrid[max(0,int(b+.2*(d-b))):min(h,int(d-.2*(d-b))):2,max(0,int(a+.2*(c-a))):min(w,int(c-.2*(c-a))):2]
    z=depth[yy,xx];valid=(z>250)&(z<5000)&np.isfinite(z)
    for oa,ob,oc,od in other_boxes:valid &=~((xx>=oa)&(xx<=oc)&(yy>=ob)&(yy<=od))
    if valid.sum()<40 or valid.mean()<profile['valid_fraction']:return None
    delta=gaps(z,plane,xx,yy)[valid]
    if np.median(delta)<max(35.,profile['pass_mm']) or np.percentile(delta,90)<max(60.,2*profile['pass_mm']) or (delta>profile['pass_mm']).mean()<.60:return None
    import cv2
    raised=np.zeros(valid.shape,np.uint8);raised[valid]=(delta>profile['pass_mm']).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(raised,8)
    largest=int(stats[1:,cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
    if largest<max(20,int(valid.sum()*.30)):return None
    pad=max(12,int(.25*(d-b)))
    ry,rx=np.mgrid[max(0,int(b)-pad):max(0,int(b)):2,max(0,int(a)):min(w,int(c)):2]
    rz=depth[ry,rx];rv=(rz>250)&(rz<5000)&np.isfinite(rz)
    for oa,ob,oc,od in other_boxes:rv &=~((rx>=oa)&(rx<=oc)&(ry>=ob)&(ry<=od))
    if rv.sum()<30 or rv.mean()<.60:return None
    core=float(np.median(z[valid]));contrast=rz[rv]-core
    if np.median(contrast)<120 or (contrast>120).mean()<.60:return None
    return {'state':'PASS','reason':'REAR_GLOBAL_AND_BACKGROUND_STRUCTURE','reject':False,
            'ray_gap_median_mm':float(np.median(delta)),'ray_gap_p90_mm':float(np.percentile(delta,90)),
            'background_gap_median_mm':float(np.median(contrast)),
            'raised_connected_points':largest,'valid_fraction':float(valid.mean()),
            'water_structure':row.get('class_id') in WATER_DEPTH_CLASSES}


def _evaluate_geometry(depth,plane,row,other_boxes,profile):
    if profile.get('enabled') is False:return unknown('CLASS_DEPTH_DISABLED')
    if depth is None:return unknown('DEPTH_MISSING')
    if plane is None:return unknown('UNRELIABLE_TABLE_SUPPORT')
    if plane.get('p90_error_mm',0.)>7.:return unknown('NOISY_TABLE_REFERENCE')
    if any(row.get(flag, False) for flag in ('transparent_penetration', 'reflective',
                                            'horizontal', 'occluded', 'stacked')):
        return unknown('UNRELIABLE_MATERIAL_OR_POSE')
    outside=round_floor_evidence(depth,plane,row)
    if outside is not None:return outside
    box=row['xyxy']
    row=dict(row)
    row['_foreign_edge_mask']=foreign_edge_mask(depth,plane,box,other_boxes,gaps)
    if profile.get('bottle_print_check') or profile.get('dense_flat_check'):
        proof=bottle_print_evidence(depth,plane,row,other_boxes,profile)
        if proof is not None:
            if proof['state'] in ('FLAT','UNKNOWN','PASS'):
                return dict(proof,depth_sample_region='VISIBLE_SILHOUETTE')
    opaque_proof=opaque_core_print_evidence(depth,plane,row,other_boxes,profile)
    if opaque_proof is not None:return opaque_proof
    rear_proof=rear_structure_evidence(depth,plane,row,other_boxes,profile)
    if rear_proof is not None:return rear_proof
    if row.get('class_id') in WATER_DEPTH_CLASSES and profile.get('transparent_support_check'):
        transparent=transparent_bottle_support(depth,plane,row,other_boxes,profile)
        if transparent is not None:return transparent
    h,w=depth.shape
    bx1,by1,bx2,by2=box
    bw,bh=bx2-bx1,by2-by1
    if profile.get('bottle_positive_only') and bw >= bh:
        return unknown('BOTTLE_HORIZONTAL_OR_UNCERTAIN_POSE')
    x1,x2=max(0,int(bx1+.2*bw)),min(w,int(bx2-.2*bw))
    y1,y2=max(0,int(by1+.2*bh)),min(h,int(by2-.2*bh))
    if x2-x1<5 or y2-y1<5:return unknown('OBJECT_TOO_SMALL')
    roi=depth[y1:y2:2,x1:x2:2]
    yy,xx=np.mgrid[y1:y2:2,x1:x2:2]
    visible=np.ones(roi.shape,dtype=bool)
    edge_foreign=row['_foreign_edge_mask']
    for oa,ob,oc,od in other_boxes:
        visible &= ~((xx>=oa-3)&(xx<=oc+3)&(yy>=ob-3)&(yy<=od+3))
    if edge_foreign is not None:visible &=~edge_foreign[yy,xx]
    if visible.mean()<.60:return unknown('STACK_OR_OCCLUSION')
    valid=visible&(roi>250)&(roi<5000)&np.isfinite(roi)
    fraction=float(valid.sum()/max(1,visible.sum()))
    if fraction<profile['valid_fraction'] or valid.sum()<30:
        return dict(unknown('INSUFFICIENT_OBJECT_DEPTH'),valid_fraction=fraction)
    delta=gaps(roi,plane,xx,yy)[valid]
    result={'state':'UNKNOWN','reason':'AMBIGUOUS_GEOMETRY','reject':False,
            'valid_fraction':fraction,'ray_gap_p75_mm':float(np.percentile(delta,75)),
            'ray_gap_p90_mm':float(np.percentile(delta,90))}
    if np.percentile(delta,90)<-60 and (delta<-60).mean()>=.8:
        return dict(result,reason='BELOW_TABLE',table_relation='BELOW_TABLE',reject=False)
    local = None
    if profile.get('local_flat_check'):
        pad=max(12,int(.5*max(bw,bh)))
        surround_box=(max(0,bx1-pad),max(0,by1-pad),min(w,bx2+pad),min(h,by2+pad))
        local=fit_plane(depth,surround_box,[box]+list(other_boxes),'square',
                        min_points=80,min_inliers=65,min_fraction=.85)
        if local is not None and not local_plane_agrees(local,plane,box):
            result['local_support_status']='DISAGREES_WITH_GLOBAL_TABLE'
            local=None
        if local is not None:
            local_delta=gaps(roi,local,xx,yy)[valid]
            flat_error=float(np.percentile(np.abs(local_delta),90))
            if flat_error<=profile['flat_mm'] and profile['flat_veto']:
                return dict(result,state='FLAT',reason='LOCAL_FLAT_SUPPORT',reject=True,
                            reliable_flat=True,reliable_local_flat=True,local_flat_p90_mm=flat_error)
            delta = local_delta
            result.update(ray_gap_p75_mm=float(np.percentile(delta,75)),
                          ray_gap_p90_mm=float(np.percentile(delta,90)))
    # FLAT needs a clean local surrounding ring agreeing with the global table.
    pad=max(4,int(.15*max(bw,bh)))
    ry1,ry2=max(0,int(by1)-pad),min(h,int(by2)+pad)
    rx1,rx2=max(0,int(bx1)-pad),min(w,int(bx2)+pad)
    ry,rx=np.mgrid[ry1:ry2:2,rx1:rx2:2]
    ring=(rx<bx1)|(rx>=bx2)|(ry<by1)|(ry>=by2)
    ring &= ((rx-plane['cx'])/plane['rx'])**2+((ry-plane['cy'])/plane['ry'])**2<.95**2
    for ox1,oy1,ox2,oy2 in other_boxes:ring &= ~((rx>=ox1)&(rx<=ox2)&(ry>=oy1)&(ry<=oy2))
    surround=depth[ry1:ry2:2,rx1:rx2:2]
    ring_area = int(ring.sum())
    ring &= (surround>250)&(surround<5000)&np.isfinite(surround)
    if local is None:
        quadrants=quadrant_count(rx,ry,ring,(bx1+bx2)/2,(by1+by2)/2)
        if ring.sum()<30 or ring.sum()/max(1,ring_area)<.60 or quadrants<3:
            return dict(result,reason='UNRELIABLE_LOCAL_SUPPORT')
        if np.percentile(np.abs(gaps(surround,plane,rx,ry)[ring]),90)>7:return dict(result,reason='NONPLANAR_LOCAL_SUPPORT')
    if np.percentile(delta,75)>=profile['pass_mm'] and (delta>profile['pass_mm']).mean()>=.2:
        import cv2
        raised = np.zeros(valid.shape, dtype=np.uint8)
        raised[valid] = (delta > profile['pass_mm']).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(raised, connectivity=8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if len(stats)>1 else 0
        if largest < max(8, int(valid.sum()*.15)):
            return dict(result,reason='DISCONTINUOUS_DEPTH_STRUCTURE')
        return dict(result,state='PASS',reason='PROTRUDES_FROM_SUPPORT',
                    water_structure=row.get('class_id') in WATER_DEPTH_CLASSES)
    if np.percentile(np.abs(delta),90)<=profile['flat_mm']:
        if not profile['flat_veto']:
            return dict(result,reason='FLAT_POSITIVE_ONLY_CLASS')
        return dict(result,state='FLAT',reason='CONSISTENT_SUPPORT_PLANE',reject=True,reliable_flat=True)
    return result


def evaluate(depth,plane,row,other_boxes,profile):
    candidate=dict(row,_height_area_gate={})
    result=_evaluate_geometry(depth,plane,candidate,other_boxes,profile)
    area_stats=candidate['_height_area_gate']
    if result['state']=='PASS' and not area_stats.get('pass_area_ready',False):
        result=dict(result,state='UNKNOWN',reason='INSUFFICIENT_WHOLE_BOX_HEIGHT_AREA',reject=False)
    return dict(result,**area_stats)


class DepthValidator:
    def __init__(self,path=None):
        self.path=path or Path.home()/'national_competition/depth/depth_rules.json'
        self.rules=json.loads(self.path.read_text(encoding='utf-8'))
        self.stats=Counter()
        self.last_plane=None
        self.last_plane_time=0.
        self.recorder=None
        self.pending_support=[]

    def support_plane(self,depth,table,boxes,mode,sequence=None):
        candidate=fit_plane(depth,table.xyxy,boxes,mode)
        if candidate is None and mode=='square':
            # TABLE's rectangular bounds can include background corners in an
            # oblique view. Retry its inscribed centre without relaxing quality.
            candidate=fit_plane(depth,table.xyxy,boxes,'square_center')
            if candidate is not None:
                candidate=dict(candidate,sampling='square_inner_ellipse')
                self.stats['TABLE_PLANE_INNER_RETRY_PASS']+=1
        if candidate is None and mode=='square' and sequence is not None:
            # A crowded/oblique tabletop can leave 50–60% table points after
            # masking. Accept only after three independent, spatially broad,
            # low-noise fits agree; never lower the one-frame quality gate.
            pending=fit_plane(depth,table.xyxy,boxes,'square_center',min_inliers=300,min_fraction=.50)
            if pending is not None and pending['p90_error_mm']<=6:
                now=time.monotonic()
                self.pending_support=[entry for entry in self.pending_support if now-entry[2]<=1.5 and entry[0]!=sequence]
                self.pending_support.append((sequence,pending,now));self.pending_support=self.pending_support[-3:]
                if len(self.pending_support)==3:
                    a,b,c,d=table.xyxy
                    xx,yy=np.meshgrid(np.linspace(a+.2*(c-a),c-.2*(c-a),3),np.linspace(b+.2*(d-b),d-.2*(d-b),3))
                    predicted=[gaps(np.zeros_like(xx),entry[1],xx,yy) for entry in self.pending_support]
                    spread=float(np.max(np.ptp(np.stack(predicted),axis=0)))
                    if spread<=8:
                        candidate=dict(pending,sampling='square_temporal_inner_ellipse',temporal_support=3,consensus_max_difference_mm=spread)
                        self.stats['TABLE_PLANE_TEMPORAL_RECOVERY']+=1
        if candidate is not None and sequence is not None:
            now=time.monotonic()
            self.pending_support=[entry for entry in self.pending_support if now-entry[2]<=1.5 and entry[0]!=sequence]
            self.pending_support.append((sequence,candidate,now));self.pending_support=self.pending_support[-3:]
        previous=self.last_plane if time.monotonic()-self.last_plane_time<=2. else None
        if candidate is not None and previous is not None:
            coeff=candidate['coeff']
            inv=coeff[0]*(previous['cx']-candidate['cx'])/candidate['rx']+coeff[1]*(previous['cy']-candidate['cy'])/candidate['ry']+coeff[2]
            prior_z=1/previous['coeff'][2]
            if inv<=0 or abs(1/inv-prior_z)>25:
                self.stats['TABLE_PLANE_JUMP_IGNORED']+=1
                candidate=None
        if candidate is not None:
            self.last_plane=candidate;self.last_plane_time=time.monotonic()
            return candidate
        if previous is None or time.monotonic()-self.last_plane_time>2.:
            return None
        # A cached plane is usable only while fresh depth still independently
        # agrees over a wide part of that support surface.
        h,w=depth.shape;yy,xx=np.mgrid[0:h:4,0:w:4]
        z=depth[::4,::4]
        mask=((xx-previous['cx'])/previous['rx'])**2+((yy-previous['cy'])/previous['ry'])**2<.8**2
        for x1,y1,x2,y2 in boxes:mask &= ~((xx>=x1-3)&(xx<=x2+3)&(yy>=y1-3)&(yy<=y2+3))
        mask &= (z>250)&(z<5000)
        valid=mask&(np.abs(gaps(z,previous,xx,yy))<=7)
        quadrants={(int(x>=previous['cx']),int(y>=previous['cy'])) for x,y in zip(xx[valid],yy[valid])}
        ratio=float(valid.sum()/max(1,mask.sum()))
        ordinary=valid.sum()>=100 and ratio>=.60 and len(quadrants)>=3
        broad_current=(valid.sum()>=150 and ratio>=.50 and len(quadrants)==4 and
                       np.ptp(xx[valid])>=.4*(2*previous['rx']) and np.ptp(yy[valid])>=.3*(2*previous['ry']))
        if ordinary or broad_current:
            return dict(previous,cached=True)
        return None

    def capture(self,sequence):
        """Copy the validated packet before its ring slot can be overwritten."""
        packet=None;timing={}
        if not self.rules['enabled']:
            reason='DEPTH_DISABLED';depth=None
        elif not self.rules['alignment_verified']:
            reason='ALIGNMENT_NOT_VERIFIED';depth=None
        else:
            depth,reason,packet,timing=read_depth_packet(sequence)
        return {'packet':packet,'source_status':reason,'pair_metadata':timing}

    def annotate(self,sequence,table,objects,all_objects,mode,captured=None):
        captured=self.capture(sequence) if captured is None else captured
        packet=captured['packet'];reason=captured['source_status'];timing=captured['pair_metadata']
        depth=None
        if reason=='OK' and packet is not None:
            # Capture already checked age, sequence, registration and timing.
            # Deferred processing uses this private copy, never the live slot.
            magic,version,seq,stamp,w,h,scale,flags=struct.unpack('<4sIQQIIfI',packet[:40])
            if (magic,version,seq,w,h)!=(b'NDP1',2,sequence,640,480) or flags&3!=3:
                reason='INVALID_CAPTURED_DEPTH_PACKET'
            else:
                depth=np.frombuffer(packet[40:40+w*h*2],dtype='<u2').reshape(h,w).astype(np.float32)*scale
        info=self.annotate_arrays(sequence,table,objects,all_objects,mode,depth,reason)
        info['pair_metadata']=timing
        if self.recorder is not None and packet is not None:
            context={'table_xyxy':table.xyxy if table else None,'mode':mode,
                     'objects':json.loads(json.dumps(all_objects)),'support_plane':info['plane']}
            info['recorded_packet']=self.recorder.submit(sequence,packet,dict(timing,source_status=reason,frame_context=context))
        return info

    def annotate_arrays(self,sequence,table,objects,all_objects,mode,depth,reason='OK'):
        if not self.rules['enabled']:depth=None;reason='DEPTH_DISABLED'
        elif not self.rules['alignment_verified']:depth=None;reason='ALIGNMENT_NOT_VERIFIED'
        boxes=[r['xyxy'] for r in all_objects]
        plane=self.support_plane(depth,table,boxes,mode,sequence=sequence) if depth is not None and table else None
        foreign,ownership=foreign_water_voids(depth,plane,all_objects,gaps) if depth is not None and plane else ({},[])
        for row in objects:
            row['sequence'] = sequence
            cls=row['class_id'];profile=self.rules['profiles'][NAMES[cls]]
            others=[r['xyxy'] for r in all_objects if r is not row]
            evidence_row=dict(row,positive_exclusion_boxes=others)
            if id(row) in foreign:evidence_row['foreign_depth_void_mask']=foreign[id(row)]
            evidence=evaluate(depth,plane,evidence_row,others,profile) if depth is not None else unknown(reason)
            if profile.get('enabled') is False:evidence=unknown('CLASS_DEPTH_DISABLED')
            if cls in WATER_DEPTH_CLASSES:
                policy=self.rules.get('water_count_policy',{})
                evidence['water_pass_min_frames']=policy.get('minimum_distinct_pass_frames',3)
                evidence['water_pass_min_moments']=policy.get('round_minimum_pass_moments',2)
                evidence['water_transparent_min_frames']=policy.get('transparent_minimum_support_frames',3)
                evidence['water_transparent_min_identity']=policy.get('transparent_minimum_identity_frames',5)
                evidence['water_transparent_min_confidence']=policy.get('transparent_minimum_mean_confidence',.70)
            row['depth_state']=evidence['state'];row['depth_evidence']=evidence
            self.stats[evidence['reason']]+=1
        return {'sequence':sequence,'source_status':reason,'plane':plane,'water_void_ownership':ownership}

    def revalidate_square_frames(self,frames):
        """A static table shares a surface model, not another frame's depth.

        Learn the plane from a clean frame in this attempt, then independently
        verify wide uncovered table points in each matched archived packet.
        Never reconstruct exposure pairs, guess a plane, or bridge table moves.
        """
        references=[f for f in frames if f.get('depth',{}).get('plane') is not None]
        if not references or self.recorder is None:return 0
        reference=max(references,key=lambda f:f['depth']['plane']['inliers'])
        plane=reference['depth']['plane'];fixed_box=reference['table']['xyxy'];recovered=0
        yy,xx=np.mgrid[0:480:4,0:640:4]
        for frame in frames:
            if frame['depth'].get('plane') is not None or frame['depth'].get('source_status')!='OK':continue
            if frame.get('table') is None or iou(fixed_box,frame['table']['xyxy'])<.75:continue
            name=frame['depth'].get('recorded_packet')
            if not name:continue
            try:
                data=(self.recorder.directory/name).read_bytes()
                magic,version,seq,stamp,w,h,scale,flags=struct.unpack('<4sIQQIIfI',data[:40])
                if (magic,version,seq,w,h)!=(b'NDP1',2,frame['sequence'],640,480) or flags&3!=3:continue
                depth=np.frombuffer(data[40:614440],dtype='<u2').reshape(480,640).astype(np.float32)*scale
            except (OSError,ValueError,struct.error):continue
            z=depth[::4,::4]
            all_objects=frame['objects']+frame.get('rejected',[])
            mask=((xx-plane['cx'])/plane['rx'])**2+((yy-plane['cy'])/plane['ry'])**2<.8**2
            for row in all_objects:
                a,b,c,d=row['xyxy'];mask &=~((xx>=a-3)&(xx<=c+3)&(yy>=b-3)&(yy<=d+3))
            mask &=(z>250)&(z<5000)
            good=mask&(np.abs(gaps(z,plane,xx,yy))<=7)
            quadrants={(int(x>=plane['cx']),int(y>=plane['cy'])) for x,y in zip(xx[good],yy[good])}
            agreement=float(good.sum()/max(1,mask.sum()))
            if good.sum()<100 or agreement<.50 or len(quadrants)!=4:continue
            if np.ptp(xx[good])<.4*(2*plane['rx']) or np.ptp(yy[good])<.3*(2*plane['ry']):continue
            foreign,ownership=foreign_water_voids(depth,plane,all_objects,gaps)
            for row in frame['objects']:
                cls=row['class_id'];profile=self.rules['profiles'][NAMES[cls]]
                others=[r['xyxy'] for r in all_objects if r is not row]
                evidence_row=dict(row,positive_exclusion_boxes=others)
                if id(row) in foreign:evidence_row['foreign_depth_void_mask']=foreign[id(row)]
                ev=evaluate(depth,plane,evidence_row,others,profile)
                # Retain per-track transparent diagnostics alongside geometry.
                for key,value in row.get('depth_evidence',{}).items():
                    if key.startswith('water_') and key.endswith(('frames','moments','identity','confidence')):ev[key]=value
                row['depth_state']=ev['state'];row['depth_evidence']=ev;self.stats[ev['reason']]+=1
            frame['depth']['plane']=dict(plane,reference_sequence=reference['sequence'],
                 independently_revalidated=True,own_support_points=int(good.sum()),own_support_fraction=agreement)
            recovered+=1
        self.stats['TABLE_PLANE_ATTEMPT_REVALIDATED']+=recovered
        return recovered


def consistent_reject(rows, mode='square', class_id=None):
    if not rows or (class_id is not None and class_id not in HARD_DEPTH_CLASSES|PRINT_DEPTH_CLASSES):
        return False
    if any(r.get('depth_evidence',{}).get('state')=='PASS' for r in rows):
        return False
    groups = {}
    for i, row in enumerate(rows):
        key = row.get('moment', row.get('sequence', i)) if mode=='round' else row.get('sequence', row.get('moment', i))
        groups.setdefault(key, []).append(row)
    flat = 0
    outside_moments=0
    outside_sequences=set()
    for group in groups.values():
        outside_rows=[r for r in group if
            r.get('depth_evidence',{}).get('reliable_outside_table',False)
            and r.get('depth_evidence',{}).get('state')=='FLAT'
            and r.get('depth_evidence',{}).get('reject',False)
            and r.get('depth_evidence',{}).get('reliable_flat',False)
            and (class_id is None or r.get('class_id',class_id)==class_id or
                 (class_id in WATER_DEPTH_CLASSES and r.get('class_id') in WATER_DEPTH_CLASSES))]
        if outside_rows:
            outside_moments+=1
            for r in outside_rows:
                ev=r['depth_evidence']
                outside_sequences.update(ev.get('outside_table_sequences',
                    [r.get('sequence',r.get('moment'))]))
        if any((r.get('class_id', class_id) in HARD_DEPTH_CLASSES or
                (r.get('class_id',class_id) in PRINT_DEPTH_CLASSES and
                 r.get('depth_evidence',{}).get('reliable_planar_print',False)) or
                (r.get('class_id', class_id) in WATER_DEPTH_CLASSES and
                 r.get('depth_evidence',{}).get('reliable_bottle_print',False)) or
                r.get('depth_evidence',{}).get('reliable_outside_table',False))
               and (class_id is None or r.get('class_id', class_id)==class_id or
                    (class_id in WATER_DEPTH_CLASSES and r.get('class_id') in WATER_DEPTH_CLASSES))
               and r.get('depth_evidence',{}).get('state')=='FLAT'
               and r.get('depth_evidence',{}).get('reject',False)
               and r.get('depth_evidence',{}).get('reliable_flat',False)
               for r in group):
            flat += 1
    # Strong below-table proof is distinct from a shallow/printed surface.
    # A repeated camera sequence cannot supply independent floor evidence.
    if mode=='round' and outside_moments>=3 and len(outside_sequences)>=3:
        return True
    return flat >= 4 if mode=='round' else flat >= 3 and flat/len(groups)>=.75


def confirmed_local_flat(rows, mode='square'):
    return consistent_reject(rows, mode=mode, class_id=5)


def water_confirmed(rows,mode='square',identity_confidence=None):
    sequences=set();moments=set()
    required=max((r.get('depth_evidence',{}).get('water_pass_min_frames',3) for r in rows),default=3)
    required_moments=max((r.get('depth_evidence',{}).get('water_pass_min_moments',2) for r in rows),default=2)
    for index,row in enumerate(rows):
        ev=row.get('depth_evidence',{})
        if ev.get('state')!='PASS' or not ev.get('water_structure',False):continue
        sequences.update(ev.get('water_pass_sequences',[row.get('sequence',row.get('moment',index))]))
        moments.add(row.get('moment',index))
    if len(sequences)>=required and (mode!='round' or len(moments)>=required_moments):return True
    transparent=set();transparent_moments=set();identity=set()
    for index,row in enumerate(rows):
        for obs in row.get('capacity_observations',[(row.get('sequence',row.get('moment',index)),)]):identity.add(obs[0])
        ev=row.get('depth_evidence',{})
        if ev.get('water_transparent_support',False):
            transparent.update(ev.get('water_transparent_sequences',[row.get('sequence',row.get('moment',index))]))
            transparent_moments.add(row.get('moment',index))
    min_trans=max((r.get('depth_evidence',{}).get('water_transparent_min_frames',3) for r in rows),default=3)
    min_identity=max((r.get('depth_evidence',{}).get('water_transparent_min_identity',5) for r in rows),default=5)
    min_conf=max((r.get('depth_evidence',{}).get('water_transparent_min_confidence',.70) for r in rows),default=.70)
    if not transparent:return False
    confidence=identity_confidence if identity_confidence is not None else sum(r['confidence'] for r in rows)/max(1,len(rows))
    return len(transparent)>=min_trans and len(identity)>=min_identity and confidence>=min_conf and (mode!='round' or len(transparent_moments)>=required_moments)


def merge_burst_evidence(representative,samples):
    evidence=[]
    for sample in samples:
        candidates=[r for r in sample['objects'] if (r['class_id']==representative['class_id'] or
                    {r['class_id'],representative['class_id']}<=WATER_DEPTH_CLASSES)
                    and iou(r['xyxy'],representative['xyxy'])>=.30]
        if candidates:evidence.append(dict(max(candidates,key=lambda r:iou(r['xyxy'],representative['xyxy'])),
                                           sequence=sample['sequence']))
    positive=any(r.get('depth_evidence',{}).get('state')=='PASS'
                 and iou(r['xyxy'],representative['xyxy'])>=.30
                 for sample in samples for r in sample['objects'])
    water_pass_sequences=[r['sequence'] for r in evidence
                          if r.get('depth_evidence',{}).get('state')=='PASS'
                          and r.get('depth_evidence',{}).get('water_structure',False)]
    if representative['class_id'] in WATER_DEPTH_CLASSES:
        positive=bool(water_pass_sequences)
    transparent_sequences=[r['sequence'] for r in evidence if r.get('depth_evidence',{}).get('water_transparent_support',False)]
    planar_print=any(r.get('depth_evidence',{}).get('reliable_planar_print',False) for r in evidence)
    bottle_print=representative['class_id'] in WATER_DEPTH_CLASSES and any(
        r.get('depth_evidence',{}).get('reliable_bottle_print',False) for r in evidence)
    outside=any(r.get('depth_evidence',{}).get('reliable_outside_table',False) for r in evidence)
    outside_sequences=[r['sequence'] for r in evidence
                       if r.get('depth_evidence',{}).get('reliable_outside_table',False)]
    flat=outside or ((representative['class_id'] in HARD_DEPTH_CLASSES or bottle_print or planar_print) and any(
        r.get('depth_evidence',{}).get('state')=='FLAT'
        and r.get('depth_evidence',{}).get('reject',False)
        and r.get('depth_evidence',{}).get('reliable_flat',False) for r in evidence))
    state='PASS' if positive else 'FLAT' if flat else 'UNKNOWN'
    return {'state':state,'reject':flat and not positive,'reliable_flat':flat and not positive,
            'water_structure':bool(water_pass_sequences), 'water_pass_sequences':water_pass_sequences,
            'water_transparent_support':bool(transparent_sequences),'water_transparent_sequences':transparent_sequences,
            'water_pass_min_frames':representative.get('depth_evidence',{}).get('water_pass_min_frames',3),
            'water_pass_min_moments':representative.get('depth_evidence',{}).get('water_pass_min_moments',2),
            'reliable_bottle_print':bottle_print and flat and not positive,
            'reliable_planar_print':planar_print and flat and not positive,
            'reliable_outside_table':outside and not positive,
            'outside_table_sequences':outside_sequences if not positive else [],
            'reason':'KEYMOMENT_OUTSIDE_TABLE' if outside and not positive else
                     'KEYMOMENT_FLAT' if flat and not positive else 'DEPTH_NONVETO',
            'samples':len(evidence)}
