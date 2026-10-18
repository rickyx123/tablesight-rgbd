"""Square OCR: a finite search, followed by native resolution reading."""
import time
import cv2
import numpy as np
from national_competition.runtime.ocr_paper import paper_candidates,read_paper,paper_target

def scan_square(engine,image,crop,offset,scales,mask,settings,restore_box,word_reader,deadline,prior_regions=None,mode='square'):
    hits=[];raw=[];calls=0;pages=[];paper_scans=0;word_scans=0
    def inside(polygon):
        cx,cy=(np.asarray(polygon).mean(axis=0)-np.asarray(offset)).astype(int)
        return 0<=cy<mask.shape[0] and 0<=cx<mask.shape[1] and bool(mask[cy,cx])
    # Confirm already located paper text first on a new independent image.
    # Recognition is rerun; previous words/scores are never copied as evidence.
    for prior in prior_regions or []:
        if time.monotonic()>=deadline:break
        polygon=np.roll(np.asarray(prior['box']),int(prior.get('rotation',0))//90,axis=0).tolist()
        if not inside(polygon):continue
        target,count=word_reader(engine,image,polygon,settings,deadline=deadline)
        calls+=count;word_scans+=int(count>0)
        if target and target.get('diagnostic_text') and prior.get('generic_paper_text'):
            target=paper_target(target['text'],target['confidence'],settings)
        if target and not target.get('diagnostic_text'):
            hits.append(dict(target,box=polygon,rotation=0));pages.append(polygon)
            raw.append(dict(text=target['text'],confidence=target['confidence'],box=polygon,source='independent_confirmation',rotation=0))
    if prior_regions and len(hits)==len(prior_regions):
        return hits,raw,calls,paper_scans,word_scans
    geometric_pages=paper_candidates(crop,offset,scales,broad=True,edge_context=mode=='round')
    for polygon in geometric_pages:
        if time.monotonic()>=deadline:break
        if not inside(polygon):continue
        if any(cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(np.asarray(page).mean(axis=0)),False)>=0 for page in pages):continue
        target,records,count=read_paper(engine,image,polygon,settings,deadline)
        calls+=count;raw.extend(records);paper_scans+=int(count>0)
        if target:hits.append(target);pages.append(polygon)
    # Keep detection orientation coverage; unlike the former fallback, never
    # repeat whole-table recognition in sixteen overlapping tile scans.
    candidates=[];height,width=crop.shape[:2]
    for angle in ((0,90) if mode=='round' else (0,90,180,270)):
        if time.monotonic()>=deadline:break
        view=crop if not angle else cv2.rotate(crop,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        original={key:getattr(engine,key,True) for key in ('use_det','use_rec','use_cls')}
        try:output=engine(view,use_det=True,use_rec=False,use_cls=False)
        finally:engine.update_params(**original)
        calls+=1
        if output.boxes is None:continue
        for box in output.boxes:
            polygon=restore_box(box,angle,width,height,offset,scales)
            if not inside(polygon):continue
            centre=tuple(np.asarray(polygon).mean(axis=0))
            if any(cv2.pointPolygonTest(np.asarray(page,np.float32),centre,False)>=0 for page in pages):continue
            candidates.append(polygon)
    candidates.sort(key=lambda p:cv2.contourArea(np.asarray(p,np.float32)),reverse=True)
    unique=[]
    for polygon in candidates:
        points=np.asarray(polygon);a,b=points.min(axis=0);c,d=points.max(axis=0)
        duplicate=False
        for prior in unique:
            p=np.asarray(prior);x,y=p.min(axis=0);z,w=p.max(axis=0)
            overlap=max(0,min(c,z)-max(a,x))*max(0,min(d,w)-max(b,y))
            if overlap/max(1,min((c-a)*(d-b),(z-x)*(w-y)))>.65:duplicate=True;break
        if not duplicate:unique.append(polygon)
    for polygon in unique[:settings.get('maximum_word_regions_square',6)]:
        if time.monotonic()>=deadline:break
        target,count=word_reader(engine,image,polygon,settings,deadline=deadline)
        calls+=count;word_scans+=int(count>0)
        if target:
            raw.append(dict(text=target['text'],confidence=target['confidence'],box=polygon,source='bounded_native_word',rotation=0))
            # Unknown words need paper geometry, rather than treating every
            # high confidence package label as a new object.
            if target.get('diagnostic_text'):
                if any(cv2.pointPolygonTest(np.asarray(page,np.float32),tuple(np.asarray(polygon).mean(axis=0)),False)>=0
                       for page in geometric_pages):
                    target=paper_target(target['text'],target['confidence'],settings)
                else:target=None
            if target:hits.append(dict(target,box=polygon,rotation=0))
    return hits,raw,calls,paper_scans,word_scans
