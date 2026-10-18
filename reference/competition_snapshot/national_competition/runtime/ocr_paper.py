"""Geometry-first reading of printed paper, independent of keyword text boxes."""
import re,time
import cv2,numpy as np
from national_competition.runtime.ocr_logic import classify,normalized

def remove_partial_paper_hits(hits):
    kept=[]
    for hit in hits:
        partial=normalized(hit['text']);points=np.asarray(hit['box']);a,b=points.min(axis=0);c,d=points.max(axis=0)
        area=max(1.,(c-a)*(d-b));duplicate=False
        if hit.get('generic_paper_text'):
            for other in hits:
                full=normalized(other['text'])
                if len(full)<=len(partial) or partial not in full or other['confidence']<hit['confidence']:continue
                p=np.asarray(other['box']);x,y=p.min(axis=0);z,w=p.max(axis=0)
                overlap=max(0,min(c,z)-max(a,x))*max(0,min(d,w)-max(b,y))
                if overlap/min(area,max(1.,(z-x)*(w-y)))>=.30:duplicate=True;break
        if not duplicate:kept.append(hit)
    return kept

def paper_target(text,score,settings):
    known=classify(text,score,settings,paper_label=True)
    if known:return known
    value=normalized(text)
    if any(normalized(word) in value for word in settings.get('negative_keywords',[])):return None
    if not settings.get('generic_paper_text_enabled',False) or score<settings.get('generic_paper_minimum_confidence',.90):return None
    # Only geometric paper candidates reach this path; package OCR stays diagnostic.
    if not 1<=len(value)<=24 or not re.search(r'[\u3400-\u9fffA-Za-z0-9]',value):return None
    return {'name':str(text).strip(),'goal_id':None,'keyword':str(text).strip(),
            'text':str(text).strip(),'confidence':float(score),'generic_paper_text':True}

def paper_candidates(crop,offset,scales,broad=False,edge_context=False):
    h,w=crop.shape[:2];hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV);candidates=[]
    size=max(3,int(min(h,w)*.012)|1)
    thresholds=[(5,185,35),(8,185,35),(12,185,35)]
    if edge_context:thresholds.append((65,220,size))
    if broad:thresholds.extend(((25,185,size),(55,155,size)))
    for saturation,brightness,kernel in thresholds:
        white=((hsv[:,:,1]<saturation)&(hsv[:,:,2]>brightness)).astype(np.uint8)*255
        if kernel!=35:white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel)))
        white=cv2.morphologyEx(white,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel,kernel)))
        contours,_=cv2.findContours(white,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area=cv2.contourArea(contour);x,y,bw,bh=cv2.boundingRect(contour)
            if edge_context and sum((x<=1,y<=1,x+bw>=w-1,y+bh>=h-1))>=2:continue
            if broad:
                if not .02*w*h<area<.50*w*h or area/(bw*bh)<(.30 if edge_context else .50):continue
            elif not .025*w*h<area<.45*w*h or area/(bw*bh)<.60:continue
            hull=cv2.convexHull(contour);points=cv2.approxPolyDP(hull,.035*cv2.arcLength(hull,True),True).reshape(-1,2).astype(np.float32)
            if len(points)!=4:
                if not edge_context:continue
                rectangle=cv2.minAreaRect(hull)
                if area/max(1.,rectangle[1][0]*rectangle[1][1])<.35:continue
                points=cv2.boxPoints(rectangle)
            total=points.sum(axis=1);difference=points[:,0]-points[:,1]
            if edge_context:
                centre=points.mean(axis=0)
                ordered=points[np.argsort(np.arctan2(points[:,1]-centre[1],points[:,0]-centre[0]))]
                ordered=np.roll(ordered,-int(np.argmin(ordered.sum(axis=1))),axis=0)
            else:
                ordered=np.array([points[np.argmin(total)],points[np.argmax(difference)],
                                  points[np.argmax(total)],points[np.argmin(difference)]],np.float32)
            if len(np.unique(ordered,axis=0))!=4:continue
            ordered[:,0]=ordered[:,0]/scales[0]+offset[0];ordered[:,1]=ordered[:,1]/scales[1]+offset[1]
            box=[float(ordered[:,0].min()),float(ordered[:,1].min()),float(ordered[:,0].max()),float(ordered[:,1].max())]
            duplicate=False
            for _,_,prior in candidates:
                inter=max(0.,min(box[2],prior[2])-max(box[0],prior[0]))*max(0.,min(box[3],prior[3])-max(box[1],prior[1]))
                union=(box[2]-box[0])*(box[3]-box[1])+(prior[2]-prior[0])*(prior[3]-prior[1])-inter
                if inter/max(1.,union)>.65:duplicate=True;break
            if not duplicate:candidates.append((area,ordered,box))
    return [polygon for _,polygon,_ in sorted(candidates,key=lambda r:r[0],reverse=True)[:6 if broad else 4]]

def read_paper(engine,image,polygon,settings,deadline=None):
    width=max(20,int(max(np.linalg.norm(polygon[1]-polygon[0]),np.linalg.norm(polygon[2]-polygon[3]))))
    height=max(20,int(max(np.linalg.norm(polygon[3]-polygon[0]),np.linalg.norm(polygon[2]-polygon[1]))))
    destination=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    warp=cv2.getPerspectiveTransform(polygon,destination)
    paper=cv2.warpPerspective(image,warp,(width,height),borderValue=(255,255,255))
    ink=cv2.cvtColor(paper,cv2.COLOR_BGR2GRAY)<150
    mx=max(1,int(width*.08));my=max(1,int(height*.08))
    ink[:my]=False;ink[-my:]=False;ink[:,:mx]=False;ink[:,-mx:]=False
    yy,xx=np.where(ink)
    if len(xx)<20 or len(xx)/(width*height)>.28:return None,[],0
    a,b,c,d=max(0,int(xx.min())-6),max(0,int(yy.min())-6),min(width,int(xx.max())+7),min(height,int(yy.max())+7)
    word=paper[b:d,a:c]
    word_polygon=cv2.perspectiveTransform(np.float32([[[a,b],[c,b],[c,d],[a,d]]]),np.linalg.inv(warp))[0].tolist()
    original={'use_det':getattr(engine,'use_det',True),'use_rec':getattr(engine,'use_rec',True),
              'use_cls':getattr(engine,'use_cls',True),'text_score':getattr(engine,'text_score',.45)}
    records=[];targets=[];calls=0
    for angle in (0,90,180,270):
        if deadline is not None and time.monotonic()>=deadline:break
        view=word if angle==0 else cv2.rotate(word,{90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        try:result=engine(view,use_det=False,use_rec=True,use_cls=False,text_score=0.)
        finally:engine.update_params(**original)
        calls+=1
        for text,score in zip(result.txts or [],result.scores if result.scores is not None else []):
            records.append({'text':str(text),'confidence':float(score),'box':word_polygon,'rotation':angle,'source':'whole_paper'})
            target=paper_target(str(text),float(score),settings)
            if target:targets.append(dict(target,box=word_polygon,rotation=angle))
    if not targets:return None,records,calls
    if settings.get('partial_labels_enabled',False):
        # Stable canonical names join "洁" and "洁精" across independent
        # images. Prefer a complete label, then the more informative partial.
        mapped=[t for t in targets if t.get('goal_id')]
        if mapped:
            winner=max(mapped,key=lambda t:(not t.get('partial_label',False),
                len(normalized(t['text'])),t['confidence']))
            return winner,records,calls
    # One physical page yields one name. Closely scored partial characters
    # cannot override a complete reading of the same printed ink.
    strongest=max(target['confidence'] for target in targets)
    complete=[target for target in targets if target['confidence']>=strongest-.02]
    winner=max(complete,key=lambda target:(len(normalized(target['text'])),target['confidence']))
    return winner,records,calls
