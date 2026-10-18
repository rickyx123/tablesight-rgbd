"""Persistent, offline CPU PP-OCRv5 worker, isolated from Ascend Python."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

import cv2
import numpy as np

from national_competition.runtime.ocr_logic import classify, deduplicate
from national_competition.runtime.ocr_exclusivity import appearance_signature
from national_competition.runtime.ocr_paper import paper_candidates,read_paper,remove_partial_paper_hits
from national_competition.runtime.ocr_budget import scan_square

ROOT = Path.home()/'national_competition'
SOCKET = Path('/dev/shm/national_ocr.sock')
READY = Path('/dev/shm/national_ocr.ready')


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def make_engine(settings):
    from rapidocr import RapidOCR, OCRVersion, ModelType
    model = ROOT/'ocr/models'
    manifest = json.loads((model/'manifest.json').read_text())
    for name in ('det.onnx','rec.onnx','cls.onnx'):
        if hashlib.sha256((model/name).read_bytes()).hexdigest()!=manifest[name]['SHA256']:
            raise ValueError('OCR model checksum mismatch: '+name)
    return RapidOCR(params={
        'Det.ocr_version':OCRVersion.PPOCRV5, 'Rec.ocr_version':OCRVersion.PPOCRV5,
        'Det.model_type':ModelType.MOBILE, 'Rec.model_type':ModelType.MOBILE,
        'Det.model_path':str(model/'det.onnx'),'Rec.model_path':str(model/'rec.onnx'),
        'Cls.model_path':str(model/'cls.onnx'),'Global.model_root_dir':str(model),
        'EngineConfig.onnxruntime.intra_op_num_threads':settings['intra_op_num_threads'],
        'EngineConfig.onnxruntime.inter_op_num_threads':1,
        'Det.limit_side_len':settings['det_limit_side_len'],'Det.limit_type':'max',
        'Global.text_score':.45,'Global.log_level':'warning'})


def prepare(image, table, known, mode, settings):
    h,w = image.shape[:2]
    x1,y1,x2,y2 = table
    x1,y1 = max(0,int(x1)),max(0,int(y1))
    x2,y2 = min(w,int(np.ceil(x2))),min(h,int(np.ceil(y2)))
    if x2-x1<12 or y2-y1<12:
        raise ValueError('Invalid TABLE crop')
    original_table=(x1,y1,x2,y2)
    tw,th=x2-x1,y2-y1
    table_polygon=np.float32([[x1+.08*tw,y1+.02*th],[x2-.08*tw,y1+.02*th],
                             [x2-.01*tw,y2-.02*th],[x1+.05*tw,y2-.02*th]])
    label_objects=[]
    if settings.get('include_table_object_labels',False):
        # Table detection describes the support surface, not the tops of
        # standing bottles. Extend OCR only along objects whose contact point
        # belongs to that surface in this very image.
        for row in known:
            if row.get('class_id')==16 or row.get('confidence',0)<.35:continue
            a,b,c,d=map(float,row['xyxy'])
            if not all(np.isfinite((a,b,c,d))) or c<=a or d<=b:continue
            px=(a+c)/2;py=d-.08*(d-b) if mode=='round' else b+.94*(d-b)
            if mode=='round':
                inside=((px-(original_table[0]+original_table[2])/2)/(tw/2))**2+((py-(original_table[1]+original_table[3])/2)/(th/2))**2<=1.05
            else:inside=cv2.pointPolygonTest(table_polygon,(px,py),True)>=-8*w/640.
            if not inside:continue
            pad=max(2.,.04*min(c-a,d-b))
            label_objects.append((max(0,int(a-pad)),max(0,int(b-pad)),
                                  min(w,int(np.ceil(c+pad))),min(h,int(np.ceil(d+pad)))))
    if mode=='round':
        margin=settings.get('round_context_margin',0.)
        px=int((x2-x1)*margin);py=int((y2-y1)*margin)
        x1,y1,x2,y2=max(0,x1-px),max(0,y1-py),min(w,x2+px),min(h,y2+py)
    for a,b,c,d in label_objects:
        x1,y1,x2,y2=min(x1,a),min(y1,b),max(x2,c),max(y2,d)
    crop = image[y1:y2,x1:x2].copy()
    ch,cw = crop.shape[:2]
    mask = np.zeros((ch,cw),dtype=np.uint8)
    if mode=='round':
        a,b,c,d=original_table
        cv2.ellipse(mask,((a+c)//2-x1,(b+d)//2-y1),(max(1,(c-a)//2),max(1,(d-b)//2)),0,0,360,255,-1)
    else:
        polygon=np.asarray(table_polygon-np.float32([x1,y1]),dtype=np.int32)
        cv2.fillPoly(mask,[polygon],255)
    for a,b,c,d in label_objects:
        cv2.rectangle(mask,(max(0,a-x1),max(0,b-y1)),(min(cw-1,c-x1),min(ch-1,d-y1)),255,-1)
    # Keep the original crop context for detection. Enforce the table mask
    # on recognized text centres instead of painting across the paper edge.
    for row in known:
        if settings.get('ocr_priority_over_yolo',False):continue
        if row['confidence']<settings['known_mask_confidence']:
            continue
        a,b,c,d=row['xyxy']
        if c<=x1 or a>=x2 or d<=y1 or b>=y2:
            continue
        cv2.rectangle(crop,(max(0,int(a)-x1),max(0,int(b)-y1)),
                      (min(cw-1,int(c)-x1),min(ch-1,int(d)-y1)),(255,255,255),-1)
    scale=min(settings['resize_scale'],settings['maximum_crop_side']/max(ch,cw))
    scaled=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    return scaled, (x1,y1), (scaled.shape[1]/cw,scaled.shape[0]/ch), mask


def restore_box(box, angle, width, height, offset, scales):
    points=np.asarray(box,dtype=float).copy()
    u,v=points[:,0].copy(),points[:,1].copy()
    if angle==90:points[:,0],points[:,1]=v,height-1-u
    elif angle==180:points[:,0],points[:,1]=width-1-u,height-1-v
    elif angle==270:points[:,0],points[:,1]=width-1-v,u
    points[:,0]=points[:,0]/scales[0]+offset[0]
    points[:,1]=points[:,1]/scales[1]+offset[1]
    return points.tolist()


def refine_word(engine,image,record,settings):
    """Recheck an exact low-score keyword with its own rectified character crop."""
    tentative=classify(record['text'],settings['minimum_text_confidence'],settings)
    if tentative is None or not hasattr(engine,'update_params'):return None,0
    points=np.asarray(record['box'],dtype=np.float32)
    width=int(round(np.linalg.norm(points[1]-points[0])))
    height=int(round(np.linalg.norm(points[3]-points[0])))
    if min(width,height)<5:return None,0
    destination=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    crop=cv2.warpPerspective(image,cv2.getPerspectiveTransform(points,destination),(width,height),borderValue=(255,255,255))
    crop=cv2.resize(crop,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
    use_det=getattr(engine,'use_det',True);text_score=getattr(engine,'text_score',.45)
    calls=0
    for fraction in (.05,.15,.30):
        padding=max(2,int(round(crop.shape[0]*fraction)))
        for flipped in (False,True):
            oriented=cv2.rotate(crop,cv2.ROTATE_180) if flipped else crop
            view=cv2.copyMakeBorder(oriented,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=(255,255,255))
            try:result=engine(view,use_det=False,text_score=0.)
            finally:engine.update_params(use_det=use_det,text_score=text_score)
            calls+=1
            if result.txts is None:continue
            for text,score in zip(result.txts,result.scores):
                candidate=classify(str(text),float(score),settings)
                if candidate and candidate['name']==tentative['name']:
                    return dict(record,text=str(text),confidence=float(score),source=record['source']+'_word_refine'),calls
    return None,calls


def high_resolution_word(engine,image,polygon,settings,deadline=None):
    points=np.asarray(polygon,np.float32)
    width=int(round(np.linalg.norm(points[1]-points[0])));height=int(round(np.linalg.norm(points[3]-points[0])))
    if min(width,height)<5:return None,0
    destination=np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    crop=cv2.warpPerspective(image,cv2.getPerspectiveTransform(points,destination),(width,height),borderValue=(255,255,255))
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    if float(((hsv[:,:,1]<40)&(hsv[:,:,2]>170)).mean())<.50:return None,0
    original={'use_det':getattr(engine,'use_det',True),'use_rec':getattr(engine,'use_rec',True),
              'use_cls':getattr(engine,'use_cls',True),'text_score':getattr(engine,'text_score',.45)}
    calls=0;best_text=None
    # Restored detector quads can rectify a horizontal word into a portrait
    # crop. Rotate BEFORE resizing so its characters are not compressed.
    rotations=(90,270,0,180) if height>width else (0,180,90,270)
    for padding in (4,10):
        for angle in rotations:
            if deadline is not None and time.monotonic()>=deadline:break
            oriented=(crop if angle==0 else cv2.rotate(crop,{90:cv2.ROTATE_90_CLOCKWISE,
                180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle]))
            oh,ow=oriented.shape[:2]
            oriented=cv2.resize(oriented,(max(8,int(ow*64/oh)),64),interpolation=cv2.INTER_CUBIC)
            view=cv2.copyMakeBorder(oriented,padding,padding,padding,padding,cv2.BORDER_CONSTANT,value=(255,255,255))
            try:result=engine(view,use_det=False,use_rec=True,use_cls=True,text_score=0.)
            finally:engine.update_params(**original)
            calls+=1
            for text,score in zip(result.txts or [],result.scores if result.scores is not None else []):
                if best_text is None or float(score)>best_text['confidence']:
                    best_text={'text':str(text),'confidence':float(score),'diagnostic_text':True,
                               'reading_rotation':angle}
                candidate=classify(str(text),float(score),settings,paper_label=True)
                if candidate:return dict(candidate,reading_rotation=angle),calls
    return best_text if best_text and best_text['confidence']>=.45 else None,calls


def recognize(engine, request, settings):
    start=time.monotonic()
    if start>=float(request.get('deadline_monotonic',float('inf'))):
        return {'status':'OK','sequence':request['sequence'],'seconds':0.,'table_valid':request.get('table') is not None,
                'hits':[],'raw_texts':[],'ocr_calls':0,'budget_exhausted':True,'skipped_expired_request':True}
    path=Path(request['image_path']).resolve()
    allowed=(ROOT.resolve(), (Path.home()/'Desktop/result').resolve())
    if not any(path.is_relative_to(base) for base in allowed):
        raise ValueError('OCR image must be inside the competition workspace or result archive')
    image=cv2.imread(str(path))
    if image is None:raise ValueError('OCR image is unreadable')
    table=request.get('table')
    table_valid=table is not None and not request.get('diagnostic_only',False)
    if not table_valid:
        if not request.get('diagnostic_only',False):
            raise ValueError('TABLE missing: full-image OCR is diagnostic only')
        table=[0,0,image.shape[1],image.shape[0]]
    try:
        crop,offset,scales,mask=prepare(image,table,request.get('objects',[]),request['mode'],settings)
    except ValueError:
        table_valid=False
        crop,offset,scales,mask=prepare(image,[0,0,image.shape[1],image.shape[0]],[],request['mode'],settings)
    height,width=crop.shape[:2]
    raw,hits=[],[];calls=0;fallback_calls=0
    base_box_thresh=getattr(getattr(getattr(engine,'text_det',None),'postprocess_op',None),'box_thresh',None)
    def collect(output,angle,view_width,view_height,view_offset,source):
        nonlocal calls
        retry=False
        if output.txts is None:return False
        for text,score,box in zip(output.txts,output.scores,output.boxes):
            restored=restore_box(box,angle,view_width,view_height,view_offset,scales)
            record={'text':str(text),'confidence':float(score),'box':restored,'rotation':angle,'source':source}
            raw.append(record)
            if settings.get('word_refinement_enabled',False) and .45<=float(score)<settings['minimum_text_confidence']:
                refined,refine_calls=refine_word(engine,image,record,settings);calls+=refine_calls
                if refined is not None:
                    raw.append(refined);text=refined['text'];score=refined['confidence']
            if .45<=float(score)<settings['minimum_text_confidence'] and classify(str(text),settings['minimum_text_confidence'],settings):retry=True
            candidate=classify(str(text),float(score),settings)
            if candidate is None or not table_valid:continue
            centre=np.asarray(restored).mean(axis=0)-np.asarray(offset)
            cx,cy=int(centre[0]),int(centre[1])
            if not (0<=cy<mask.shape[0] and 0<=cx<mask.shape[1] and mask[cy,cx]):continue
            px,py=centre+np.asarray(offset)
            if not settings.get('ocr_priority_over_yolo',False) and any(row['confidence']>=settings['known_mask_confidence']
                   and row['xyxy'][0]<=px<=row['xyxy'][2]
                   and row['xyxy'][1]<=py<=row['xyxy'][3] for row in request.get('objects',[])):
                continue
            hits.append(dict(candidate,box=restored,rotation=angle,
                             appearance_signature=appearance_signature(image,restored)))
        return retry
    def run(view,angle,view_offset,source,fallback=False):
        nonlocal calls,fallback_calls
        rotated=view if angle==0 else cv2.rotate(view,{90:cv2.ROTATE_90_CLOCKWISE,
                       180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        if fallback:
            try:output=engine(rotated,box_thresh=.30)
            finally:
                if base_box_thresh is not None:engine.update_params(box_thresh=base_box_thresh)
        else:output=engine(rotated)
        calls+=1;fallback_calls+=int(fallback)
        if not fallback:cv2.imwrite(str(path.with_name(path.stem+f'_crop_{angle}.jpg')),rotated)
        return collect(output,angle,view.shape[1],view.shape[0],view_offset,source)
    high_word_regions=0;paper_scans=0;paper_polygons=[];read_pages=[]
    bounded=settings.get('bounded_'+request['mode']+'_enabled',False)
    deadline=min(float(request.get('deadline_monotonic',float('inf'))),start+settings.get('frame_budget_seconds_'+request['mode'],6.))
    if bounded:
        priors=request.get('prior_regions',[])
        if request.get('source_to_reference') is not None:
            inverse=np.linalg.inv(np.asarray(request['source_to_reference'],np.float64))
            priors=[dict(row,box=cv2.perspectiveTransform(np.float32([row['box']]),inverse)[0].tolist()) for row in priors]
        hits,raw,calls,paper_scans,high_word_regions=scan_square(engine,image,crop,offset,scales,mask,settings,
            restore_box,high_resolution_word,deadline,priors,mode=request['mode'])
        for hit in hits:hit['appearance_signature']=appearance_signature(image,hit['box'])
        if not table_valid:hits=[]
    if not bounded and table_valid and settings.get('whole_paper_scan_enabled',False):
        paper_polygons=paper_candidates(crop,offset,scales)
        for polygon in paper_polygons:
            centre=np.asarray(polygon).mean(axis=0)-np.asarray(offset);cx,cy=int(centre[0]),int(centre[1])
            if not (0<=cy<mask.shape[0] and 0<=cx<mask.shape[1] and mask[cy,cx]):continue
            candidate,records,paper_calls=read_paper(engine,image,polygon,settings)
            raw.extend(records);calls+=paper_calls;paper_scans+=int(paper_calls>0)
            if candidate:
                hits.append(dict(candidate,appearance_signature=appearance_signature(image,candidate['box'])))
                read_pages.append(polygon)
    if bounded:pass
    elif request.get('source_to_reference') is not None:
        candidates=[]
        for angle in settings['rotations']:
            rotated=crop if angle==0 else cv2.rotate(crop,{90:cv2.ROTATE_90_CLOCKWISE,
                180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
            original_rec=getattr(engine,'use_rec',True);original_cls=getattr(engine,'use_cls',True)
            try:detected=engine(rotated,use_rec=False,use_cls=False)
            finally:engine.update_params(use_rec=original_rec,use_cls=original_cls)
            calls+=1
            if detected.boxes is not None:
                candidates.extend((restore_box(box,angle,width,height,offset,scales),angle) for box in detected.boxes)
        unique=[]
        def bounds(polygon):
            points=np.asarray(polygon);return points[:,0].min(),points[:,1].min(),points[:,0].max(),points[:,1].max()
        def area(polygon):
            a,b,c,d=bounds(polygon);return max(0.,c-a)*max(0.,d-b)
        for polygon,angle in sorted(candidates,key=lambda entry:area(entry[0]),reverse=True):
            a,b,c,d=bounds(polygon);duplicate=False
            for prior,_ in unique:
                x,y,z,w=bounds(prior)
                overlap=max(0.,min(c,z)-max(a,x))*max(0.,min(d,w)-max(b,y))
                small=min(area(polygon),area(prior));union=area(polygon)+area(prior)-overlap
                if overlap/max(1.,union)>.45 or (overlap/max(1.,small)>.90 and small/max(area(polygon),area(prior),1.)>.40):
                    duplicate=True;break
            if not duplicate:unique.append((polygon,angle))
        for polygon,angle in unique:
            centre=np.asarray(polygon).mean(axis=0)-np.asarray(offset);cx,cy=int(centre[0]),int(centre[1])
            if not (0<=cy<mask.shape[0] and 0<=cx<mask.shape[1] and mask[cy,cx]):continue
            # A page already read as a whole must not yield a second fragmented item.
            if any(cv2.pointPolygonTest(np.asarray(page,np.float32),tuple(np.asarray(polygon).mean(axis=0)),False)>=0 for page in read_pages):continue
            candidate,word_calls=high_resolution_word(engine,image,polygon,settings)
            calls+=word_calls;high_word_regions+=int(word_calls>0)
            if candidate:
                raw.append({'text':candidate['text'],'confidence':candidate['confidence'],'box':polygon,
                            'rotation':angle,'source':'high_resolution_word'})
                if not candidate.get('diagnostic_text',False):
                    hits.append(dict(candidate,box=polygon,rotation=angle,
                        appearance_signature=appearance_signature(image,polygon)))
    else:
        for angle in settings['rotations']:run(crop,angle,offset,'table')
    if not bounded and not hits and table_valid and settings.get('tiled_fallback_enabled',False) and request['mode']=='square':
        # Uniform overlapping context scans: no fixed target position or keyword correction.
        tile_w=max(12,width//2);tile_h=max(12,height//2)
        for tile,(tx,ty) in enumerate(((0,0),(width-tile_w,0),(0,height-tile_h),(width-tile_w,height-tile_h))):
            view=crop[ty:ty+tile_h,tx:tx+tile_w]
            view_offset=(offset[0]+tx/scales[0],offset[1]+ty/scales[1])
            for angle in settings['rotations']:
                retry=run(view,angle,view_offset,f'tile_{tile}',True)
                if retry:
                    ew,eh=int(width*.65),int(height*.65)
                    ex=max(0,min(width-ew,tx+tile_w//2-ew//2));ey=max(0,min(height-eh,ty+tile_h//2-eh//2))
                    expanded=crop[ey:ey+eh,ex:ex+ew]
                    expanded_offset=(offset[0]+ex/scales[0],offset[1]+ey/scales[1])
                    run(expanded,angle,expanded_offset,f'tile_{tile}_context',True)
    hits=remove_partial_paper_hits(deduplicate(hits))
    report={'status':'OK','sequence':request['sequence'],'seconds':time.monotonic()-start,
            'table_valid':table_valid, 'diagnostic_only':not table_valid,
            'ocr_calls':calls,
            'tiled_fallback_calls':fallback_calls,
            'high_resolution_word_regions':high_word_regions,
            'whole_paper_regions':paper_scans,
            'budget_exhausted':bounded and time.monotonic()>=deadline,
            'image_path':str(path),'raw_texts':raw,'hits':hits}
    annotated=image.copy()
    for hit in hits:
        cv2.polylines(annotated,[np.rint(hit['box']).astype(np.int32)],True,(0,220,255),2)
    # Chinese labels use Pillow; retain the source image for actual evidence.
    from PIL import Image,ImageDraw,ImageFont
    pil=Image.fromarray(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB))
    font_path=Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    draw=ImageDraw.Draw(pil)
    for hit in hits:
        a,b=np.asarray(hit['box']).min(axis=0)
        draw.text((max(0,a),max(0,b-22)),f'{hit["text"]} {hit["confidence"]:.2f}',font=font,fill='#FF9500')
    annotated_path=path.with_name(path.stem+'_detected.jpg')
    pil.save(annotated_path)
    report['annotated_path']=str(annotated_path)
    report['source_image_size']=[image.shape[1],image.shape[0]]
    report['source_status']=request.get('source_status','RGB_640')
    if request.get('source_to_reference') is not None:
        matrix=np.asarray(request['source_to_reference'],dtype=float)
        scale=np.diag([image.shape[1]/640.,image.shape[0]/480.,1.])
        reference=cv2.warpPerspective(cv2.resize(image,(640,480),interpolation=cv2.INTER_AREA),matrix@scale,(640,480))
        for rows in (raw,hits):
            for row in rows:
                row['box']=cv2.perspectiveTransform(np.float32([row['box']]),matrix)[0].tolist()
                if rows is hits:row['appearance_signature']=appearance_signature(reference,row['box'])
        report['coordinate_space']='RGBD_640x480'
    atomic_json(path.with_suffix('.ocr.json'),report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'ocr/ocr_rules.json')
    parser.add_argument('--request',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    settings=json.loads(args.config.read_text(encoding='utf-8'))
    start=time.monotonic()
    engine=make_engine(settings)
    print(f'OCR_READY model=PP-OCRv5_mobile init={time.monotonic()-start:.3f}s',flush=True)
    if args.request:
        result=recognize(engine,json.loads(args.request.read_text()),settings)
        if args.output:atomic_json(args.output,result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return
    SOCKET.unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(str(SOCKET));os.chmod(SOCKET,0o600);server.listen(2);server.settimeout(1.)
        atomic_json(READY,{'pid':os.getpid(),'model':'PP-OCRv5_mobile','version':settings['version']})
        stop=False
        def shutdown(*_):
            nonlocal stop
            stop=True
        signal.signal(signal.SIGTERM,shutdown)
        signal.signal(signal.SIGINT,shutdown)
        try:
            while not stop:
                try:connection,_=server.accept()
                except socket.timeout:continue
                with connection:
                    connection.settimeout(10.)
                    try:
                        line=connection.makefile('rb').readline(1024*1024)
                        request=json.loads(line)
                        settings=json.loads(args.config.read_text(encoding='utf-8'))
                        result=recognize(engine,request,settings)
                    except Exception as error:
                        result={'status':'ERROR','error':f'{type(error).__name__}: {error}'}
                        print(result,flush=True)
                    try:connection.sendall((json.dumps(result,ensure_ascii=False)+'\n').encode('utf-8'))
                    except OSError:pass
        finally:
            SOCKET.unlink(missing_ok=True);READY.unlink(missing_ok=True)


if __name__=='__main__':main()
