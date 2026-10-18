"""Bounded OCR sampling from sequence-matched RGB; CPU work stays off the UI thread."""
from __future__ import annotations
import json
from pathlib import Path
import queue
import socket
import subprocess
import threading
import time
from collections import Counter

import cv2

from national_competition.runtime.depth_validation import rgb_for_sequence
from national_competition.runtime.ocr_logic import aggregate_frames, summary_lines, status_text,deduplicate
from national_competition.runtime.ocr_exclusivity import confirmed_regions
from national_competition.runtime.ocr_source import read_ocr_source,transform_rectangle
from national_competition.runtime.capture_policy import read_capture_policy
import numpy as np

ROOT=Path.home()/'national_competition'
SOCKET=Path('/dev/shm/national_ocr.sock')


class OcrSession:
    def __init__(self,mode,directory,settings=None,fallback_image_path=None):
        self.mode=mode
        self.directory=Path(directory)/'ocr'
        if settings is None:
            try:settings=json.loads((ROOT/'ocr/ocr_rules.json').read_text(encoding='utf-8'))
            except (OSError,ValueError):settings={'enabled':False,'subjects':[],'version':'CONFIG_MISSING'}
        self.settings=settings
        self.fallback_image_path=Path(fallback_image_path) if fallback_image_path else None
        self.fallback_request=None
        self.final_report=None
        self.frames=[];self.errors=[];self.seen=set();self.submitted=0;self.skipped=0;self.last=-float('inf')
        self.lock=threading.Lock();self.queue=queue.Queue();self.collecting=True
        self.cancel=threading.Event();self.thread=None;self.prewarm=None
        self.deadline=None;self.budget_started=None
        self.created_at=time.monotonic()
        if settings.get('enabled'):
            self.directory.mkdir(parents=True,exist_ok=True)
            self.thread=threading.Thread(target=self._run,daemon=True)
            self.thread.start()

    def _start_budget(self):
        if self.settings.get('bounded_'+self.mode+'_enabled',False) and self.deadline is None:
            self.budget_started=time.monotonic()
            self.deadline=min(self.budget_started+self.settings['total_budget_seconds_'+self.mode],
                              self.created_at+(22. if self.mode=='square' else 25.))

    def submit(self,frame):
        if not self.settings.get('enabled') or not self.collecting:
            return
        policy=read_capture_policy(frame['sequence'])
        if policy and policy['compatibility_mode']:
            self.disable_for_compatibility()
            return
        if not frame.get('table'):
            if self.fallback_request is None:
                image=rgb_for_sequence(frame['sequence'])
                if image is not None:
                    self.fallback_request=self._save_request(frame,image,diagnostic=True)
            return
        seq=frame['sequence']
        limit=self.settings['maximum_frames_'+self.mode]
        interval=self.settings['sample_interval_'+self.mode]
        if seq in self.seen or self.submitted>=limit or frame['elapsed']-self.last<interval:
            return
        matrix=None;source_status='RGB_640'
        if self.settings.get('high_resolution_rgb_enabled',False):
            image,matrix,source_status=read_ocr_source(seq)
        else:image=None
        if image is None:image=rgb_for_sequence(seq);source_status='LOW_RES_FALLBACK'
        if image is None:
            self.errors.append(f'RGB_PAIR_MISSING sequence={seq}')
            return
        request=self._save_request(frame,image,matrix=matrix,source_status=source_status)
        if request is None:return
        self._start_budget()
        self.seen.add(seq);self.last=frame['elapsed'];self.submitted+=1
        self.queue.put(request)

    def disable_for_compatibility(self):
        if self.settings.get('disabled_reason')=='HIGH_RES_ALIGNMENT_FAILED':return
        with self.lock:
            self.settings=dict(self.settings,enabled=False,disabled_reason='HIGH_RES_ALIGNMENT_FAILED')
            self.collecting=False
            self.final_report=None
        self.cancel.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=.3)

    def _save_request(self,frame,image,diagnostic=False,matrix=None,source_status='RGB_640'):
        seq=frame['sequence']
        path=self.directory/f'frame_{seq:09d}.jpg'
        if not cv2.imwrite(str(path),image):
            self.errors.append(f'RGB_SAVE_FAILED sequence={seq}')
            return
        request={'sequence':seq,'image_path':str(path),'mode':self.mode,
                 'table':None if diagnostic else frame['table']['xyxy'],
                 'objects':frame.get('objects',[]), 'diagnostic_only':diagnostic}
        request['source_status']=source_status
        if matrix is not None:
            inverse=np.linalg.inv(matrix)
            if request['table'] is not None:request['table']=transform_rectangle(request['table'],inverse)
            request['objects']=[dict(row,xyxy=transform_rectangle(row['xyxy'],inverse)) for row in request['objects']]
            request['source_to_reference']=matrix.tolist()
        path.with_suffix('.request.json').write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
        return request

    def _ensure_attempt(self):
        if not self.settings.get('enabled') or self.submitted:
            return
        request=self.fallback_request
        if request is None and self.fallback_image_path is not None:
            try:
                if time.time()-self.fallback_image_path.stat().st_mtime<=1.2:
                    image=cv2.imread(str(self.fallback_image_path))
                    if image is not None:
                        request=self._save_request({'sequence':-time.monotonic_ns(),'objects':[]},
                                                   image,diagnostic=True)
            except OSError:pass
        if request is not None:
            self._start_budget()
            self.submitted+=1
            self.queue.put(request)
        else:
            self.errors.append('OCR_NO_RGB_IMAGE: 无可用图像，文字识别无法执行')

    def _run(self):
        try:
            with (self.directory/'prewarm.log').open('w') as stream:
                self.prewarm=subprocess.Popen(['bash',str(ROOT/'board/prewarm_ocr.sh')],stdout=stream,stderr=stream)
            while not self.cancel.is_set():
                try:request=self.queue.get(timeout=.1)
                except queue.Empty:
                    if not self.collecting:break
                    continue
                if self.mode=='square' and self.settings.get('bounded_square_enabled',False):
                    with self.lock:
                        request=dict(request,prior_regions=deduplicate([hit for frame in self.frames for hit in frame.get('hits',[])]))
                    if len(self.frames)>=2:
                        required=Counter()
                        for frame in self.frames:
                            required |= Counter(hit['name'] for hit in frame.get('hits',[]))
                        confirmed=Counter({row['name']:row['count'] for row in self.report()['results']})
                        if all(confirmed[name]>=count for name,count in required.items()):
                            self.skipped+=1;self.queue.task_done();continue
                result=self._request(request)
                with self.lock:
                    if result.get('status')=='OK':self.frames.append(result)
                    else:self.errors.append(result.get('error','OCR_REQUEST_FAILED'))
                self.queue.task_done()
        except Exception as error:
            with self.lock:self.errors.append(f'{type(error).__name__}: {error}')
        finally:
            if self.prewarm is not None:self.prewarm.poll()

    def _request(self,request):
        end=time.monotonic()+self.settings['request_timeout_seconds']
        if self.deadline is not None:
            # A session deadline is shared by all requests, not restarted per frame.
            end=self.deadline
            reserve=1. if self.mode=='round' else .3
            request=dict(request,deadline_monotonic=min(end-reserve,time.monotonic()+self.settings.get('frame_budget_seconds_'+self.mode,6.)))
        while not self.cancel.is_set() and time.monotonic()<end:
            try:
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
                    client.settimeout(max(.1,end-time.monotonic()))
                    client.connect(str(SOCKET))
                    client.sendall((json.dumps(request)+'\n').encode('utf-8'))
                    line=client.makefile('rb').readline(2*1024*1024)
                    return json.loads(line)
            except (FileNotFoundError,ConnectionRefusedError):
                self.cancel.wait(.05)
            except (OSError,ValueError) as error:
                return {'status':'ERROR','error':str(error)}
        return {'status':'ERROR','error':'OCR_BUDGET_LIMIT' if self.deadline is not None else 'OCR_WORKER_TIMEOUT'}

    def report(self):
        if self.final_report is not None:
            return self.final_report
        with self.lock:
            frames=list(self.frames);errors=list(self.errors)
        results=aggregate_frames(frames,self.settings) if self.settings.get('enabled') else []
        if self.mode=='square' and self.settings.get('ocr_priority_over_yolo',False):
            regions=confirmed_regions({'config':self.settings,'results':results,'frames':frames})
            spatial_counts=Counter(r['name'] for r in regions)
            results=[dict(row,count=min(row['count'],spatial_counts[row['name']])) for row in results if spatial_counts[row['name']]]
        status='DISABLED' if not self.settings.get('enabled') else (
            'ERROR' if errors and not frames else 'NO_VALID_SAMPLES' if not self.collecting and not self.submitted
            else ('DIAGNOSTIC_ONLY' if frames and not any(f.get('table_valid',True) for f in frames)
                  else 'COMPLETE') if not self.collecting and len(frames)+self.skipped==self.submitted
            else 'INCOMPLETE' if not self.collecting else 'PROCESSING')
        if not self.collecting and self.deadline is not None and time.monotonic()>=self.deadline and len(frames)+self.skipped<self.submitted:
            status='TIME_LIMIT'
        return {'enabled':self.settings.get('enabled',False),'status':status,
                'disabled_reason':self.settings.get('disabled_reason'),
                'config':self.settings,'submitted_frames':self.submitted,
                'processed_frames':len(frames),'executed_calls':sum(f.get('ocr_calls',0) for f in frames),
                'skipped_after_confirmation':self.skipped,
                'budget_seconds':self.settings.get('total_budget_seconds_'+self.mode) if self.deadline is not None else None,
                'budget_exhausted':self.deadline is not None and time.monotonic()>=self.deadline,
                'results':results,'frames':frames,'errors':errors}

    def finish(self):
        if self.final_report is not None:
            return self.final_report
        self._ensure_attempt()
        self.collecting=False
        wait=self.settings.get('finish_timeout_seconds',0.) if self.settings.get('enabled') else 0.
        if self.deadline is not None:wait=max(0.,self.deadline-time.monotonic())
        if self.thread:self.thread.join(timeout=wait)
        report=self.report()
        self.final_report=report
        self.cancel.set()
        if self.settings.get('enabled'):
            (self.directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        return report

    def preview(self):
        report=self.report()
        return report, f'文字识别：{status_text(report)}，已处理 {report["processed_frames"]} 帧'

    def abort(self):
        self.collecting=False;self.cancel.set()


def write_ocr_text(path,report):
    Path(path).write_text('\n'.join(summary_lines(report))+'\n',encoding='utf-8')
