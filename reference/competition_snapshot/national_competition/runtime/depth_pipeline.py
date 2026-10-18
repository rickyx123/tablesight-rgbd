"""Ordered background depth work over private, sequence-matched packets."""
import copy
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from national_competition.ui.competition_visualizer import UiDetection


class DepthPipeline:
    def __init__(self,validator,mode,cancel):
        self.validator=validator;self.mode=mode;self.cancel=cancel
        # A single ordered worker preserves the validator's plane history.
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='depth-filter')
        self.jobs=[];self.completed=0;self.seconds=0.;self.maximum_pending=0
        self.started=time.monotonic();self.first_depth_started=None
        self.lock=threading.Lock();self.closed=False
        self.stop=threading.Event();self.latest=None

    def submit(self,frame):
        captured=frame.pop('_depth_capture',None)
        if captured is None:return
        payload={key:copy.deepcopy(frame[key]) for key in ('sequence','table','objects','rejected')}
        if 'square_surface' in frame:payload['square_surface']=copy.deepcopy(frame['square_surface'])
        with self.lock:
            self.maximum_pending=max(self.maximum_pending,len(self.jobs)+1-self.completed)
        future=self.executor.submit(self._process,payload,captured)
        self.jobs.append((frame,future))

    def _process(self,frame,captured):
        if self.cancel.is_set() or self.stop.is_set():return None
        table=UiDetection(**frame['table']) if frame['table'] else None
        started=time.monotonic()
        if self.first_depth_started is None:self.first_depth_started=started
        info=self.validator.annotate(frame['sequence'],table,frame['objects'],
            frame['objects']+frame['rejected'],self.mode,captured=captured)
        elapsed=time.monotonic()-started
        with self.lock:
            self.completed+=1;self.seconds+=elapsed
            self.latest={'sequence':frame['sequence'],'packet':captured.get('packet'),'table':frame['table'],
                         'objects':frame['objects'],'rejected':frame['rejected'],'depth':info}
            if 'square_surface' in frame:self.latest['square_surface']=frame['square_surface']
        return {'objects':frame['objects'],'rejected':frame['rejected'],
                'depth':info,'depth_processing_seconds':elapsed}

    def finish(self,deadline=None):
        started=time.monotonic()
        limited=False
        try:
            for frame,future in self.jobs:
                while not self.cancel.is_set():
                    if deadline is not None and time.monotonic()>=deadline and not future.done():
                        limited=True;self.stop.set();break
                    try:
                        result=future.result(timeout=.1)
                        if result is not None:frame.update(result)
                        break
                    except TimeoutError:continue
        finally:self.close(wait=not limited)
        return {'status':'CANCELLED' if self.cancel.is_set() else 'TIME_LIMIT' if limited else 'OK','submitted_frames':len(self.jobs),
                'processed_frames':self.completed,'total_depth_seconds':self.seconds,
                'first_depth_started_delay_seconds':None if self.first_depth_started is None else self.first_depth_started-self.started,
                'finish_wait_seconds':time.monotonic()-started,'maximum_pending_frames':self.maximum_pending}

    def close(self,wait=True):
        if not self.closed:
            self.closed=True;self.executor.shutdown(wait=wait,cancel_futures=not wait)
