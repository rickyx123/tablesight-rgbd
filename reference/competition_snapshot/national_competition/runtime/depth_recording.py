"""Bounded background recording of the exact packets used for depth decisions."""
import json
import queue
import threading
import time
from pathlib import Path


class DepthEvidenceRecorder:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.queue=queue.Queue(maxsize=64);self.rows=[];self.errors=[];self.seen=set()
        self.closed=False;self.dropped=0
        self.thread=threading.Thread(target=self._write,daemon=True);self.thread.start()

    def submit(self,sequence,packet,timing):
        if self.closed or sequence in self.seen:return None
        name=f'frame_{sequence:09d}.bin'
        try:self.queue.put_nowait((sequence,name,packet,dict(timing)))
        except queue.Full:
            self.dropped+=1;return None
        self.seen.add(sequence)
        return name

    def _write(self):
        while True:
            item=self.queue.get()
            try:
                if item is None:return
                sequence,name,packet,timing=item
                path=self.directory/name;temp=path.with_suffix('.tmp')
                temp.write_bytes(packet);temp.replace(path)
                self.rows.append({'sequence':sequence,'packet':name,'bytes':len(packet),'pair_metadata':timing})
            except Exception as error:self.errors.append(str(error))
            finally:self.queue.task_done()

    def finish(self,timeout=20.):
        if not self.closed:
            self.closed=True;self.queue.put(None);self.thread.join(timeout=timeout)
        report={'status':'OK' if not self.errors and not self.dropped and not self.thread.is_alive() else 'INCOMPLETE',
                'saved_frames':len(self.rows),'dropped_frames':self.dropped,'errors':list(self.errors),
                'frames':list(self.rows),'directory':str(self.directory)}
        temp=self.directory/'manifest.tmp';temp.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(self.directory/'manifest.json')
        return report

    def drain(self,timeout=2.):
        end=time.monotonic()+timeout
        while self.queue.unfinished_tasks and time.monotonic()<end:time.sleep(.005)
        return self.queue.unfinished_tasks==0
