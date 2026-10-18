"""Official table adapter; existing test entry and recognition rules stay intact."""
import argparse
import json
import time
from pathlib import Path

from national_competition.runtime.judge_protocol import StabilityGate
from national_competition.runtime.depth_validation import read_rgbd
from national_competition.ui.competition_visualizer import read_worker_csv
from national_competition.ui.timed_test import TimedUi, atomic_json


class CompetitionTableUi(TimedUi):
    def event(self, name, **values):
        print(json.dumps(dict(event=name,table=self.args.table,
            monotonic=time.monotonic(),**values),ensure_ascii=False),flush=True)

    def sample(self):
        try:
            if self.args.rotation_sent is not None:
                rotating_table = self.args.table==3
                self.set_state(status='等待转向',detail=(
                    '转向后等待2.5秒，圆桌保持转动即可' if rotating_table else
                    '转向后至少等待2.5秒；桌框辅助确认，最多等待7秒后继续识别'))
                gate = StabilityGate(self.args.rotation_sent,require_stability=not rotating_table)
                while not self.cancel.is_set():
                    now=time.monotonic()
                    try:
                        before=self.args.result_csv.stat()
                        seq,rows=read_worker_csv(self.args.result_csv)
                        after=self.args.result_csv.stat()
                        fresh=(before.st_mtime_ns==after.st_mtime_ns and
                            0 <= time.time()-after.st_mtime <= .4 and
                            after.st_mtime >= self.args.rotation_wall)
                        candidates=[r for r in rows if r.class_id==16 and r.confidence>=.25]
                        table=max(candidates,key=lambda r:r.confidence) if candidates else None
                        # Same-sequence RGB-D must also exist; CSV alone is insufficient.
                        rgb,depth,_=read_rgbd(seq) if seq is not None else (None,None,None)
                        fresh=fresh and rgb is not None and depth is not None
                    except (OSError,ValueError,IndexError):
                        seq,table,fresh=None,None,False
                    if gate.observe(seq,table.xyxy if table else None,now,fresh):
                        event_name=('pan_stable' if gate.stability_confirmed else
                            'pan_fixed_wait_complete' if rotating_table else 'pan_wait_limit_continue')
                        self.event(event_name,rotation_sent_monotonic=self.args.rotation_sent,
                            stable_since_monotonic=gate.stable_since,distinct_frames=gate.frames,
                            wait_reason=gate.release_reason,stability_confirmed=gate.stability_confirmed,
                            elapsed_since_rotation=now-self.args.rotation_sent)
                        break
                    time.sleep(.025)
                if self.cancel.is_set():return
            self.event('table_recognition_begin')
            # Pan waiting has its own clock; vision's bounded budgets start here.
            self.start=time.monotonic()
            super().sample()
        except Exception as error:
            atomic_json(self.args.output_path.with_suffix('.json'),
                {'status':'FAILED','table':self.args.table,'error':str(error)})
            self.set_state(status='启动失败',detail=str(error),final=[],done=True)
            self.event('table_failed',error=str(error))

    def set_state(self, **values):
        super().set_state(**values)
        if values.get('status')=='识别中':
            self.event('effective_sampling_begin',rotation_sent_monotonic=self.args.rotation_sent)

    def update(self):
        # Let sample's finally finish evidence recording before closing the window.
        with self.lock:done=self.state['done']
        if done:
            if not self.thread.is_alive():
                self.root.destroy()
            else:self.root.after(100,self.update)
            return
        super().update()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--table',type=int,choices=(1,2,3),required=True)
    parser.add_argument('--model-version',required=True)
    parser.add_argument('--output-path',type=Path,required=True)
    parser.add_argument('--rotation-sent',type=float)
    parser.add_argument('--rotation-wall',type=float,default=0.)
    args=parser.parse_args()
    args.table_label={1:'方桌1',2:'方桌2',3:'圆桌'}[args.table]
    args.mode='round' if args.table==3 else 'square'
    args.frame=Path('/dev/shm/astra_yolo_frame.jpg')
    args.result_csv=Path('/dev/shm/astra_yolo_result.txt')
    args.fullscreen,args.status_file,args.demo=False,None,False
    ui=CompetitionTableUi(args)
    ui.run()
    try:report=json.loads(args.output_path.with_suffix('.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):return 1
    return 0 if report.get('status')=='COMPLETE' and ui.state['done'] else 1


if __name__=='__main__':
    raise SystemExit(main())
