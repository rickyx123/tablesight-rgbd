"""One click, one table, automatic capture and final rule-format output."""
import argparse
import json
import math
import subprocess
import signal
import threading
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

from national_competition.config import GOAL_IDS, NAMES
from national_competition.runtime.result_writer import write_result, write_comparison, comparison_lines
from national_competition.runtime import timed_fusion as fusion
from national_competition.runtime.round_surface import surface_from_capture,surface_from_arrays
from national_competition.runtime.square_surface import (
    SquareTableGeometry,
    surface_from_capture as square_surface_from_capture,
    surface_from_arrays as square_surface_from_arrays)
from national_competition.runtime.depth_validation import DepthValidator, read_rgbd,decode_rgbd_packet
from national_competition.runtime.depth_recording import DepthEvidenceRecorder
from national_competition.runtime.depth_pipeline import DepthPipeline
from national_competition.runtime.ocr_source import read_ocr_source
from national_competition.runtime.capture_policy import read_capture_policy
from national_competition.runtime.ocr_exclusivity import confirmed_regions,exclude_detections,motion_context,live_regions
from national_competition.runtime.ocr_client import OcrSession, write_ocr_text
from national_competition.runtime.ocr_logic import status_text as ocr_status_text
from national_competition.ui.competition_visualizer import (
    CompetitionUi, UiDetection, read_worker_csv, render_dashboard, depth_to_image, FONT_22)


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def preview_pair(sequence,processed,now):
    """Prefer processed same-packet preview, with a fresh-pair fallback."""
    rgb,depth,status=read_rgbd(sequence) if sequence is not None else (None,None,'WAITING_RESULT')
    if processed is not None and processed.get('packet'):
        prior_rgb,prior_depth,prior_status=decode_rgbd_packet(processed['packet'],processed['sequence'],now)
        if prior_rgb is not None and prior_depth is not None:
            return processed['sequence'],prior_rgb,prior_depth,prior_status,processed
    return sequence,rgb,depth,status,None


def round_preview_crop(table,size=(640,480)):
    """Fixed camera viewport, even when the table translates or changes size."""
    return (0,0,*size)


class TimedUi(CompetitionUi):
    def __init__(self, args):
        import cv2
        cv2.setNumThreads(1)
        super().__init__(args)
        self.root.title('国赛方桌定时识别' if args.mode == 'square' else '国赛圆桌自适应识别')
        if getattr(args,'table_label',None):self.root.title('国赛比赛识别 - '+args.table_label)
        self.root.geometry('1360x800+0+30')
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.result_popup = None
        self.popup_shown = False
        self.ocr_session = None
        self.round_geometry = fusion.RoundTableGeometry() if args.mode == 'round' else None
        self.round_live_geometry = fusion.RoundTableGeometry() if args.mode=='round' else None
        self.square_geometry=SquareTableGeometry() if args.mode=='square' else None
        self.square_live_geometry=SquareTableGeometry() if args.mode=='square' else None
        self.depth_validator = DepthValidator()
        self.live_depth_validator = DepthValidator()
        self.live_pair_cache = None
        self.preview_live_depth_ready = False
        self.state = {'status': '准备中', 'detail': '相机和模型加载中',
                      'final': None, 'elapsed': 0., 'started': None, 'done': False}
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()

    def set_state(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self, seen, deadline, defer_depth=False):
        while time.monotonic() < deadline and not self.cancel.is_set():
            try:
                if time.time() - self.args.result_csv.stat().st_mtime > 1.2:
                    time.sleep(.01)
                    continue
                sequence, detections = read_worker_csv(self.args.result_csv)
            except (OSError, ValueError, IndexError):
                time.sleep(.01)
                continue
            if sequence is not None and sequence not in seen:
                seen.add(sequence)
                captured_at=time.monotonic()
                table, objects, rejected = fusion.scope(detections, self.args.mode)
                raw_table = table
                captured=None
                square_surface=None
                if self.args.mode=='square':
                    captured=self.depth_validator.capture(sequence)
                    geometry=getattr(self,'square_geometry',None)
                    if geometry is None:self.square_geometry=geometry=SquareTableGeometry()
                    square_surface=geometry.capture(captured,raw_table,detections,sequence)
                    if square_surface is not None:
                        located=UiDetection(16,raw_table.confidence,tuple(square_surface['xyxy']))
                        table,objects,rejected=fusion.scope(detections,'square',located,
                                                          square_surface['polygon'],square_surface['rim_contacts'])
                if self.round_geometry is not None:
                    captured=self.depth_validator.capture(sequence)
                    surface=surface_from_capture(captured,raw_table,detections,sequence)
                    if (surface is None and raw_table is not None and
                            self.round_geometry.unverified_expansion(raw_table.xyxy)):
                        surface=surface_from_capture(captured,raw_table,detections,sequence,
                            reference_box=self.round_geometry.shape_reference)
                    stable_table = self.round_geometry.update(raw_table, sequence,surface=surface)
                    if self.round_geometry.moved:
                        self.depth_validator.last_plane=None
                        self.depth_validator.last_plane_time=0.
                        self.depth_validator.pending_support=[]
                    if stable_table is None:
                        table, objects = None, []
                        rejected = [asdict(d) for d in detections if d.class_id != 16]
                    else:
                        table, objects, rejected = fusion.scope(detections, 'round', stable_table)
                if defer_depth and captured is None:captured=self.depth_validator.capture(sequence)
                depth_info = ({'sequence':sequence,'source_status':captured['source_status'],'plane':None}
                    if defer_depth else self.depth_validator.annotate(sequence,table,objects,objects+rejected,self.args.mode,captured=captured))
                frame = {'sequence': sequence, 'objects': objects, 'rejected': rejected,
                        'table': asdict(table) if table else None, 'depth': depth_info,
                        'raw_table': asdict(raw_table) if raw_table else None,
                        'elapsed': captured_at-self.capture_start}
                if self.args.mode=='square':
                    frame['square_surface']=square_surface or {'source':'yolo_fallback','polygon':None}
                try:
                    input_policy=json.loads(self.args.result_csv.with_suffix('.input.json').read_text())
                    if input_policy.get('sequence')==sequence:frame['yolo_input']=input_policy
                except (OSError,ValueError,AttributeError,TypeError):pass
                if defer_depth:frame['_depth_capture']=captured
                if self.ocr_session is not None and self.ocr_sampling:
                    self.ocr_session.submit(frame)
                frame['capture_seconds']=time.monotonic()-captured_at
                return frame
            time.sleep(.01)
        return None

    def process_depth(self,frames):
        pipeline=getattr(self,'depth_pipeline',None)
        if pipeline is not None:
            self.set_state(status='深度汇总中',detail='采样已完成，正在汇总后台深度结果')
            try:self.depth_pipeline_report=pipeline.finish(deadline=(self.start+26. if self.args.mode=='round' and hasattr(self,'start') else None))
            finally:self.depth_pipeline=None
            return self.depth_pipeline_report['total_depth_seconds']
        started=time.monotonic()
        self.set_state(status='深度筛选中',detail=f'采样完成，正在处理 {len(frames)} 帧深度')
        for index,frame in enumerate(frames):
            if self.cancel.is_set():break
            captured=frame.pop('_depth_capture',None)
            if captured is None:continue
            table=UiDetection(**frame['table']) if frame['table'] else None
            before=time.monotonic()
            frame['depth']=self.depth_validator.annotate(frame['sequence'],table,frame['objects'],
                frame['objects']+frame['rejected'],self.args.mode,captured=captured)
            frame['depth_processing_seconds']=time.monotonic()-before
            if index%5==0:self.set_state(detail=f'深度筛选 {index+1}/{len(frames)} 帧')
        return time.monotonic()-started

    def square(self):
        attempts = [];pipelines=[]
        for attempt in range(2):
            self.depth_pipeline=DepthPipeline(self.depth_validator,self.args.mode,self.cancel)
            frames, seen = [], set()
            started = time.monotonic()
            deadline = started + fusion.SQUARE_TIMEOUT
            next_time = started
            while len(frames) < fusion.SQUARE_FRAMES and time.monotonic() < deadline:
                if self.cancel.wait(max(0., next_time-time.monotonic())):
                    return [], attempts, 'CANCELLED'
                frame = self.snapshot(seen, deadline, defer_depth=True)
                if frame is None:
                    break
                frames.append(frame)
                self.depth_pipeline.submit(frame)
                next_time = started+len(frames)*fusion.SQUARE_INTERVAL
                valid = sum(f['table'] is not None for f in frames)
                self.set_state(status='识别中', detail=f'方桌 {len(frames)}/7帧；桌面有效 {valid}')
            attempts.append(frames)
            self.process_depth(frames)
            pipelines.append(self.depth_pipeline_report)
            valid = sum(f['table'] is not None for f in frames)
            if valid >= 5:
                if self.depth_validator.rules['enabled']:
                    self.depth_recorder.drain()
                    self.depth_validator.revalidate_square_frames(frames)
                    if not any(f['depth'].get('plane') is not None for f in frames) and attempt==0:
                        self.set_state(detail='深度桌面参考不足，0.5秒后完整重采')
                        if self.cancel.wait(.5):return [],attempts,'CANCELLED'
                        continue
                final, decisions = fusion.square_fuse(frames)
                pure, pure_decisions = fusion.square_fuse(frames, use_depth=False)
                return final, {'attempts': attempts, 'tracks': decisions,
                               'pipelines':pipelines,
                               'pure_yolo_detections': [asdict(d) for d in pure],
                               'pure_yolo_tracks': pure_decisions}, ('DEPTH_REFERENCE_UNAVAILABLE' if
                                   self.depth_validator.rules['enabled'] and not any(f['depth'].get('plane') is not None for f in frames) else 'OK')
            if attempt == 0:
                self.set_state(detail='桌面不足，0.5秒后完整重采')
                if self.cancel.wait(.5):
                    return [], attempts, 'CANCELLED'
        reason='SAMPLING_INSUFFICIENT' if len(attempts[-1])<5 else 'TABLE_INSUFFICIENT'
        return [], {'attempts': attempts,'pipelines':pipelines}, reason

    def round(self):
        self.depth_pipeline=DepthPipeline(self.depth_validator,self.args.mode,self.cancel)
        frames, seen, deltas = [], set(), []
        sampling={'raw_frames':0,'raw_table_frames':0,'table_frames':0,'single_frame_moments':0}
        coverage, signed, direction = 0., 0., 0
        started = time.monotonic()
        end = min(started+fusion.ROUND_MAX_SECONDS,self.start+25.) if hasattr(self,'start') else started+fusion.ROUND_MAX_SECONDS
        for index in range(fusion.ROUND_MAX_MOMENTS):
            target = started+index*fusion.ROUND_INTERVAL
            if target >= end or self.cancel.wait(max(0., target-time.monotonic())):
                break
            samples = []
            burst_end = min(end, time.monotonic()+fusion.ROUND_BURST_SECONDS)
            while len(samples) < fusion.ROUND_BURST:
                snapshot = self.snapshot(seen, burst_end, defer_depth=True)
                if snapshot is None:
                    break
                sampling['raw_frames']+=1
                sampling['raw_table_frames']+=int(snapshot['raw_table'] is not None)
                sampling['table_frames']+=int(snapshot['table'] is not None)
                if snapshot['table']:
                    samples.append(snapshot)
                else:
                    # No usable TABLE: keep audit evidence without expensive depth work.
                    snapshot.pop('_depth_capture',None)
            if len(samples) >= 2:
                frame = fusion.burst_vote(samples)
                frame['samples'] = samples
                frame['elapsed'] = time.monotonic()-self.capture_start
                if frames:
                    delta = fusion.rotation_delta(frames[-1], frame)
                    # Angular limits assume adjacent 0.75-second moments. Do not
                    # estimate motion across a missing moment (possible aliasing).
                    if frame['elapsed']-frames[-1]['elapsed'] > 1.2:
                        delta = None
                    frame['rotation_delta'] = delta
                    if delta is not None:
                        deltas.append(delta)
                        if direction == 0 and len(deltas) >= 3:
                            direction = 1 if statistics_median(deltas[-5:]) >= 0 else -1
                        if direction == 0 or direction*delta > 0:
                            signed += delta
                            coverage += abs(delta)
                frame['signed_rotation'] = signed
                frames.append(frame)
                for sample in samples:self.depth_pipeline.submit(sample)
            else:
                sampling['single_frame_moments']+=int(len(samples)==1)
                for sample in samples:sample.pop('_depth_capture',None)
            turns = coverage/(2*math.pi)
            self.set_state(status='YOLO/深度/OCR同步识别', detail=f'圆桌 {len(frames)}时刻 / {turns:.2f}圈\n采样最多剩余 {max(0., end-time.monotonic()):.1f}秒')
            # Fixed 20-second capture while validating the rotation estimator.
            # Estimated coverage must not make identical runs end at different times.
        sampling['capture_seconds']=time.monotonic()-started
        sampling['depth_frames_finished_during_capture']=self.depth_pipeline.completed
        sampling['depth_processing_seconds']=self.process_depth([s for f in frames for s in f['samples']])
        sampling['pipeline']=self.depth_pipeline_report
        for frame in frames:
            frame.update(fusion.burst_vote(frame['samples']))
        if not frames:
            reason=('SAMPLING_INSUFFICIENT' if sampling['raw_table_frames']==sampling['raw_frames']
                    and self.round_geometry.table is not None else 'TABLE_INSUFFICIENT')
            return [], {'frames': frames, 'turns': coverage/(2*math.pi),
                        'sampling':sampling,'geometry': self.round_geometry.diagnostics()}, reason
        partial=len(frames)<fusion.ROUND_MIN_MOMENTS
        final, decisions = fusion.round_fuse(frames,allow_partial=partial)
        pure, pure_decisions = fusion.round_fuse(frames, use_depth=False,allow_partial=partial)
        reason = ('PARTIAL_SAMPLING' if partial else
                  'DEPTH_PROCESSING_TIME_LIMIT' if sampling['pipeline']['status']=='TIME_LIMIT' else
                  'OK' if coverage >= 2*math.pi else 'COVERAGE_UNCONFIRMED')
        return final, {'frames': frames, 'turns': coverage/(2*math.pi),
                       'geometry': self.round_geometry.diagnostics(),
                       'sampling':sampling,
                       'algorithm_version': 'round_track_depth_capacity_v3_20261003',
                       'pure_yolo_detections': [asdict(d) for d in pure],
                       'pure_yolo_decisions': pure_decisions,
                       'decisions': decisions}, reason

    def sample(self):
        target = self.args.output_path
        tag = time.strftime('%Y%m%d_%H%M%S') + f'_{time.time_ns()%1000000000:09d}'
        archive = target.parent / 'archive' / self.args.mode / tag
        archive.mkdir(parents=True, exist_ok=True)
        self.depth_recorder=DepthEvidenceRecorder(archive/'rgbd')
        self.depth_validator.recorder=self.depth_recorder
        self.ocr_session = OcrSession(self.args.mode,archive,settings=(
            {'enabled':False,'subjects':[]} if not getattr(self.args,'ocr_enabled',True) else None),
            fallback_image_path=getattr(self.args,'frame',None))
        self.ocr_sampling = False
        if target.exists():
            target.replace(archive/'previous_result.txt')
        metadata = {'mode': self.args.mode, 'table': self.args.table, 'model': self.args.model_version,
                    'sampler_version':'parallel_depth_20s_20261007',
                    'depth_rules': self.depth_validator.rules['version'],
                    'depth_rules_snapshot': self.depth_validator.rules,
                    'depth_alignment_verified': self.depth_validator.rules['alignment_verified'],
                    'status': 'RUNNING', 'started_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        latest_json = target.with_suffix('.json')
        atomic_json(latest_json, metadata)
        try:
            self.capture_start = time.monotonic()
            # Model may still be loading after login; this is preparation, not sampling.
            first = self.snapshot(set(), time.monotonic()+(20. if self.args.mode=='round' else 45.), defer_depth=True)
            if self.cancel.is_set():
                return
            if first is None:
                raise RuntimeError('规定准备时间内未收到新的模型结果' if self.args.mode=='round' else '45秒内未收到新的模型结果')
            capture_policy=read_capture_policy(first['sequence'])
            metadata['capture_policy']=capture_policy
            compatibility_mode=bool(capture_policy and capture_policy['compatibility_mode'])
            if compatibility_mode:
                self.ocr_session.disable_for_compatibility()
                self.set_state(detail='兼容采集：原生640×480，OCR已关闭，物品检测继续')
            if ((self.depth_validator.rules['enabled'] and self.depth_validator.rules['alignment_verified'])
                or self.round_geometry is not None):
                self.set_state(detail='正在准备独立桌面帧')
                calibration_seen={first['sequence']}
                calibration_end=time.monotonic()+3.
                preparation=[first]
                while time.monotonic()<calibration_end and not self.cancel.is_set():
                    geometry_ready=self.round_geometry is None or self.round_geometry.table is not None
                    if self.args.mode=='square':geometry_ready=self.square_geometry.surface is not None
                    if len(preparation)>=4 and geometry_ready:break
                    prepared=self.snapshot(calibration_seen,calibration_end,defer_depth=True)
                    if prepared is None:break
                    preparation.append(prepared)
                self.process_depth(preparation[-4:])
                preparation.clear()
                if self.cancel.is_set():return
            else:
                self.process_depth([first])
            if compatibility_mode:self.start=time.monotonic()
            self.capture_start = time.monotonic()
            self.set_state(started=self.capture_start, status='识别中')
            self.ocr_sampling = True
            final, evidence, reason = self.square() if self.args.mode == 'square' else self.round()
            # Sampling and its depth work are complete. Square OCR waiting can
            # now use live depth without competing with the capture pipeline.
            self.preview_live_depth_ready = self.args.mode=='square'
            self.ocr_sampling = False
            if self.cancel.is_set() or reason == 'CANCELLED':
                self.ocr_session.abort()
                return
            elapsed = time.monotonic()-self.capture_start
            if self.args.mode=='square':
                captured_frames=[f for attempt in evidence['attempts'] for f in attempt]
                capture_elapsed=max((f['elapsed'] for f in captured_frames),default=0.)
                depth_seconds=sum(f.get('depth_processing_seconds',0.) for f in captured_frames)
                depth_wait=sum(p.get('finish_wait_seconds',0.) for p in evidence['pipelines'])
            else:
                capture_elapsed=evidence['sampling']['capture_seconds']
                depth_seconds=evidence['sampling']['depth_processing_seconds']
                depth_wait=evidence['sampling']['pipeline']['finish_wait_seconds']
            self.set_state(status='文字识别中',detail='正在汇总物品识别结果')
            ocr_started = time.monotonic()
            ocr_report = self.ocr_session.finish()
            recording_report=self.depth_recorder.finish(timeout=max(0.,self.start+28.-time.monotonic()) if self.args.mode=='round' else 20.)
            ocr_finish_seconds = time.monotonic()-ocr_started
            contexts=motion_context(evidence,self.args.mode)
            text_regions=confirmed_regions(ocr_report,contexts)
            if ocr_report.get('frames'):
                spatial_counts=Counter(r['name'] for r in text_regions)
                ocr_report['results']=[dict(row,count=min(row['count'],spatial_counts[row['name']]))
                    for row in ocr_report.get('results',[]) if spatial_counts[row['name']]]
            evidence['raw_pure_yolo_detections']=evidence.get('pure_yolo_detections',[])
            final,final_removed=exclude_detections(final,text_regions,contexts,evidence)
            pure_filtered,pure_removed=exclude_detections(evidence.get('pure_yolo_detections',[]),text_regions,contexts,evidence)
            evidence['pure_yolo_detections']=pure_filtered
            ocr_report['exclusive_regions']=text_regions
            ocr_report['yolo_excluded']={'pure':pure_removed,'depth':final_removed}
            counts = Counter(d.class_id for d in final)
            pure_detections = evidence.get('pure_yolo_detections', [])
            pure_counts = Counter(d['class_id'] for d in pure_detections)
            rows = [{'class_id': cls, 'count': count, 'table': self.args.table} for cls, count in counts.items()]
            reason_text = {'OK': '筛选完成', 'TABLE_INSUFFICIENT': '桌面检测不足，本桌未计入结果',
                           'PARTIAL_SAMPLING':'采样不足，输出已确认物品；圆桌覆盖不完整',
                           'SAMPLING_INSUFFICIENT':'采样帧数不足，本桌未计入结果；详见采样记录',
                           'DEPTH_REFERENCE_UNAVAILABLE':'深度桌面参考不足；保留RGB结果，本轮未完成深度过滤',
                           'DEPTH_PROCESSING_TIME_LIMIT':'达到深度处理时限；保留未处理物品，详见深度记录',
                           'COVERAGE_UNCONFIRMED': '旋转覆盖未确认，输出保守结果'}.get(reason, reason)
            extra_items = ocr_report['results']
            write_comparison(target, pure_counts, counts, self.args.table, reason_text,extra_items=extra_items)
            write_comparison(archive/'result.txt', pure_counts, counts, self.args.table, reason_text,extra_items=extra_items)
            extra_ocr = f'\n文字识别：{ocr_status_text(ocr_report)}；已处理 {ocr_report.get("processed_frames",0)} 帧\n'
            for comparison_path in (target,archive/'result.txt'):
                with comparison_path.open('a',encoding='utf-8') as stream:
                    stream.write(extra_ocr)
            ocr_target = target.with_name(target.stem+'_ocr.txt')
            write_ocr_text(ocr_target,ocr_report)
            write_ocr_text(archive/'ocr.txt',ocr_report)
            competition_target = target.with_name(target.stem+'_competition.txt')
            unknown_rows = [dict(r,table=self.args.table) for r in ocr_report['results'] if r.get('goal_id')]
            write_result(competition_target, rows, unknown_rows=unknown_rows)
            write_result(archive/'competition.txt', rows, unknown_rows=unknown_rows)
            def inventory(detections):
                inventory_counts=Counter(d.class_id for d in detections)
                return [{'goal_id':GOAL_IDS[cls], 'name':NAMES[cls], 'count':count,
                         'table':self.args.table,'source':'YOLO'} for cls,count in sorted(inventory_counts.items())]+[
                         dict(item,table=self.args.table,source='OCR') for item in extra_items]
            items=inventory(final)
            pure_items=inventory([UiDetection(d['class_id'],d['confidence'],tuple(d['xyxy'])) for d in pure_detections])
            metadata.update(status='COMPLETE', reason=reason, elapsed_seconds=time.monotonic()-self.capture_start,
                            partial_result=reason=='PARTIAL_SAMPLING',
                            total_elapsed_seconds=time.monotonic()-self.start,
                            object_processing_seconds=elapsed,
                            sampler_version='parallel_depth_20s_20261007',
                            capture_elapsed_seconds=capture_elapsed,depth_processing_seconds=depth_seconds,
                            depth_finish_wait_seconds=depth_wait,
                            depth_statistics=dict(self.depth_validator.stats),
                            pure_yolo_counts=dict(pure_counts), depth_counts=dict(counts),
                            pure_yolo_detections=pure_detections,
                            comparison_path=str(target), competition_path=str(competition_target),
                            ocr=ocr_report, ocr_path=str(ocr_target), ocr_finish_seconds=ocr_finish_seconds,
                            items=items, pure_items=pure_items,depth_recording=recording_report,
                            total_count=sum(item['count'] for item in items),
                            rows=rows, final_detections=[asdict(d) for d in final], evidence=evidence)
            atomic_json(latest_json, metadata)
            atomic_json(archive/'evidence.json', metadata)
            summary = ['方桌7帧结果' if self.args.mode == 'square' else '圆桌自适应结果',
                       f'采样耗时：{capture_elapsed:.2f}秒；结束后等待深度：{depth_wait:.2f}秒',
                       f'后台深度累计运算：{depth_seconds:.2f}秒（与采样同时进行）',
                       f'总耗时（含相机、模型准备）：{time.monotonic()-self.start:.2f}秒',
                       f'状态：{reason_text}',
                       '深度：PASS / FLAT / UNKNOWN；眼镜不做深度过滤',
                       ('深度证据统计（含校准）：'
                        f'立体确认 {sum(self.depth_validator.stats.get(k,0) for k in ("PROTRUDES_FROM_SUPPORT","BOTTLE_VISIBLE_STRUCTURE","VISIBLE_3D_STRUCTURE"))}；'
                        f'平面 {self.depth_validator.stats["CONSISTENT_SUPPORT_PLANE"]}；'
                       f'局部平面 {self.depth_validator.stats["LOCAL_FLAT_SUPPORT"]}；'
                       f'水瓶贴图 {self.depth_validator.stats.get("DENSE_PLANAR_BOTTLE_PRINT",0)}；'
                       f'通用贴图 {self.depth_validator.stats.get("DENSE_PLANAR_PRINT",0)}；'
                        f'桌下 {self.depth_validator.stats["BELOW_TABLE"]}；'
                        f'眼镜豁免 {self.depth_validator.stats["CLASS_DEPTH_DISABLED"]}'), '']
            if self.args.mode == 'square':
                last_attempt = evidence['attempts'][-1]
                summary.append(f'确认帧：{len(last_attempt)}/7；有效TABLE帧：{sum(f["table"] is not None for f in last_attempt)}；采样次数：{len(evidence["attempts"])}')
            else:
                summary.append(f'有效关键时刻：{len(evidence["frames"])}；估计旋转覆盖：{evidence["turns"]:.2f}圈')
            summary += ['', '数量对比（相同采样与桌面/时序筛选）：']
            summary += comparison_lines(pure_counts, counts,extra_items=extra_items)
            summary.append(f'文字识别：{ocr_status_text(ocr_report)}；已处理 {ocr_report.get("processed_frames",0)} 帧')
            summary.append(f'对照结果文件：{target}')
            summary.append(f'比赛格式文件（加深度）：{competition_target}')
            summary.append(f'OCR证据目录：{archive/"ocr"}')
            summary_path = target.with_name(target.stem+'_summary.txt')
            summary_path.write_text('\n'.join(summary)+'\n', encoding='utf-8')
            (archive/'summary.txt').write_text('\n'.join(summary)+'\n', encoding='utf-8')
            detail = '筛选结果已保存' if reason == 'OK' else (
                '采样不足，已保存确认物品；覆盖不完整' if reason=='PARTIAL_SAMPLING' else
                'TABLE不足：本桌未计入结果' if reason == 'TABLE_INSUFFICIENT' else
                '采样帧数不足：本桌未计入结果' if reason=='SAMPLING_INSUFFICIENT' else
                '深度参考不足；本轮未完成深度过滤' if reason=='DEPTH_REFERENCE_UNAVAILABLE' else '覆盖未确认；输出保守结果')
            self.set_state(status='识别完成', detail=detail, final=final, elapsed=time.monotonic()-self.capture_start,
                           summary_path=str(summary_path), done=True)
            print(f'TIMED_COMPLETE mode={self.args.mode} seconds={elapsed:.3f} reason={reason} counts={dict(counts)}', flush=True)
        except Exception as error:
            if self.ocr_session is not None:
                if not self.cancel.is_set():
                    try:
                        ocr_report=self.ocr_session.finish()
                        metadata['ocr']=ocr_report
                        write_ocr_text(target.with_name(target.stem+'_ocr.txt'),ocr_report)
                    except Exception as ocr_error:
                        metadata['ocr_error']=str(ocr_error)
                self.ocr_session.abort()
            metadata.update(status='FAILED', error=str(error))
            atomic_json(latest_json, metadata)
            self.set_state(status='启动失败', detail=str(error)[:35], final=[], done=True)
            print(f'TIMED_ERROR {error}', flush=True)
        finally:
            if getattr(self,'depth_pipeline',None) is not None:self.depth_pipeline.close()
            self.depth_recorder.finish()

    def update(self):
        with self.lock:
            state = dict(self.state)
        elapsed = state['elapsed'] if state['done'] else (
            time.monotonic()-state['started'] if state['started'] else 0.)
        try:
            sequence, rows = read_worker_csv(self.args.result_csv)
            processed=None
            pipeline=getattr(self,'depth_pipeline',None)
            if not state['done'] and pipeline is not None:
                with pipeline.lock:processed=pipeline.latest
            sequence,rgb,depth,pair_status,processed=preview_pair(sequence,processed,time.monotonic())
            if rgb is None or depth is None:
                raise ValueError(pair_status)
            if (self.live_pair_cache is not None and self.live_pair_cache['sequence']==sequence
                    and self.live_pair_cache.get('processed',False)==(processed is not None)):
                frame,depth_frame,live,depth_detail,preview,preview_matrix=self.live_pair_cache['render_data']
            else:
                frame,depth_frame,live,depth_detail,preview,preview_matrix=self.render_pair(sequence,rgb,depth,rows,processed=processed)
                self.live_pair_cache={'sequence':sequence,'processed':processed is not None,'render_data':(frame,depth_frame,live,depth_detail,preview,preview_matrix)}
        except (OSError, ValueError, IndexError) as error:
            if self.live_pair_cache is not None:
                frame,depth_frame,_,_,preview,preview_matrix=self.live_pair_cache['render_data']
                live=[]
                depth_detail=f'预览暂缓，显示上一有效画面；原因：{error}'
            else:
                frame = Image.new('RGB', (640, 480), '#10283A')
                depth_frame=Image.new('RGB',(640,480),'black')
                preview=preview_matrix=None;live=[]
                depth_detail=f'正在等待有效RGB-D图像；原因：{error}'
        ocr_report,ocr_summary = self.ocr_session.preview() if self.ocr_session is not None else ({'results':[]},'文字识别：准备中')
        crop=None
        if self.args.mode=='round':
            crop=round_preview_crop(None)
        rendered = render_dashboard(frame, live, table=self.args.table, status=state['status'],
            elapsed=elapsed, model_version=self.args.model_version, output_path=str(self.args.output_path),
            summary_detections=state['final'], detail=state['detail'], ocr_summary=ocr_summary,
            extra_items=ocr_report['results'],depth_frame=depth_frame,depth_detail=depth_detail,
            rgb_preview=preview,preview_to_reference=preview_matrix,preview_crop=crop,
            round_table_outline=self.args.mode=='round',table_label=getattr(self.args,'table_label',None),
            square_table_polygon=getattr(self,'live_square_polygon',None) if self.args.mode=='square' else None)
        rendered.thumbnail((self.root.winfo_screenwidth(), self.root.winfo_screenheight()))
        self.photo = ImageTk.PhotoImage(rendered)
        self.label.configure(image=self.photo)
        if state['done'] and not self.popup_shown:
            self.popup_shown = True
            title = '方桌测量结果' if self.args.mode == 'square' else '圆桌测量结果'
            command = (['zenity', '--text-info', '--title='+title,
                        '--filename='+state['summary_path'], '--width=850', '--height=550',
                        '--font=WenQuanYi Zen Hei 13', '--ok-label=关闭结果']
                       if state.get('summary_path') else
                       ['zenity', '--error', '--title='+title, '--text='+state['detail']])
            try:
                self.result_popup = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                                     stderr=subprocess.DEVNULL)
            except OSError as error:
                print(f'RESULT_POPUP_ERROR {error}', flush=True)
        self.root.after(250 if state.get('started') and not state['done'] else 100, self.update)

    def render_pair(self,sequence,rgb,depth,rows,processed=None):
            import cv2
            frame=Image.fromarray(cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB))
            depth_frame=depth_to_image(depth)
            table, accepted, rejected = fusion.scope(rows, self.args.mode)
            square_polygon=None
            with self.lock:busy=not self.state['done']
            if busy and processed is not None:
                table=UiDetection(**processed['table']) if processed['table'] else None
                accepted=processed['objects'];info=processed['depth']
                square_polygon=processed.get('square_surface',{}).get('polygon')
            else:
                if self.args.mode=='square' and (not busy or getattr(self,'preview_live_depth_ready',False)):
                    geometry=getattr(self,'square_live_geometry',None)
                    if geometry is None:self.square_live_geometry=geometry=SquareTableGeometry()
                    surface=geometry.locate(rgb,depth,table,rows,sequence)
                    if surface is not None:
                        table=UiDetection(16,table.confidence,tuple(surface['xyxy']))
                        square_polygon=surface['polygon']
                        table,accepted,rejected=fusion.scope(rows,'square',table,square_polygon,surface['rim_contacts'])
                if self.round_geometry is not None:
                    surface=surface_from_arrays(rgb,depth,table,rows)
                    if (surface is None and table is not None and
                            self.round_live_geometry.unverified_expansion(table.xyxy)):
                        surface=surface_from_arrays(rgb,depth,table,rows,
                            reference_box=self.round_live_geometry.shape_reference)
                    tracked=self.round_live_geometry.update(table,sequence,surface)
                    if self.round_live_geometry.moved:
                        self.live_depth_validator.last_plane=None
                        self.live_depth_validator.last_plane_time=0.
                        self.live_depth_validator.pending_support=[]
                    if tracked is None:
                        table=None;accepted=[]
                        rejected=[asdict(d) for d in rows if d.class_id!=16]
                    else:
                        table,accepted,rejected=fusion.scope(rows,'round',tracked)
                if not busy or getattr(self,'preview_live_depth_ready',False):
                    info=self.live_depth_validator.annotate_arrays(sequence,table,accepted,accepted+rejected,self.args.mode,depth)
                else:info={'plane':None}
            if busy and processed is None and not getattr(self,'preview_live_depth_ready',False):
                # Preview depth must not compete with timed capture or its final processing.
                for row in accepted:row['depth_evidence']={};row['depth_state']='UNKNOWN'
                info={'plane':None}
            if self.ocr_session is not None:
                report=self.ocr_session.report()
                regions=confirmed_regions(report) if self.args.mode=='square' else []
                visible_regions=live_regions(rgb,regions)
                accepted,_=exclude_detections(accepted,visible_regions)
            live = [UiDetection(r['class_id'], r['confidence'], tuple(r['xyxy']),
                    'TRANSPARENT' if r['depth_evidence'].get('water_transparent_support') else r['depth_state']) for r in accepted]
            if table:
                live.append(table)
            plane=info['plane']
            plane_text=(f'桌面可靠：内点 {plane["fraction"]:.0%}，残差 {plane["p90_error_mm"]:.1f}mm'
                        if plane else '定时采样与深度筛选进行中' if busy else '桌面参考不可靠')
            counts=Counter(r['depth_state'] for r in accepted)
            valid_ratio=float(((depth>250)&(depth<5000)).mean())
            transparent_count=sum(r['depth_evidence'].get('water_transparent_support',False) for r in accepted)
            depth_detail=f'RGB/深度同包序号：{sequence}\n{plane_text}\n三维 {counts["PASS"]} / 平面 {counts["FLAT"]} / 透明支持 {transparent_count} / 不确定 {counts["UNKNOWN"]-transparent_count}\n有效深度 {valid_ratio:.0%}；色标 0.25–3.0m；黑色为无效深度'
            preview,preview_matrix,preview_status=read_ocr_source(sequence)
            self.live_square_polygon=square_polygon
            return frame,depth_frame,live,depth_detail,preview,preview_matrix

    def run(self):
        signal.signal(signal.SIGTERM, lambda *_: self.root.after(0,self.root.destroy))
        try:
            super().run()
        finally:
            self.cancel.set()
            if self.ocr_session is not None:self.ocr_session.abort()
            self.thread.join(timeout=1.)
            if self.result_popup is not None and self.result_popup.poll() is None:
                self.result_popup.terminate()


def statistics_median(values):
    import statistics
    return statistics.median(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('square', 'round'), required=True)
    parser.add_argument('--model-version', required=True)
    args = parser.parse_args()
    args.table = 1 if args.mode == 'square' else 3
    args.frame = Path('/dev/shm/astra_yolo_frame.jpg')
    args.result_csv = Path('/dev/shm/astra_yolo_result.txt')
    args.output_path = Path.home()/'Desktop/result'/('CSU-3DVISION-R1.txt' if args.mode == 'square' else 'CSU-3DVISION-R2.txt')
    args.fullscreen, args.status_file, args.demo = False, None, False
    TimedUi(args).run()


if __name__ == '__main__':
    main()
