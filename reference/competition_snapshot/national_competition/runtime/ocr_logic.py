"""Book OCR evidence: exact keywords, spatial deduplication, distinct-frame counts."""
from collections import Counter
import math
import re
import unicodedata


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def classify(text, score, settings, *, paper_label=False):
    if score < settings['minimum_text_confidence']:
        return None
    value = normalized(text)
    if any(normalized(word) in value for word in settings.get('negative_keywords', [])):
        return None
    matches = []
    for subject in settings['subjects']:
        words = [word for word in subject['keywords'] if len(normalized(word))>=2 and normalized(word) in value]
        if words:
            matches.append((subject, max(words, key=len)))
    # Do not invent a category from fuzzy corrections or competing subjects.
    partial=False
    if not matches and settings.get('literal_label_characters_enabled',False):
        candidates=[]
        for subject in settings['subjects']:
            if subject.get('goal_id') not in ('W001','W002','W003','W004'):continue
            keyword=''.join(c for c in normalized(subject['name']) if c in value)
            if keyword:candidates.append((subject,keyword))
        strongest=max((len(word) for _,word in candidates),default=0)
        matches=[entry for entry in candidates if len(entry[1])==strongest]
        partial=bool(matches)
    if not matches and paper_label and settings.get('partial_labels_enabled',False):
        # Only literal characters from one of the four supplied labels qualify.
        # A single character additionally needs a located white label and a
        # higher recognition score; generic package prose is never corrected.
        required=settings.get('single_character_min_confidence',.90) if len(value)==1 else settings.get('partial_label_min_confidence',.85)
        if 1<=len(value)<=2 and score>=required:
            for subject in settings['subjects']:
                name=normalized(subject['name'])
                if subject.get('goal_id') not in ('W001','W002','W003','W004'):continue
                remaining=iter(name)
                if all(any(c==letter for c in remaining) for letter in value):
                    matches.append((subject,value))
            partial=bool(matches)
    if len(matches) != 1:
        return None
    subject, keyword = matches[0]
    goal = subject.get('goal_id')
    if goal is not None and not re.fullmatch(r'W\d{3}', goal):
        raise ValueError('OCR goal_id must be null or a confirmed Wxxx code')
    result={'name': subject['name'], 'goal_id': goal, 'keyword': keyword,
            'text': text, 'confidence': float(score)}
    if partial:result.update(partial_label=True,matched_characters=len(keyword))
    return result


def bounds(polygon):
    xs, ys = zip(*polygon)
    return min(xs), min(ys), max(xs), max(ys)


def deduplicate(hits):
    kept = []
    for hit in sorted(hits, key=lambda h:h['confidence'], reverse=True):
        x1,y1,x2,y2 = bounds(hit['box'])
        cx,cy = (x1+x2)/2,(y1+y2)/2
        duplicate = False
        for prior in kept:
            if prior['name'] != hit['name']:
                continue
            a,b,c,d = bounds(prior['box'])
            distance = math.hypot(cx-(a+c)/2, cy-(b+d)/2)
            radius = max(8., .5*max(x2-x1,y2-y1,c-a,d-b))
            if distance <= radius:
                duplicate = True
                break
        if not duplicate:
            kept.append(hit)
    return kept


def aggregate_frames(frames, settings):
    by_sequence = {frame['sequence']: frame for frame in frames if frame.get('status')=='OK'
                   and frame.get('table_valid', True)}
    quantities = {sequence: Counter(hit['name'] for hit in deduplicate(frame.get('hits', [])))
                  for sequence,frame in by_sequence.items()}
    confirmed = []
    subjects=list(settings.get('subjects',[]));known_names={s['name'] for s in subjects}
    for frame in by_sequence.values():
        for hit in frame.get('hits',[]):
            if hit.get('generic_paper_text') and hit['name'] not in known_names:
                subjects.append({'name':hit['name'],'goal_id':None});known_names.add(hit['name'])
    for subject in subjects:
        name = subject['name']
        values = [quantity[name] for quantity in quantities.values()]
        count = max((n for n in range(1, max(values, default=0)+1)
                     if sum(v >= n for v in values) >= settings['minimum_distinct_frames']), default=0)
        if count:
            scores=[hit['confidence'] for frame in by_sequence.values() for hit in deduplicate(frame.get('hits',[]))
                    if hit['name']==name]
            confirmed.append({'name': name, 'goal_id': subject.get('goal_id'), 'count': count,
                              'frame_support': sum(v>=count for v in values),
                              'confidence':sum(scores)/len(scores) if scores else 0.})
    return confirmed


def summary_lines(report):
    lines = ['', '文字识别诊断：',
             f'状态：{report.get("status", "DISABLED")}；处理 {report.get("processed_frames",0)} 帧']
    if report.get('disabled_reason')=='HIGH_RES_ALIGNMENT_FAILED':
        lines.append('高清对齐失败：本次使用原生640兼容模式，OCR已关闭；物品检测继续。')
        return lines
    rows = report.get('results', [])
    if rows:
        lines += [f'{r["goal_id"] or "编码待公布"}  {r["name"]}  数量={r["count"]}  支持帧={r["frame_support"]}'
                  for r in rows]
    else:
        lines.append('暂无通过多帧确认的文字目标；原始文字和候选见OCR记录。')
    if report.get('errors'):
        lines += ['OCR诊断：'+str(error) for error in report['errors'][:3]]
    return lines


def status_text(report):
    if report.get('disabled_reason')=='HIGH_RES_ALIGNMENT_FAILED':
        return '兼容模式，OCR已关闭'
    return {'PROCESSING':'识别中','COMPLETE':'完成','INCOMPLETE':'尚未完成','TIME_LIMIT':'达到文字识别时限，保留已确认结果',
            'DIAGNOSTIC_ONLY':'桌面未确认，仅保存文字证据','NO_VALID_SAMPLES':'缺少图像',
            'ERROR':'失败','DISABLED':'未启用'}.get(report.get('status'),report.get('status','准备中'))
