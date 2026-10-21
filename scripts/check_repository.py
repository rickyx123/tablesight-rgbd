"""检查整理稿链接、证据哈希、参考快照和敏感信息模式。"""
import csv,hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
errors=[]
secret=re.compile(r'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
personal_path=re.compile(r'[A-Za-z]:[\\/](?:Users[\\/]|xzh[\\/]|RoboCup_|迅雷下载)')
files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '.venv' not in p.parts and '__pycache__' not in p.parts
       and not p.name.endswith('.pyc') and '.egg-info' not in '/'.join(p.parts)]
for p in files:
    if p.name in ('.draft-build','weighted_train.txt'):continue
    if p.suffix in ('.pt','.onnx','.om','.bin','.zip','.pem','.key'):errors.append(('unexpected_binary',str(p.relative_to(ROOT))))
    if p.suffix.lower() in ('.jpg','.jpeg','.png'):
        raw=p.read_bytes()
        valid=(raw.startswith(b'\xff\xd8\xff') and raw.endswith(b'\xff\xd9')) if p.suffix.lower() in ('.jpg','.jpeg') else raw.startswith(b'\x89PNG\r\n\x1a\n')
        if not valid:errors.append(('invalid_image',str(p.relative_to(ROOT))))
        continue
    try:text=p.read_text(encoding='utf-8-sig')
    except UnicodeError:errors.append(('non_utf8',str(p.relative_to(ROOT))));continue
    if secret.search(text):errors.append(('credential_pattern',str(p.relative_to(ROOT))))
    if personal_path.search(text):errors.append(('personal_path',str(p.relative_to(ROOT))))
    if p.suffix=='.md':
        for link in re.findall(r'\]\(([^)]+)\)',text):
            link=link.strip('<>').split('#')[0]
            if not link or re.match(r'^[a-zA-Z]+:',link):continue
            if not (p.parent/link).exists():errors.append(('broken_link',str(p.relative_to(ROOT)),link))
for item in json.loads((ROOT/'artifacts/model-comparisons/manifest.json').read_text(encoding='utf-8')):
    path=ROOT/item['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:errors.append(('metric_hash',item['file']))
for item in json.loads((ROOT/'reference/source_manifest.json').read_text(encoding='utf-8'))['files']:
    path=ROOT/'reference/competition_snapshot/national_competition'/item['path']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:errors.append(('source_hash',item['path']))
cases=json.loads((ROOT/'artifacts/casebook.json').read_text(encoding='utf-8'))
if len({c['case_id'] for c in cases})!=len(cases):errors.append(('duplicate_case_id',))
if (ROOT/'reference/competition_snapshot/national_competition/runtime/ocr_global.py').exists():errors.append(('cancelled_experiment_in_reference',))
for name in ['v7_v8_same_split.csv','v8_v9_same_split.csv']:
    with (ROOT/'artifacts/model-comparisons'/name).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        columns=[k for k in row if re.fullmatch(r'v\d+',k)]
        if len(columns)==2:
            actual=(float(row[columns[1]])-float(row[columns[0]]))*100
            expected=float(row['delta_percentage_points'])
            if abs(actual-expected)>1e-7:errors.append(('metric_delta',name,row.get('validation'),row.get('metric')))
print(json.dumps({'checked_files':len(files),'cases':len(cases),'errors':errors},ensure_ascii=False,indent=2))
raise SystemExit(bool(errors))
