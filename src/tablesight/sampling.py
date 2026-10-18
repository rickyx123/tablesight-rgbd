"""生成图像级加权训练清单；验证集不进入训练、不做重复采样。"""
import argparse
from collections import Counter
import csv,json,re
from pathlib import Path


def weighted_manifest(rows):
    rows=list(rows);groups={};physical={};paths=set();ids=set();weighted=[];counts=Counter();images=Counter()
    for row in rows:
        path=row['path'].strip();split=row['split'].strip();group=row['scene_group'].strip();sid=row['sample_id'].strip()
        if not path or any(c in path for c in '\r\n'):raise ValueError('图片路径不能为空或含换行')
        if not sid or sid in ids:raise ValueError('sample_id 为空或重复')
        if path in paths:raise ValueError('输入表中路径重复；请通过 weight 表达采样次数')
        ids.add(sid);paths.add(path)
        if split not in ('train','val','test'):raise ValueError('split 只能是 train/val/test')
        if not group:raise ValueError('scene_group 不能为空')
        if group in groups and groups[group]!=split:raise ValueError('同一场景组跨集合')
        groups[group]=split
        digest=row['source_sha'].lower().strip()
        if not re.fullmatch(r'[0-9a-f]{64}',digest):raise ValueError('source_sha 必须是物理原图 SHA256')
        if digest in physical and physical[digest]!=split:raise ValueError('同一物理原图跨集合')
        physical[digest]=split
        try:weight=int(row['weight'])
        except (ValueError,TypeError):raise ValueError('weight 必须是正整数')
        if str(weight)!=str(row['weight']).strip() or weight<1:raise ValueError('weight 必须是正整数')
        if split!='train' and weight!=1:raise ValueError('验证和测试样本权重必须是 1')
        if split=='train':
            weighted.extend([path]*weight);label=row.get('priority','ordinary')
            counts[label]+=weight;images[label]+=1
    return weighted,{'unique_training_images':sum(images.values()),'weighted_training_entries':len(weighted),
        'groups':{name:{'unique_images':images[name],'weighted_entries':counts[name],
                        'entry_fraction':counts[name]/len(weighted) if weighted else 0.} for name in sorted(images)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='含来源 SHA 和场景组的 CSV')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check-files',action='store_true',help='检查路径相对 CSV 所在目录确实存在')
    args=parser.parse_args()
    with args.input.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    lines,report=weighted_manifest(rows)
    if args.check_files:
        for row in rows:
            if not (args.input.parent/row['path']).is_file():raise FileNotFoundError(row['path'])
    args.output.write_text(''.join(p+'\n' for p in lines),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
