"""按桌评估计数；所有输入行均参与，不丢弃失败预测。"""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path


def count_vector(rows):
    result={};seen=set()
    for row in rows:
        table=row['table'];count=row['count'];goal=row['goal_id']
        if isinstance(table,bool) or not isinstance(table,int) or table<1:raise ValueError('table 必须是正整数')
        if isinstance(count,bool) or not isinstance(count,int) or count<0:raise ValueError('count 必须是非负整数')
        if not isinstance(goal,str) or not goal.strip():raise ValueError('goal_id 不能为空')
        key=(table,goal)
        if key in seen:raise ValueError('同一桌和类别不能出现重复行')
        seen.add(key)
        if count:result[key]=count
    return result


def evaluate(records):
    scenes=table_total=scene_correct=table_correct=error=ground_total=0
    background_total=background_counted=0;background_observed=False
    repeats=defaultdict(list)
    for index,row in enumerate(records):
        gt=count_vector(row['ground_truth']);pred=count_vector(row.get('prediction',[]));scenes+=1
        tables=set(row.get('tables',[]))|{k[0] for k in gt}|{k[0] for k in pred}
        if not tables:raise ValueError('空桌也需要在 tables 中声明桌号')
        if any(isinstance(t,bool) or not isinstance(t,int) or t<1 for t in tables):raise ValueError('无效桌号')
        scene_correct+=gt==pred;ground_total+=sum(gt.values())
        for table in tables:
            expected={k:v for k,v in gt.items() if k[0]==table}
            actual={k:v for k,v in pred.items() if k[0]==table}
            table_total+=1;table_correct+=expected==actual
            error+=sum(abs(expected.get(k,0)-actual.get(k,0)) for k in expected.keys()|actual.keys())
        key=(str(row.get('scene_id',index)),str(row.get('pipeline','default')))
        repeats[key].append(tuple(sorted(pred.items())))
        if 'background_total' in row:
            n=row['background_total'];m=row.get('background_counted',0)
            if any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in (n,m)) or m>n:
                raise ValueError('背景指标按独立实物计数，必须 0 <= background_counted <= background_total')
            background_total+=n;background_counted+=m;background_observed=True
    pairs=matching=0
    for vectors in repeats.values():
        pairs+=math.comb(len(vectors),2)
        matching+=sum(math.comb(n,2) for n in Counter(vectors).values())
    return {'scene_count':scenes,'table_count':table_total,
            'scene_exact_match_rate':scene_correct/scenes if scenes else None,
            'table_exact_match_rate':table_correct/table_total if table_total else None,
            'mean_absolute_table_count_error':error/table_total if table_total else None,
            'relative_absolute_count_error':error/ground_total if ground_total else None,
            'repeated_scene_pair_count':pairs,'pairwise_repeat_consistency':matching/pairs if pairs else None,
            'background_false_inclusion_rate':background_counted/background_total if background_observed and background_total else None,
            'background_total':background_total if background_observed else None,
            'background_counted':background_counted if background_observed else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path,help='每行一条场景记录的 JSONL')
    args=parser.parse_args()
    with args.input.open(encoding='utf-8-sig') as stream:rows=[json.loads(line) for line in stream if line.strip()]
    print(json.dumps(evaluate(rows),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
