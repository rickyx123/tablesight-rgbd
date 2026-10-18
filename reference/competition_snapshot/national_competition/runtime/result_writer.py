#!/usr/bin/env python3
"""Write the exact rule-specified START/Goal_ID/END result format."""
import argparse
import json
import re
import unicodedata
from pathlib import Path

from national_competition.config import GOAL_IDS, NAMES


def comparison_lines(yolo_counts, depth_counts, extra_items=None):
    """Align both quantities by class, including classes removed by depth."""
    classes = sorted(set(yolo_counts) | set(depth_counts))
    pairs = [('纯YOLO+OCR', '加深度+OCR')] if extra_items is not None else [('纯YOLO', '加深度')]
    for cls in classes:
        label = f'{GOAL_IDS[cls]} {NAMES[cls]}'
        pairs.append((f'{label} 数量={yolo_counts.get(cls, 0)}',
                      f'{label} 数量={depth_counts.get(cls, 0)}'))
    for item in extra_items or []:
        label=f'{item.get("goal_id") or "待公布"} {item["name"]}'
        value=f'{label} 数量={item["count"]}'
        pairs.append((value,value))
    if not classes and not extra_items:
        pairs.append(('无通过筛选的目标', '无通过筛选的目标'))
    def width(value):
        return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in value)
    column_width = max(width(left) for left, _ in pairs)
    return [left + ' ' * (column_width - width(left)) + ' | ' + right
            for left, right in pairs]


def write_comparison(path, yolo_counts, depth_counts, table, status, extra_items=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'Table={table}', f'状态：{status}',
             '两列使用相同采样、置信度、桌面范围及多帧筛选；右列增加深度过滤。',
             '文字识别确认的物品并入两列，数量一致。',
             '', *comparison_lines(yolo_counts, depth_counts, extra_items)]
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    temporary.replace(path)
    return path


def write_result(path: Path, rows, unknown_rows=None):
    normalized = []
    for row in rows:
        class_id = int(row["class_id"])
        table = int(row["table"])
        count = int(row["count"])
        if class_id < 0 or class_id >= 16 or GOAL_IDS[class_id] is None:
            raise ValueError(f"Invalid reportable class_id: {class_id}")
        if table not in (1, 2, 3) or count < 1:
            raise ValueError(f"Invalid table/count: {row}")
        normalized.append((table, GOAL_IDS[class_id], count))
    for row in unknown_rows or []:
        goal = row.get('goal_id')
        table, count = int(row['table']), int(row['count'])
        if not isinstance(goal,str) or not re.fullmatch(r'W\d{3}',goal):
            raise ValueError('Unknown OCR results need a confirmed Wxxx code')
        if table not in (1,2,3) or count<1:
            raise ValueError(f'Invalid OCR table/count: {row}')
        normalized.append((table,goal,count))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["START"]
    lines.extend(f"Goal_ID={goal};Num={count};Table={table}"
                 for table, goal, count in sorted(normalized))
    lines.append("END")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="ascii")
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("--team", default="CSU")
    parser.add_argument("--team-name", default="3DVISION")
    parser.add_argument("--round", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Desktop/result")
    args = parser.parse_args()
    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    target = args.output_dir / f"{args.team}-{args.team_name}-R{args.round}.txt"
    print(write_result(target, rows))


if __name__ == "__main__":
    main()
