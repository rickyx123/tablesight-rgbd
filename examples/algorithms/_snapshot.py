"""Resolve the preserved implementation without installing the hardware app."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / 'reference/competition_snapshot'
sys.path.insert(0, str(SNAPSHOT))


def emit(name, result):
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true',
                        help='Compare with the recorded output for this example')
    args = parser.parse_args()
    # Compare JSON structures, independent of tuple types and integer dict keys.
    result = json.loads(json.dumps(result, ensure_ascii=False))
    if args.check:
        expected = json.loads((Path(__file__).parent / 'expected' / (name+'.json')).read_text(encoding='utf-8'))
        if result != expected:
            raise SystemExit('Output differs from expected/'+name+'.json')
    print(json.dumps(result, ensure_ascii=False, indent=2))

