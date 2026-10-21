"""Synthetic millimetre arrays -> actual depth_validation.evaluate()."""
import json
from _snapshot import ROOT, emit
import numpy as np
from national_competition.runtime.depth_validation import evaluate


def run():
    rules = json.loads((ROOT/'reference/competition_snapshot/national_competition/depth/depth_rules.json').read_text(encoding='utf-8'))
    # Class 2 is the remote control; use the existing deployed class profile.
    profile = rules['profiles']['遥控器']
    row = dict(class_id=2, confidence=.90, xyxy=[60, 60, 120, 120])
    # Known inverse-depth plane Z=1400mm. Plane estimation is deliberately
    # excluded here so the example isolates object evidence classification.
    plane = dict(coeff=[0., 0., 1/1400], cx=100., cy=100., rx=100., ry=100.,
                 p90_error_mm=0.)
    flat = np.full((200, 200), 1400., dtype=np.float32)
    raised = flat.copy()
    raised[60:120, 60:120] = 1280.
    missing = flat.copy()
    missing[60:120, 60:120] = 0.
    result = {}
    for name, depth in [('raised', raised), ('flat', flat), ('missing', missing),
                        ('no_depth_array', None)]:
        evidence = evaluate(depth, plane, row, [], profile)
        result[name] = dict(state=evidence['state'], reject=evidence['reject'],
                            reason=evidence['reason'])
    return result


if __name__ == '__main__':
    emit('depth_states', run())

