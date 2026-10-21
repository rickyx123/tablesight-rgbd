"""Valid tabletop moments -> actual round_fuse() -> instance quantity."""
import math
from collections import Counter
from _snapshot import emit
from national_competition.runtime.timed_fusion import round_fuse, square_fuse, burst_vote


def row(instance, rotation):
    radius = .40 if instance == 0 else .65
    angle = (.10 if instance == 0 else 1.30) + rotation
    cx = 320 + 200*radius*math.cos(angle)
    cy = 240 + 150*radius*math.sin(angle)
    return dict(class_id=0, confidence=.90,
                xyxy=[cx-15, cy-8, cx+15, cy+8],
                radius=radius, angle=angle, depth_state='UNKNOWN')


def run():
    # A valid observation with zero items is different from missing TABLE.
    # There are 20 capture attempts, but only 12 valid tabletop moments.
    frames = []
    for i in range(12):
        rotation = .08*i
        objects = [row(0, rotation), row(1, rotation)] if i < 6 else []
        frames.append(dict(objects=objects, signed_rotation=rotation))

    selected, diagnostic = round_fuse(frames, use_depth=False)
    # Counterexample only: incorrectly treating 8 missing-table attempts as
    # valid empty moments increases the denominator and suppresses quantity.
    invalid_empty = [dict(objects=[], signed_rotation=0.) for _ in range(8)]
    wrong_selected, wrong_diag = round_fuse(frames+invalid_empty, use_depth=False)

    # One moment may contain two raw frames. Glasses have a one-frame exception;
    # tape still needs both frames before this moment reaches round_fuse().
    tape = dict(class_id=1, confidence=.90, xyxy=[400, 300, 430, 330], depth_state='UNKNOWN')
    burst = burst_vote([
        dict(sequence=1, objects=[row(0, 0.), tape]),
        dict(sequence=2, objects=[])])
    square_row = dict(class_id=3, confidence=.90,
                      xyxy=[250, 180, 300, 230], depth_state='UNKNOWN')
    square_frames = [dict(objects=[square_row] if i < 5 else []) for i in range(7)]
    square_selected, square_decisions = square_fuse(square_frames, use_depth=False)

    def report(detections, info):
        return dict(counts=dict(Counter(d.class_id for d in detections)),
                    valid_moments=info['support_reference_moments'],
                    id_min_moments=info['id_min_moments'],
                    multi_min_moments=info['multi_min_moments'])

    return dict(capture_attempts=20, missing_table_attempts=8,
                objects_present_in_valid_moments=6,
                correct_valid_only=report(selected, diagnostic),
                wrong_missing_as_empty=report(wrong_selected, wrong_diag),
                square=dict(input_frames=7, frames_with_object=5,
                            counts=dict(Counter(d.class_id for d in square_selected)),
                            accepted_tracks=sum(d['accepted'] for d in square_decisions)),
                one_burst=dict(input_sequences=burst['sequences'],
                               accepted_class_ids=[r['class_id'] for r in burst['objects']]))


if __name__ == '__main__':
    emit('temporal_count', run())

