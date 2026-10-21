"""Synthetic detections -> actual runtime scope() -> accepted/rejected rows."""
from _snapshot import emit
from national_competition.runtime.timed_fusion import scope
from national_competition.ui.competition_visualizer import UiDetection


def run():
    # Coordinates refer to the same 640x480 image, not model letterbox pixels.
    table = UiDetection(16, .95, (100, 80, 540, 400))
    on_table = UiDetection(3, .90, (250, 180, 300, 230))
    on_floor = UiDetection(3, .90, (560, 330, 610, 390))
    # Inside the rectangular TABLE box, outside its perspective rear corner.
    rear_corner = UiDetection(3, .90, (105, 80, 125, 120))
    rows = [table, on_table, on_floor, rear_corner]
    polygon = [(180, 100), (460, 100), (530, 390), (110, 390)]
    detected_table, accepted, rejected = scope(rows, 'square', table_polygon=polygon)

    # The circle's rectangular bounds are not equivalent to its ellipse.
    _, circle_yes, circle_no = scope(rows, 'round')
    # Even with a polygon, no TABLE candidate means no accepted table objects.
    missing, missing_yes, missing_no = scope(rows[1:], 'square', table_polygon=polygon)

    def boxes(items):
        return [dict(class_id=r['class_id'], xyxy=r['xyxy']) for r in items]

    return dict(
        square=dict(table_xyxy=detected_table.xyxy, table_polygon=polygon,
                    accepted=boxes(accepted), rejected=boxes(rejected)),
        round=dict(accepted=boxes(circle_yes), rejected=boxes(circle_no)),
        missing_table=dict(table=missing, accepted=boxes(missing_yes), rejected=boxes(missing_no)))


if __name__ == '__main__':
    emit('table_scope', run())

