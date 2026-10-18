"""Suppress two capacity labels emitted for the same physical water bottle."""

WATER_CLASSES = frozenset((10, 11))


def overlap_of_smaller(left, right):
    a, b = left[2], right[2]
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return intersection / max(1., min(area_a, area_b))


def suppress_water_capacity_duplicates(detections, containment=0.65):
    """Keep the higher-confidence label when 350/550 boxes describe one bottle."""
    kept = []
    for detection in sorted(detections, key=lambda item: item[1], reverse=True):
        duplicate = any(
            detection[0] in WATER_CLASSES and existing[0] in WATER_CLASSES
            and detection[0] != existing[0]
            and overlap_of_smaller(detection, existing) >= containment
            for existing in kept)
        if not duplicate:
            kept.append(detection)
    return kept
