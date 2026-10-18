"""Read input policy from the same fresh RGB-D packet as the detections."""
import struct
import time
from pathlib import Path
from national_competition.runtime.depth_validation import capture_active

COMPAT_RGB_FLAG=8
COMPAT_INPUT_POLICY='native-640-compat-v1'


def read_capture_policy(sequence):
    if not capture_active():return None
    try:
        path=Path('/dev/shm/national_depth')/f'slot_{sequence%32}.bin'
        with path.open('rb') as stream:header=stream.read(40)
        magic,version,seq,stamp,w,h,scale,flags=struct.unpack('<4sIQQIIfI',header)
        if (magic,version,seq,w,h)!=(b'NDP1',2,sequence,640,480):return None
        if not 0<=time.monotonic()-stamp/1e6<=1.2:return None
        compatible=bool(flags&COMPAT_RGB_FLAG)
        return {'sequence':seq,'compatibility_mode':compatible,
                'mode':'native_640_compat' if compatible else 'standard',
                'ocr_enabled':not compatible,'depth_pair_valid':flags&3==3,
                'reason':'HIGH_RES_ALIGNMENT_FAILED' if compatible else None}
    except (OSError,ValueError,struct.error):return None
