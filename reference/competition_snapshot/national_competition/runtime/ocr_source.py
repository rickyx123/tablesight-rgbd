"""Read the private high-resolution view bound to an existing RGB-D sequence."""
import struct,time
from pathlib import Path
import cv2,numpy as np
from national_competition.runtime.depth_validation import capture_active
from national_competition.runtime.capture_policy import COMPAT_RGB_FLAG

def read_ocr_source(sequence):
    if not capture_active():return None,None,'CAPTURE_INACTIVE'
    try:
        data=(Path('/dev/shm/national_depth')/f'slot_{sequence%32}.ocr.bin').read_bytes()
        magic,version,seq,stamp,w,h,scale,flags=struct.unpack('<4sIQQIIfI',data[:40])
        if (magic,version,seq)!=(b'NHR1',1,sequence):return None,None,'HIRES_SEQUENCE_MISMATCH'
        reference=(Path('/dev/shm/national_depth')/f'slot_{sequence%32}.bin').read_bytes()
        ref_magic,ref_version,ref_seq,ref_stamp,*_=struct.unpack('<4sIQQIIfI',reference[:40])
        if (ref_magic,ref_version,ref_seq,ref_stamp)!=(b'NDP1',2,sequence,stamp):
            return None,None,'HIRES_REFERENCE_MISMATCH'
        if _[-1]&COMPAT_RGB_FLAG:return None,None,'OCR_DISABLED_COMPATIBILITY'
        if not 0<=time.monotonic()-stamp/1e6<=1.2 or flags&3!=3:return None,None,'HIRES_PAIR_INVALID'
        if w>4096 or h>4096 or w<640 or h<480:return None,None,'HIRES_SIZE_INVALID'
        matrix=np.frombuffer(data[40:112],dtype='<f8').reshape(3,3).copy()
        length=struct.unpack('<I',data[112:116])[0]
        if len(data)!=116+length or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-9:
            return None,None,'HIRES_PACKET_INVALID'
        image=cv2.imdecode(np.frombuffer(data[116:],np.uint8),cv2.IMREAD_COLOR)
        if image is None or image.shape[:2]!=(h,w):return None,None,'HIRES_IMAGE_INVALID'
        return image,matrix,'HIGH_RES_SEQUENCE_MATCHED'
    except (OSError,ValueError,struct.error):return None,None,'HIRES_UNAVAILABLE'

def transform_rectangle(box,matrix):
    a,b,c,d=box
    points=cv2.perspectiveTransform(np.float32([[[a,b],[c,b],[c,d],[a,d]]]),matrix)[0]
    return [float(points[:,0].min()),float(points[:,1].min()),float(points[:,0].max()),float(points[:,1].max())]
