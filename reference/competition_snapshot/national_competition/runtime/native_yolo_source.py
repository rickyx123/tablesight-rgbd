"""Keep classification pixels independent of per-start RGB/depth registration."""
import cv2
import numpy as np
from national_competition.runtime.capture_policy import COMPAT_INPUT_POLICY

INPUT_POLICY='native-highres-one-resize-v1'


def compatibility_view(image):
    """Explicit native-640 mode, never a silent substitute for missing high RGB."""
    if image is None or image.shape[:2]!=(480,640):
        raise ValueError('Expected a sequence-matched native 640x480 RGB frame')
    return image,None,COMPAT_INPUT_POLICY


def canonical_native_view(image,source_to_reference):
    if image is None or image.shape[:2]!=(1536,2048):
        raise ValueError('Expected native 2048x1536 RGB')
    matrix=np.asarray(source_to_reference,dtype=np.float64)
    if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-9:
        raise ValueError('Invalid source-to-depth coordinate mapping')
    canonical=cv2.resize(image,(640,480),interpolation=cv2.INTER_AREA)
    canonical_to_reference=matrix@np.diag([2048/640,1536/480,1.])
    return canonical,canonical_to_reference


def project_detections(detections,matrix):
    result=[]
    for detection in detections:
        cls,confidence,box=detection[:3];a,b,c,d=box
        points=cv2.perspectiveTransform(np.float32([[[a,b],[c,b],[c,d],[a,d]]]),matrix)[0]
        if not np.isfinite(points).all():continue
        mapped=[float(np.clip(points[:,0].min(),0,639)),float(np.clip(points[:,1].min(),0,479)),
                float(np.clip(points[:,0].max(),0,639)),float(np.clip(points[:,1].max(),0,479))]
        if mapped[2]-mapped[0]<1 or mapped[3]-mapped[1]<1:continue
        result.append((cls,confidence,mapped)+detection[3:])
    return result
