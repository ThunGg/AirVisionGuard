import warnings
import numpy as np
import cv2
from skimage import transform as trans

# Suppress the skimage SimilarityTransform deprecation warning
warnings.filterwarnings("ignore", message=".*estimate.*deprecated.*", category=FutureWarning)

# Standard landmarks for 112x112 face alignment
arcface_src = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

# Reuse a single transform instance to avoid per-call allocation overhead
_tform = trans.SimilarityTransform()

def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    _tform.estimate(lmk, arcface_src)
    M = _tform.params[0:2, :]
    return M

def align_face(img, bbox, landmark, image_size=112):
    """
    Align face based on landmarks.
    """
    M = estimate_norm(landmark, image_size)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped

def preprocess_face(img):
    """
    Preprocess image for ONNX face recognition model.
    Assuming the model expects (1, 3, 112, 112) RGB images, normalized to [-1, 1].
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = (img - 127.5) / 128.0
    return img.astype(np.float32)
