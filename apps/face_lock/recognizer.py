import onnxruntime as ort
import numpy as np

class FaceRecognizer:
    def __init__(self, model_path):
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(model_path, sess_options)
        self.input_name = self.session.get_inputs()[0].name
        
    def get_embedding(self, img_tensor):
        """
        img_tensor: numpy array of shape (1, 3, 112, 112) normalized to [-1, 1]
        """
        out = self.session.run(None, {self.input_name: img_tensor})[0]
        # Normalize embedding to unit length
        out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out[0]

    def compute_similarity(self, emb1, emb2):
        """
        Cosine similarity
        """
        return np.dot(emb1, emb2)
