import os
import urllib.request
import numpy as np
import cv2
import onnxruntime as ort

MODEL_URLS = {
    "pnet.onnx": "https://github.com/kentaroy47/mtcnn-onnx/raw/master/mtcnn/models/pnet.onnx",
    "rnet.onnx": "https://github.com/kentaroy47/mtcnn-onnx/raw/master/mtcnn/models/rnet.onnx",
    "onet.onnx": "https://github.com/kentaroy47/mtcnn-onnx/raw/master/mtcnn/models/onet.onnx"
}

def download_models(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    for model_name, url in MODEL_URLS.items():
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            urllib.request.urlretrieve(url, model_path)

def nms(boxes, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if method is == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick

def pad(total_boxes, w, h):
    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.zeros((numbox,))
    dy = np.zeros((numbox,))
    edx = tmpw.copy() - 1
    edy = tmph.copy() - 1

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w - 1)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w - 1 + tmpw[tmp] - 1, 1)
    ex[tmp] = w - 1

    tmp = np.where(ey > h - 1)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h - 1 + tmph[tmp] - 1, 1)
    ey[tmp] = h - 1

    tmp = np.where(x < 0)
    dx.flat[tmp] = np.expand_dims(0 - x[tmp], 1)
    x[tmp] = 0

    tmp = np.where(y < 0)
    dy.flat[tmp] = np.expand_dims(0 - y[tmp], 1)
    y[tmp] = 0

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

def generateBoundingBox(imap, reg, scale, t):
    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))
    if reg.size == 0:
        reg = np.empty((0, 3))
    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg

def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox

def rerec(bboxA):
    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.vstack([l, l]))
    return bboxA


class MTCNNDetector:
    def __init__(self, model_dir="models/mtcnn", minsize=20, threshold=[0.6, 0.7, 0.8], factor=0.709):
        download_models(model_dir)
        self.pnet = ort.InferenceSession(os.path.join(model_dir, "pnet.onnx"))
        self.rnet = ort.InferenceSession(os.path.join(model_dir, "rnet.onnx"))
        self.onet = ort.InferenceSession(os.path.join(model_dir, "onet.onnx"))
        self.minsize = minsize
        self.threshold = threshold
        self.factor = factor

    def detect(self, img):
        factor_count = 0
        total_boxes = np.empty((0, 9))
        points = np.empty(0)
        h = img.shape[0]
        w = img.shape[1]
        minl = np.amin([h, w])
        m = 12.0 / self.minsize
        minl = minl * m
        scales = []
        while minl >= 12:
            scales += [m * np.power(self.factor, factor_count)]
            minl = minl * self.factor
            factor_count += 1

        for scale in scales:
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            im_data = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
            im_data = (im_data - 127.5) * (1. / 128.0)
            im_data = np.transpose(im_data, (2, 0, 1))
            im_data = np.expand_dims(im_data, 0).astype(np.float32)

            out = self.pnet.run(None, {self.pnet.get_inputs()[0].name: im_data})
            out0 = out[0][0]
            out1 = out[1][0]
            boxes, _ = generateBoundingBox(out1[1, :, :], out0, scale, self.threshold[0])

            if boxes.size > 0:
                pick = nms(boxes[:, 0:5], 0.5, 'Union')
                if len(pick) > 0:
                    boxes = boxes[pick, :]
                    total_boxes = np.append(total_boxes, boxes, axis=0)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            pick = nms(total_boxes[:, 0:5], 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = rerec(total_boxes.copy())
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

        numbox = total_boxes.shape[0]
        if numbox > 0:
            tempimg = np.zeros((24, 24, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1, :] = img[int(y[k]):int(ey[k]) + 1, int(x[k]):int(ex[k]) + 1, :]
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = cv2.resize(tmp, (24, 24), interpolation=cv2.INTER_AREA)
                else:
                    return np.empty((0, 5)), np.empty((0, 10))

            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 2, 0, 1)).astype(np.float32)

            out = self.rnet.run(None, {self.rnet.get_inputs()[0].name: tempimg1})
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out1[1, :]
            ipass = np.where(score > self.threshold[1])
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
            mv = out0[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                pick = nms(total_boxes, 0.7, 'Union')
                total_boxes = total_boxes[pick, :]
                total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
                total_boxes = rerec(total_boxes.copy())

        numbox = total_boxes.shape[0]
        if numbox > 0:
            total_boxes = np.fix(total_boxes).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
            tempimg = np.zeros((48, 48, 3, numbox))
            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[int(dy[k]):int(edy[k]) + 1, int(dx[k]):int(edx[k]) + 1, :] = img[int(y[k]):int(ey[k]) + 1, int(x[k]):int(ex[k]) + 1, :]
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tempimg[:, :, :, k] = cv2.resize(tmp, (48, 48), interpolation=cv2.INTER_AREA)
                else:
                    return np.empty((0, 5)), np.empty((0, 10))
            tempimg = (tempimg - 127.5) * 0.0078125
            tempimg1 = np.transpose(tempimg, (3, 2, 0, 1)).astype(np.float32)

            out = self.onet.run(None, {self.onet.get_inputs()[0].name: tempimg1})
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            out2 = np.transpose(out[2])
            score = out2[1, :]
            points = out1
            ipass = np.where(score > self.threshold[2])
            points = points[:, ipass[0]]
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
            mv = out0[:, ipass[0]]

            w = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h = total_boxes[:, 3] - total_boxes[:, 1] + 1
            points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
            points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1
            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
                pick = nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick, :]
                points = points[:, pick]

        points = np.transpose(points)
        if points.size > 0:
            # Reshape points from (N, 10) to (N, 5, 2)
            pts = np.zeros((points.shape[0], 5, 2))
            pts[:, :, 0] = points[:, 0:5]
            pts[:, :, 1] = points[:, 5:10]
            points = pts
        return total_boxes, points
