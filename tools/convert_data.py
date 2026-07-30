import os
import argparse
import pickle
import numpy as np
import cv2
from tqdm import tqdm

try:
    from .mxnet_compat import import_mxnet
except ImportError:
    from mxnet_compat import import_mxnet

mx = import_mxnet()


def load_mx_rec(rec_path, save_path, write_img=True):
    image_root = os.path.join(save_path, "images")
    if not os.path.isdir(image_root):
        os.makedirs(image_root)

    imgrec = mx.recordio.MXIndexedRecordIO(
        os.path.join(rec_path, 'train.idx'),
        os.path.join(rec_path, 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header,_ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    out_list = []
    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        filename = "{}/{}_{}.jpg".format(label, label, idx)
        out_list.append("{} {}\n".format(filename, label))
        file_path = os.path.join(image_root, str(label))
        if write_img:
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
            cv2.imwrite(os.path.join(image_root, filename), img)
    with open(os.path.join(save_path, "list.txt"), 'w') as f:
        f.writelines(out_list)


def load_bin(path, rootdir, image_size=[112,112]):
    image_root = os.path.join(rootdir, "images")
    if not os.path.isdir(image_root):
        os.makedirs(image_root)
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(image_root, "{}.jpg".format(i)), img)
    np.save('{}/issame_list.npy'.format(rootdir), np.array(issame_list))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rec_path",
                        help="mxnet record file path",
                        default='faces_emore', type=str)
    parser.add_argument("-o", "--output_path", type=str)
    args = parser.parse_args()

    load_mx_rec(args.rec_path, args.output_path, write_img=True)


if __name__ == "__main__":
    main()
