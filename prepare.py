import os

import numpy as np
import cv2
from tqdm import tqdm

OUT_SIDE = 80
DUMP_X_FILE = 'dump_X.npy'
DUMP_Y_FILE = 'dump_Y.npy'
IMAGES_DIR = 'frames'
DIM = 512

def dirac_delta(i, n):
    ret = np.zeros((n,), dtype=np.float32)

    ret[i] = 1

    return ret

def main():
    filenames = os.listdir(IMAGES_DIR)

    Y = []
    for fn in tqdm(filenames):
        raw_image = cv2.imread(os.path.join(IMAGES_DIR, fn))
        resized_image = cv2.resize(raw_image, dsize=(OUT_SIDE, OUT_SIDE))
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        image = (gray_image > 128).astype(np.float32)
        Y.append(image)

    A = np.random.randn(DIM) + 1
    B = np.random.randn(DIM) - 1

    N = len(filenames)
    print('N:', N)
    X = [dirac_delta(i, N) for i in range(N)]

    np.save(DUMP_X_FILE, np.array(X))
    np.save(DUMP_Y_FILE, np.array(Y))

if __name__ == '__main__':
    main()