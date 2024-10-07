import numpy as np
import cv2
import os

def decompress_image(pathfile):
    with open(pathfile, 'r') as f:
        filepath = f.read()
    dirpath = os.path.dirname(filepath)
    filename, _ = os.path.splitext(os.path.basename(filepath))
    img = cv2.imread(dirpath + f"\{filename}_compressed.png", 0)
    centroids_index = np.array(img)
    a,b = np.shape(centroids_index)
    centroids_index = np.reshape(centroids_index, [a*b, 1], 2)
    centroids = np.load(dirpath+f"\{filename}_codebook.npy")
    m = a*b
    X_pix = np.zeros((m, 3))
    for i in range(0, m):
        X_pix[i, :] = centroids[centroids_index[i, 0], :]
    X_pix = np.reshape(X_pix,[a, b, 3], 3)
    cv2.imwrite(dirpath + f"\{filename}_decompressed.png", X_pix)

def main():
    decompress_image('backend/path.txt')

if __name__=='__main__':
    main()