from typing import Tuple
import numpy as np
import os
import torch
import random
import cv2


def load_data(pathfile: str) -> Tuple[np.ndarray, str]:
    """
    Loading the required data
    """
    with open(pathfile, 'r') as f:
        path = f.read()
    if not os.path.exists(path):
        raise FileNotFoundError
    _, ext = os.path.splitext(path)
    if not ext in ['.png', '.jpg']:
        raise NotImplementedError
    img = cv2.imread(path, 1)
    X = np.array(img)
    return X, path

def compress_image(
        X: np.ndarray, 
        filepath: str, 
        K: int = 255, 
        max_iter: int = 200
    ):
    """
    Compress image using K-Means Clustering Algorithm
    :param X: image array
    :param dirpath: path to save directory
    :param K: number of cluster to be formed
    :param max_iter: number of iterations to be performed
    """
    a, b, c = np.shape(X)
    X = np.reshape(X, [a * b, c], 2)
    m, n = np.shape(X)

    # Randomly initializing the centroids
    centroids = np.zeros((K, n), np.float32)
    for i in range(0, K):
        w = random.randint(0, m)
        centroids[i, :] = X[w, :]

    # Initializing the required PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    centroids = torch.tensor(centroids, dtype=torch.float32, requires_grad=True)

    # Forming clusters
    for i in range(max_iter):
        print('iteration', i + 1)

        # Reshaping tensors
        centroids_mat = centroids.view(1, K, n).expand(m, K, n)
        X_mat = X.view(m, 1, n).expand(m, K, n)

        # Computing distances
        distances = torch.sum((X_mat - centroids_mat) ** 2, dim=2)

        # Assigning points to clusters
        _, centroids_index = torch.min(distances, dim=1)

        # Updating centroids
        total_sum = torch.zeros_like(centroids)
        num_total = torch.zeros(K)
        for k in range(K):
            total_sum[k] = torch.sum(X[centroids_index == k], dim=0)
            num_total[k] = torch.sum(centroids_index == k)

        centroids = total_sum / num_total.view(-1, 1)

    # Converting centroids to numpy array and reshaping centroids_index
    centroids = centroids.detach().numpy().astype(np.uint8)
    centroids_index = centroids_index.detach().numpy().astype(np.uint8).reshape(a, b, 2)

    # Saving centroids and compressed image
    dirpath = os.path.dirname(filepath)
    filename, _ = os.path.splitext(os.path.basename(filepath))
    np.save(dirpath + f"\{filename}_codebook.npy", centroids)
    cv2.imwrite(dirpath + f"\{filename}_compressed.png", centroids_index)



def main():
    img, path = load_data('backend/path.txt')
    compress_image(img, path)


if __name__=="__main__":
    main() 