import cv2
import numpy as np
import random

# Load image
original = cv2.imread("images/cameraman.tif", 0)
encrypted = cv2.imread("images/cameraman_enc.tif", 0)

def correlation(img, num_samples=5000):
    rows, cols = img.shape

    x = []
    y = []

    for _ in range(num_samples):
        i = random.randint(0, rows-1)
        j = random.randint(0, cols-2)

        x.append(img[i, j])
        y.append(img[i, j+1])  # horizontal neighbor

    x = np.array(x)
    y = np.array(y)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov = np.mean((x - mean_x)*(y - mean_y))
    std_x = np.std(x)
    std_y = np.std(y)

    return cov / (std_x * std_y)

# Hitung korelasi
corr_orig = correlation(original)
corr_enc = correlation(encrypted)

print("Correlation Original:", corr_orig)
print("Correlation Encrypted:", corr_enc)