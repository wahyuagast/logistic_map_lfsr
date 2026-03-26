import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# Load Image
# =========================
img = cv2.imread("images/cameraman.tif", 0)
enc = cv2.imread("images/cameraman_enc.tif", 0)

if img is None or enc is None:
    raise ValueError("Gambar tidak ditemukan!")

# =========================
# Fungsi ambil pasangan piksel
# =========================
def get_pairs(image, direction='horizontal', num_samples=5000):
    rows, cols = image.shape
    x = []
    y = []

    for _ in range(num_samples):
        i = np.random.randint(0, rows - 1)
        j = np.random.randint(0, cols - 1)

        if direction == 'horizontal':
            x.append(image[i, j])
            y.append(image[i, j+1])
        elif direction == 'vertical':
            x.append(image[i, j])
            y.append(image[i+1, j])
        elif direction == 'diagonal':
            x.append(image[i, j])
            y.append(image[i+1, j+1])

    return np.array(x), np.array(y)

# =========================
# Fungsi plot
# =========================
def plot_correlation(image, title):
    directions = ['horizontal', 'vertical', 'diagonal']

    plt.figure(figsize=(15, 4))

    for idx, d in enumerate(directions):
        x, y = get_pairs(image, d)

        plt.subplot(1, 3, idx+1)
        plt.scatter(x, y, s=1)
        plt.title(f"{title} ({d})")
        plt.xlabel("Pixel 1")
        plt.ylabel("Pixel 2")

    plt.tight_layout()
    plt.show()

# =========================
# Plot hasil
# =========================
plot_correlation(img, "Original Image")
plot_correlation(enc, "Encrypted Image")