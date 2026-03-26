import cv2
import numpy as np

# Load image
original = cv2.imread("images/cameraman.tif", 0)
encrypted = cv2.imread("images/cameraman_enc.tif", 0)

def calculate_entropy(image):
    hist = np.bincount(image.flatten(), minlength=256)
    prob = hist / np.sum(hist)

    # hindari log(0)
    prob = prob[prob > 0]

    entropy = -np.sum(prob * np.log2(prob))
    return entropy

# Hitung entropy
H_original = calculate_entropy(original)
H_encrypted = calculate_entropy(encrypted)

print("Entropy Original:", H_original)
print("Entropy Encrypted:", H_encrypted)