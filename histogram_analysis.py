import cv2
import matplotlib.pyplot as plt

# Load image
original = cv2.imread("images/cameraman.tif", 0)
encrypted = cv2.imread("images/cameraman_enc.tif", 0)

# Histogram
plt.figure(figsize=(10,5))

# Original
plt.subplot(1,2,1)
plt.title("Histogram Original")
plt.hist(original.flatten(), bins=256)

# Encrypted
plt.subplot(1,2,2)
plt.title("Histogram Encrypted")
plt.hist(encrypted.flatten(), bins=256)

plt.tight_layout()
plt.show()