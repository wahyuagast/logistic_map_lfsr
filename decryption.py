import numpy as np
import cv2
from logistic_map import logistic_map
from lfsr import lfsr

# =========================
# 1. Load encrypted image
# =========================
img = cv2.imread("images/cameraman_enc.tif", 0)

if img is None:
    raise ValueError("Encrypted image not found!")

rows, cols = img.shape

# =========================
# 2. Manual Flatten
# =========================
flat = []

for i in range(rows):
    for j in range(cols):
        flat.append(int(img[i][j]))

size = len(flat)

# =========================
# 3. Regenerate SAME chaos + LFSR
# =========================
chaos = logistic_map(0.5, 3.99, size)

indices = np.argsort(chaos)

key_stream = lfsr(0b10101010, [7, 5, 4, 3], size)
key_stream = [int(k * 255) for k in key_stream]

# =========================
# 4. Reverse Diffusion (with feedback)
# =========================
shuffled = []

prev = 0

for i in range(size):
    chaos_val = int(chaos[i] * 255)
    
    val = flat[i] ^ key_stream[i] ^ chaos_val ^ prev
    
    shuffled.append(val)
    
    prev = flat[i]   # IMPORTANT: use encrypted value

# =========================
# 5. Reverse Permutation
# =========================
unshuffled = [0] * size

for i in range(size):
    original_index = int(indices[i])
    unshuffled[original_index] = shuffled[i]

# =========================
# 6. Manual Reshape
# =========================
decrypted_img = []

k = 0
for i in range(rows):
    row = []
    for j in range(cols):
        row.append(unshuffled[k])
        k += 1
    decrypted_img.append(row)

# =========================
# 7. Convert to NumPy
# =========================
decrypted_img = np.array(decrypted_img, dtype=np.uint8)

# =========================
# 8. Save result
# =========================
cv2.imwrite("images/decrypted_cameraman.tif", decrypted_img)
cv2.imwrite("images/decrypted_cameraman.png", decrypted_img)

# =========================
# 9. Debug
# =========================
print("Decryption done!")
print("Shape:", decrypted_img.shape)
print("Min-Max:", decrypted_img.min(), "-", decrypted_img.max())
