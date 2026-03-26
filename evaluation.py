import numpy as np
import cv2

# =========================
# 1. Load Image
# =========================
img = cv2.imread("images/cameraman.tif", 0)  # grayscale
encrypted_img = cv2.imread("images/cameraman_enc.tif", 0)

if img is None or encrypted_img is None:
    raise ValueError("Pastikan path gambar benar!")

if img.shape != encrypted_img.shape:
    raise ValueError("Ukuran gambar harus sama!")

# =========================
# 2. MSE
# =========================
def mse(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    return np.mean((image1 - image2) ** 2)

# =========================
# 3. PSNR
# =========================
def psnr(image1, image2):
    mse_val = mse(image1, image2)
    if mse_val == 0:
        return float('inf')
    return 10 * np.log10((255 ** 2) / mse_val)

# =========================
# 4. NPCR
# =========================
def npcr(image1, image2):
    diff = image1 != image2
    return np.sum(diff) / diff.size * 100

# =========================
# 5. UACI
# =========================
def uaci(image1, image2):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)
    return np.mean(np.abs(image1 - image2)) / 255 * 100

# =========================
# 6. Hitung semua metrik
# =========================
mse_val = mse(img, encrypted_img)
psnr_val = psnr(img, encrypted_img)
npcr_val = npcr(img, encrypted_img)
uaci_val = uaci(img, encrypted_img)

# =========================
# 7. Output
# =========================
print(f"MSE  : {mse_val:.4f}")
print(f"PSNR : {psnr_val:.4f} dB")
print(f"NPCR : {npcr_val:.4f} %")
print(f"UACI : {uaci_val:.4f} %")