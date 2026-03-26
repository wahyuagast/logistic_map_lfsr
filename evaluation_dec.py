import numpy as np
import cv2

# =========================
# 1. Load Image
# =========================
original = cv2.imread("images/cameraman.tif", 0)
decrypted = cv2.imread("images/decrypted_cameraman.tif", 0)

if original is None or decrypted is None:
    raise ValueError("Gambar tidak ditemukan!")

if original.shape != decrypted.shape:
    raise ValueError("Ukuran gambar tidak sama!")

# =========================
# 2. MSE
# =========================
def mse(image1, image2):
    return np.mean((image1.astype(np.float64) - image2.astype(np.float64)) ** 2)

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
    return np.sum(image1 != image2) / image1.size * 100

# =========================
# 5. UACI
# =========================
def uaci(image1, image2):
    return np.mean(np.abs(image1.astype(np.float64) - image2.astype(np.float64))) / 255 * 100

# =========================
# 6. Exact Match Check
# =========================
def exact_match(image1, image2):
    return np.array_equal(image1, image2)

# =========================
# 7. Hitung Metrik
# =========================
mse_val = mse(original, decrypted)
psnr_val = psnr(original, decrypted)
npcr_val = npcr(original, decrypted)
uaci_val = uaci(original, decrypted)
match = exact_match(original, decrypted)

# =========================
# 8. Output
# =========================
print("=== DECRYPTION EVALUATION ===")
print(f"MSE  : {mse_val:.4f}")
print(f"PSNR : {psnr_val:.4f} dB")
print(f"NPCR : {npcr_val:.4f} %")
print(f"UACI : {uaci_val:.4f} %")
print(f"Exact Match : {match}")