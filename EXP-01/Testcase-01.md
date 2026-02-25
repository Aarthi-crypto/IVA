code :

import cv2 import numpy as np

def build_gaussian_pyramid(image, levels): pyramid = [image] for _ in range(levels - 1): image = cv2.pyrDown(image) pyramid.append(image) return pyramid

img = cv2.imread("grayscale.png", cv2.IMREAD_GRAYSCALE)

if img is None: print("Error: Grayscale image not found") else: pyramid = build_gaussian_pyramid(img, 3)

for i, level in enumerate(pyramid):
    cv2.imshow(f"Grayscale Pyramid - Level {i}", level)

cv2.waitKey(0)
cv2.destroyAllWindows()

output :

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/eef6787a-2791-46e0-87ab-442dac9cdd4f" />
