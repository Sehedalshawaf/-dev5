import cv2
import numpy as np

# Görüntüyü yükleme
image = cv2.imread('pirinc_goruntusu.jpg')

# Görüntüyü gri tona dönüştürme
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Pirinçleri belirgin hale getirmek için eşikleme
_, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Morfolojik işlemler
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Etiketleme ve pirinç sayısını bulma
_, markers = cv2.connectedComponents(sure_bg)
count = markers.max() - 1  # Arka planı saymamak için 1 çıkartılır

# Sonuçları gösterme
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresh)
cv2.imshow('Processed Image', sure_bg)
print(f"Pirinç sayısı: {count}")

cv2.waitKey(0)
cv2.destroyAllWindows()
