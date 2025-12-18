import cv2
import numpy as np
import matplotlib.pyplot as plt

def morfoloska_rekonstrukcija(marker, maska):
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pret_marker = np.zeros_like(marker)
    tren_marker = marker.copy()
    while not np.array_equal(tren_marker, pret_marker):
        pret_marker = tren_marker.copy()
        tren_marker = cv2.dilate(tren_marker, kernel3)
        tren_marker = cv2.min(tren_marker, maska)
    return tren_marker

def vrati_masku(slika):
    slika_gray = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)

    hist, bins = np.histogram(slika_gray, bins=256, range=(0, 256))
    plt.bar(bins[:-1], hist)
    plt.xlabel("Intenzitet")
    plt.ylabel("Broj piksela")
    plt.show()

    _, coins_maska = cv2.threshold(slika_gray, 190, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    coins_maska2 = cv2.morphologyEx(coins_maska, cv2.MORPH_CLOSE, kernel) # spoji i popuni
    coins_maska2 = cv2.morphologyEx(coins_maska2, cv2.MORPH_OPEN, kernel) # ocisti

    return coins_maska2

def vrati_marker(slika):
    hsv_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_slika)

    plt.hist(s.ravel(), bins=256, color='orange')
    plt.title("Histogram zasićenosti (S kanal)")
    plt.xlabel("Vrednost zasićenosti")
    plt.ylabel("Broj piksela")
    plt.show()

    _, bakarni_marker = cv2.threshold(s, 60, 255, cv2.THRESH_BINARY)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bakarni_marker2 = cv2.morphologyEx(bakarni_marker, cv2.MORPH_OPEN, kernel2)

    return bakarni_marker2


# 1) ucitavanje slike:
slika_input = cv2.imread("coins.png")
slika_rgb = cv2.cvtColor(slika_input, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(slika_rgb)
plt.title("Pocetna slika:")
plt.show()
slika = cv2.medianBlur(slika_input, 9)

# 2) maska:
maska = vrati_masku(slika)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(maska, cmap="gray")
plt.title("Konacna maska:")
plt.show()

# 3) marker:
marker = vrati_marker(slika)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(marker, cmap="gray")
plt.title("Bakarni marker")
plt.show()

# 4) morfoloska rekonstrukcija:
bakarni_novcic_maska = morfoloska_rekonstrukcija(marker, maska)
konacna_maska = np.zeros_like(bakarni_novcic_maska)
konacna_maska[bakarni_novcic_maska > 0] = 255
plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.imshow(konacna_maska, cmap = "gray")
plt.title("Konacna maska - rezultat")
plt.show()

# kako izgleda novcic:
slika_rgb = cv2.cvtColor(slika_input, cv2.COLOR_BGR2RGB)
bakarni_samo = cv2.bitwise_and(slika_rgb, slika_rgb, mask=konacna_maska)
plt.figure(figsize=(10,5))
plt.imshow(bakarni_samo)
plt.title("Samo bakarni novčić")
plt.axis("off")
plt.show()

