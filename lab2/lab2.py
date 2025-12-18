import cv2
import numpy as np

slika = cv2.imread('coins.png')

# konvertujemo sliku u sivi kanal jer tresholding radi nad jednim kanalom
# i blago zamucujemo sliku da bi uklonili sum i siten detalje koji bi eventualno smetali
siva_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
siva_slika_blur = cv2.GaussianBlur(siva_slika, (5, 5), 2.5)

# kreiramo masku za svaki novcic ali inverzno da bi pozadina bila crna a novcici beli
# i koristimo TRESH_OTSU koji sam nalazi najbolji prag razdvajanja
_, maska_bez_zatvaranja = cv2.threshold(siva_slika_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
maska_sa_zatvaranjem = cv2.morphologyEx(maska_bez_zatvaranja, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

# nalazimo "seme" bakarnog novcica a to radimo tako sto prebacujemo u HSV prostor boja
# i uzimamo samo S (saturation) kanal kako bi videli koji novcic ima vise boje a koji manje
slika_hsv = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)
s_kanal = slika_hsv[:, :, 1]

# rucni threshold (prag 100) izdvaja samo delove slike koji imaju jaku boju
_, marker = cv2.threshold(s_kanal, 100, 255, cv2.THRESH_BINARY)

# bitwise AND osigurava da marker postoji samo tamo gde vec znamo da postoji neki novcic
# erozija smanjuje marker kako bi bili sigurni da ne dodiruje ivice ili druge objekte
marker = cv2.bitwise_and(marker, maska_sa_zatvaranjem)
marker_erozija = cv2.erode(marker, np.ones((9, 9), np.uint8))

# pravimo kopiju markera i pravi kernel koji sluzi za samo dilataciju i ceo proces morfoloske rekonstrukcije
# uvek prosirujemo marker za jedan sloj piksela i to radimo tako sto ne dopustamo da se prosiri van okvira maske
# ako se ne desi nista u sirenju piksela onda izlazimo iz petlje
finalna_maska = marker_erozija.copy()
kernel = np.ones((3, 3), np.uint8)

while True:
    prethodna = finalna_maska.copy()
    finalna_maska = cv2.dilate(finalna_maska, kernel)
    finalna_maska &= maska_sa_zatvaranjem

    # cv2.imshow('animacija rekonstrukcije', finalna_maska)
    # cv2.waitKey(30)

    if (finalna_maska == prethodna).all():
        break

# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow('Maska', finalna_maska)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imshow('zamucena siva slika', siva_slika_blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imshow('maska pre zatvaranja', maska_bez_zatvaranja)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imshow('maska nakon zatvaranja', maska_sa_zatvaranjem)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imshow('saturation kanal', s_kanal)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imshow('marker pre erozije', marker)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# cv2.imshow('marker posle erozije', marker_erozija)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# novcic = cv2.bitwise_and(slika, slika, mask=finalna_maska)
# cv2.imshow('isecen novcic', novcic)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
