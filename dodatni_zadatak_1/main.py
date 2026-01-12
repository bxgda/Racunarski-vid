import cv2
import numpy as np

def uklanjanje_objekta_sa_ivica(putanja_do_slike):

    # ucitavamo sliku u gray-scale
    slika = cv2.imread(putanja_do_slike, 0)

    if slika is None:
        print("nije pronadjena slika")
        return

    # prikaz originalne slike
    cv2.imshow('originalna slika', slika)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # radimo binarizaciju (pravimo binarnu masku) kako bi pikseli bili 0 ili 1
    _, maska = cv2.threshold(slika, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('binarna maska', maska)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # kreiranje markera na ivicama
    marker = np.zeros_like(maska)
    marker[0, :] = maska[0, :]       # gornja ivica
    marker[-1, :] = maska[-1, :]     # donja ivica
    marker[:, 0] = maska[:, 0]       # leva ivica
    marker[:, -1] = maska[:, -1]     # desna ivica

    # morfoloska rekonstrukcija za objekte koji su spojeni sa pikselima na ivici
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    while True:
        # dilatacija
        marker_dilatacije = cv2.dilate(marker, kernel)

        # bitwise_and sa maskom
        novi_marker = cv2.bitwise_and(marker_dilatacije, maska)

        # "animacija"
        cv2.imshow('objekti na ivici', novi_marker)
        cv2.waitKey(10)

        # provera kraja (kraj je kad se slika vise nije promenila)
        if np.array_equal(marker, novi_marker):
            break

        marker = novi_marker

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # oduzimanje (maska - marker)
    result = cv2.subtract(maska, marker)

    # resenje
    cv2.imshow('resenje', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

uklanjanje_objekta_sa_ivica('slika1.jpg')
uklanjanje_objekta_sa_ivica('slika2.jpg')
uklanjanje_objekta_sa_ivica('test_random_1.png')
uklanjanje_objekta_sa_ivica('test_random_2.png')

