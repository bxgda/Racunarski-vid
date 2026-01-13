import cv2
import numpy as np

def spoji_slike(bazna_slika, nova_slika):

    # inicijalizacija SIFT algoritma koji detektuje kljucne tacke i deskriptore
    sift = cv2.SIFT_create()

    # detectAndCompute vraca kljucne tacke i deskriptore (matematicki opisane kljucne tacke)
    kljucne_tacke_1, deskriptori_1 = sift.detectAndCompute(nova_slika, None)
    kljucne_tacke_2, deskriptori_2 = sift.detectAndCompute(bazna_slika, None)

    # uporedjujemo svaki deskriptor iz prve slike sa svim deskriptorima iz druge slike
    poklapanja = cv2.BFMatcher().knnMatch(deskriptori_1, deskriptori_2, k=2)

    # zadrzavamo samo sigurna poklapanja koristeci Lowe-ov test
    dobra_poklapanja = [m for m, n in poklapanja if m.distance < 0.75 * n.distance]

    # ako nemamo bar 4 tacke onda ne mozemo da izracunamo matricu transformacije
    if len(dobra_poklapanja) < 4:
        print("nedovoljno tacaka za spajanje!")
        return bazna_slika

    # izdvajamo koordinate uparenih tacaka iz obe slike
    src_tacke = np.float32([kljucne_tacke_1[m.queryIdx].pt for m in dobra_poklapanja]).reshape(-1, 1, 2)
    dst_tacke = np.float32([kljucne_tacke_2[m.trainIdx].pt for m in dobra_poklapanja]).reshape(-1, 1, 2)

    # racunamo matricu 3x3 koja transformise koordinate iz nove slike u koordinate bazne slike i pomocu RANCAC izbacujemo losa poklapanja
    matrica, _ = cv2.findHomography(src_tacke, dst_tacke, cv2.RANSAC, 5.0)

    # nalazimo dimenzije obe slike
    visina_nove_slike, sirina_nove_slike = nova_slika.shape[:2]
    visina_bazne_slike, sirina_bazne_slike = bazna_slika.shape[:2]

    # definisemo coskove nove slike
    coskovi_nove_slike = np.float32([[0, 0], [0, visina_nove_slike], [sirina_nove_slike, visina_nove_slike], [sirina_nove_slike, 0]]).reshape(-1, 1, 2)

    # ovde racunamo gde ce ti coskovi da se nadju nakon transformacije
    novi_coskovi = cv2.perspectiveTransform(coskovi_nove_slike, matrica)

    # definisemo coskove bazne slike, fiksni su na 0, 0
    coskovi_baze = np.float32([[0, 0], [0, visina_bazne_slike], [sirina_bazne_slike, visina_bazne_slike], [sirina_bazne_slike, 0]]).reshape(-1, 1, 2)

    # spajamo sve coskove u jednu listu da bi nasli ukupne dimenzije
    svi_coskovi = np.concatenate((novi_coskovi, coskovi_baze), axis=0)

    # ispitujemo da li je slika otisla u minus, a xmax/ymax su ukupne dimenzije
    [xmin, ymin] = np.int32(svi_coskovi.min(axis=0).ravel())
    [xmax, ymax] = np.int32(svi_coskovi.max(axis=0).ravel())

    # ako je pobegla slika onda je ispravljamo matricom transformacije
    h_trans = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]])

    # menjamo novoj slici oblik tako da se poklopi sa baznom slikom
    rezultat = cv2.warpPerspective(nova_slika, h_trans.dot(matrica), (xmax - xmin, ymax - ymin))

    # na novo platno (rezultat) stavljamo baznu sliku na odgovarajuce mesto u 0, 0 bez krivljenja
    rezultat[-ymin:-ymin + visina_bazne_slike, -xmin:-xmin + sirina_bazne_slike] = bazna_slika

    return rezultat


slike_imena = ['slika1.jpg', 'slika2.jpg', 'slika3.jpg']
slike = [cv2.imread(ime) for ime in slike_imena]

if any(img is None for img in slike):
    print("problem sa slikama")
else:
    # spajamo prvo srednju i desnu sliku
    srednja_desna = spoji_slike(slike[1], slike[2])

    # sada to spajamo sa levom slikom
    print("Dodajem levu sliku na rezultat...")
    finalna_panorama = spoji_slike(srednja_desna, slike[0])

    cv2.imwrite('panorama_rezultat.jpg', finalna_panorama)
    cv2.imshow('Finalna Panorama', finalna_panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
