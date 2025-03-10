# Zadatak 2.4.3 Skripta zadatak_3.py ucitava sliku ’ ˇ road.jpg’. Manipulacijom odgovarajuce´ numpy matrice pokušajte:
# a) posvijetliti sliku,
# b) prikazati samo drugu cetvrtinu slike po širini, ˇ
# c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
# d) zrcaliti sliku.

import numpy as np
import matplotlib.pyplot as plt


img = plt.imread("road.jpg")
plt.imshow(img)
plt.title("Originalna slika")
plt.show()

svijetla = np.clip(img*1.7, 0, 255).astype(np.uint8)
plt.imshow(svijetla)
plt.title("Posvijetljena")
plt.show()

h, w = img.shape[:2]
cetvrtina= img[:, w//4:w//2] 

plt.imshow(cetvrtina)
plt.title("Druga četvrtina")
plt.show()

rotirana = np.rot90(img, k=-1)

plt.imshow(rotirana)
plt.title("Rotirana")
plt.show()

mirrored = img[:, ::-1]
plt.imshow(mirrored)
plt.title("Zrcalna")
plt.show()

