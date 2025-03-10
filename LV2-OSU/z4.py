# Zadatak 2.4.4 Napišite program koji ce kreirati sliku koja sadrži ´ cetiri kvadrata crne odnosno ˇ bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
# zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili u odgovarajuci oblik koristite numpy funkcije ´ hstack i vstack.

import numpy as np
import matplotlib.pyplot as plt

crni = np.zeros((50, 50))
bijeli = np.ones((50, 50)) 

row1 = np.hstack((crni, bijeli))
row2 = np.hstack((bijeli, crni))

ploca = np.vstack((row1, row2))

plt.imshow(ploca, cmap="gray")
plt.show()

