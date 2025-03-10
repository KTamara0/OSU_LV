# Zadatak 2.4.2 Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja ˇ data pri cemu je u ˇ
# prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci´ stupac polja je masa u kg.
# a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja? ˇ
# b) Prikažite odnos visine i mase osobe pomocu naredbe ´ matplotlib.pyplot.scatter.
# c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
# d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom ˇ podatkovnom skupu.
# e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
# ind = (data[:,0] == 1)

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data.csv", delimiter=",", skiprows=1)

print(f"Broj osoba: {data.shape[0]}")

plt.scatter(data[:, 1], data[:, 2], color="purple", alpha=0.5)
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase")
plt.show()

plt.scatter(data[::50, 1], data[::50, 2], color="lightblue", alpha=0.5)
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase-svaka 50. osoba")
plt.show()

min_height = np.min(data[:, 1])
max_height = np.max(data[:, 1])
arg_h = np.mean(data[:, 1])

print("Min visina: ", min_height)
print("Max visina: ", max_height)
print("Srednja: ", arg_h)

muskarci = data[data[:, 0] == 1]  
zene = data[data[:, 0] == 0] 

print(f"Muski min: {np.min(muskarci[:, 1])}, max: {np.max(muskarci[:, 1])}, srednja: {np.mean(muskarci[:, 1])}")
print(f"Zene min: {np.min(zene[:, 1])}, max: {np.max(zene[:, 1])}, srednja: {np.mean(zene[:, 1])}")
