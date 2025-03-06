# Zadatak 1.4.3 Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji ˇ
# sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
# potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
# vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
# (npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovaraju ˇ cu poruku.

lst = []

while True:
    broj = input("Unesi broj ili Done: ")

    if broj.lower() == "done":  
        break  

    try:
        broj = float(broj)  
        lst.append(broj)
    except ValueError:  
        print("Error: Unos nije broj, pokušajte ponovo.")

if lst:
    print(f"Ukupan broj unosa: {len(lst)}")
    print(f"Srednja vrijednost: {sum(lst) / len(lst)}")
    print(f"Minimalna vrijednost: {min(lst)}")
    print(f"Maksimalna vrijednost: {max(lst)}")
    print(f"Sortirana lista: {sorted(lst)}")
else:
    print("Nema unesenih brojeva.")