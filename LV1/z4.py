# Zadatak 1.4.4 Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ song.txt.
# Potrebno je napraviti rjecnik koji kao klju ˇ ceve koristi sve razli ˇ cite rije ˇ ci koje se pojavljuju u ˇ
# datoteci, dok su vrijednosti jednake broju puta koliko se svaka rijec (klju ˇ c) pojavljuje u datoteci. ˇ
# Koliko je rijeci koje se pojavljuju samo jednom u datoteci? Ispišite ih.

fhand = open("song.txt", "r")

rjecnik = {}

for line in fhand:
    line = line.strip()
    words = line.split()

    for word in words:
        word = word.lower().rstrip(",")
        if word in rjecnik:
            rjecnik[word] += 1
        else:
            rjecnik[word] = 1

fhand.close()

rijeci_s_jednim_pojavljivanjem = [rijec for rijec, broj in rjecnik.items() if broj == 1]

print(f"Rijeci koje se pojavljuju samo jednom: {rijeci_s_jednim_pojavljivanjem}")
print(f"Ukupno: {len(rijeci_s_jednim_pojavljivanjem)} rijeci koje se jednom pojavljuju.")