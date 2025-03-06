# Zadatak 1.4.5 Napišite Python skriptu koja ce u ´ citati tekstualnu datoteku naziva ˇ SMSSpamCollection.txt
# [1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke ozna ˇ cene kao ˇ spam, a neke kao ham.
# Primjer dijela datoteke:
# ham Yup next stop.
# ham Ok lar... Joking wif u oni...
# spam Did you hear about the new "Divorce Barbie"? It comes with all of Ken’s stuff!
# a) Izracunajte koliki je prosje ˇ can broj rije ˇ ci u SMS porukama koje su tipa ham, a koliko je ˇ
# prosjecan broj rije ˇ ci u porukama koje su tipa spam. ˇ
# b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?

ham_lst = []
spam_lst = []

with open("SMSSpamCollection.txt", encoding="utf8") as f:
    lines = [line.strip() for line in f.readlines()]

    for line in lines:
        if(line.startswith("ham")):
            ham_lst.append(line[4:].strip())
        elif (line.startswith("spam")):
            spam_lst.append(line[5:].strip())


ukupna_duljina = 0

for s in spam_lst:
    broj_rijeci = len(s.split())  
    ukupna_duljina += broj_rijeci  

avg_spam = ukupna_duljina / len(spam_lst) 


ukupna_duljina = 0

for h in ham_lst:
    broj_rijeci = len(h.split(" "))  
    ukupna_duljina += broj_rijeci 

avg_ham = ukupna_duljina / len(ham_lst)  

broj_spam_uskl = 0

for s in spam_lst:
    if s[-1] == '!':  
        broj_spam_uskl += 1  

spam_uskl = broj_spam_uskl  

print(f"Avg spam: {avg_spam}")
print(f"Avg ham: {avg_ham}")
print(f"Spam s uskličnikom: {broj_spam_uskl}")