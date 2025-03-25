'''
Zadatak 4.5.1 Skripta zadatak_1.py ucitava podatkovni skup iz ˇ data_C02_emission.csv.
Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih veli ˇ cina. Detalje oko ovog podatkovnog skupa mogu se prona ˇ ci u 3. ´
laboratorijskoj vježbi.
a) Odaberite željene numericke veli ˇ cine speci ˇ ficiranjem liste s nazivima stupaca. Podijelite
podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%. ˇ
b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite ˇ
plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom. ˇ
c) Izvršite standardizaciju ulaznih velicina skupa za u ˇ cenje. Prikažite histogram vrijednosti ˇ
jedne ulazne velicine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja ˇ
transformirajte ulazne velicine skupa podataka za testiranje. ˇ
d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
povežite ih s izrazom 4.6.
e) Izvršite procjenu izlazne velicine na temelju ulaznih veli ˇ cina skupa za testiranje. Prikažite ˇ
pomocu dijagrama raspršenja odnos izme ´ du stvarnih vrijednosti izlazne veli ¯ cine i procjene ˇ
dobivene modelom.
f) Izvršite vrednovanje modela na nacin da izra ˇ cunate vrijednosti regresijskih metrika na ˇ
skupu podataka za testiranje.
g) Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ¯
ulaznih velicina?
'''


import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#a)
df = pd.read_csv("data_C02_emission.csv")

features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)',
            'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)']

target = 'CO2 Emissions (g/km)'

X=df[features]
y=df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#b)
plt.scatter(X_train['Engine Size (L)'], y_train, color='blue', label='Trening skup')
plt.scatter(X_test['Engine Size (L)'], y_test, color='red', label='Testni skup')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('Ovisnost CO2 emisije o veličini motora')
plt.legend()
plt.show()

#c)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

plt.hist(X_train['Engine Size (L)'], bins=20, alpha=0.5, label="Prije skaliranja")
plt.hist(X_train_scaled[:, features.index('Engine Size (L)')], bins=20, alpha=0.5, color="cyan", label="Nakon skaliranja")
plt.legend()
plt.title("Veličina motora prije i nakon skaliranja")
plt.show()

X_test_scaled= scaler.transform(X_test)

#d)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Koeficijenti: ", model.coef_)
print("Presjek (intercept): ", model.intercept_)

#e)
y_pred = model.predict(X_test_scaled)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Stvarne vrijedosti CO2 emisije")
plt.ylabel("Predviđene vrijednosti CO2 emisije")
plt.title("Usporedba stvarnih i predviđenih vrijednosti")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()

#f
mae= mean_absolute_error(y_test, y_pred)
mse= mean_squared_error(y_test, y_pred)
r2= r2_score(y_test, y_pred)

print("MAE: ", mae)
print("MSE: ", mse)
print("R2 Score: ", r2)

#g)
def evaluate_model(features_subset, label):
    X = df[features_subset]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nEvaluacija modela: {label}")
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("R2 Score:", round(r2, 4))

evaluate_model(['Engine Size (L)', 'Fuel Consumption Comb (L/100km)'],
               "2 varijable: Engine Size + Fuel Consumption Comb")
