'''
Zadatak 4.5.2 Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku ˇ
varijable „Fuel Type“ kao ulaznu velicinu. Pri tome koristite 1-od-K kodiranje kategori ˇ ckih ˇ
velicina. Radi jednostavnosti nemojte skalirati ulazne veli ˇ cine. Komentirajte dobivene rezultate. ˇ
Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu
vozila radi?
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error

df = pd.read_csv("data_C02_emission.csv")

features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)',
            'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)', 'Fuel Type']
target = 'CO2 Emissions (g/km)'

X = pd.get_dummies(df[features], columns=['Fuel Type'])

y=df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
max_err = max_error(y_test, y_pred)

print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R2 score:", round(r2, 4))
print("Maksimalna pogreška u g/km:", round(max_err, 2))

errors = np.abs(y_test - y_pred)
max_index = errors.idxmax()

print("\nVozilo s najvećom pogreškom u predikciji CO2:")
print(df.loc[max_index][['Make', 'Model', 'Fuel Type', 'CO2 Emissions (g/km)']])
print("Predviđena emisija:", round(y_pred[list(y_test.index).index(max_index)], 2))