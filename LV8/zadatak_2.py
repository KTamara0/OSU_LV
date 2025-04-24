'''
Zadatak 8.4.2 Napišite skriptu koja ce u ´ citati izgra ˇ denu mrežu iz zadatka 1 i MNIST skup ¯
podataka. Pomocu matplotlib biblioteke potrebno je prikazati nekoliko loše klasi ´ ficiranih slika iz
skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvidenu ¯
mrežom.
'''
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Učitaj prethodno sačuvani model
model = keras.models.load_model('mnist_model.h5')

# Učitaj MNIST podatke
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Skaliranje slika na [0, 1] raspon
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

# Preoblikuj slike u (784,)
x_test_flattened = x_test_s.reshape(-1, 784)

# Predikcija sa modelom
predictions = model.predict(x_test_flattened)

# Identifikuj loše klasificirane slike
incorrect_indices = np.where(np.argmax(predictions, axis=1) != y_test)[0]

# Prikazivanje nekoliko loše klasificiranih slika
plt.figure(figsize=(6, 6))

# Odaberi prvih 9 loših klasifikacija
for i, idx in enumerate(incorrect_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    true_label = y_test[idx]
    predicted_label = np.argmax(predictions[idx])
    plt.title(f"True: {true_label}, Pred: {predicted_label}")
    plt.axis('off')

plt.tight_layout()
plt.show()