import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Učitaj prethodno sačuvani model
model = keras.models.load_model('mnist_model.h5')

# Učitaj sliku sa diska
img_path = 'slika.png'  

# Učitaj sliku koristeći PIL (Python Imaging Library)
img = Image.open(img_path).convert('L')  # 'L' pretvara sliku u grayscale (jedan kanal)

# Prikazivanje originalne slike
plt.imshow(img, cmap='gray')
plt.title("Originalna Slika")
plt.axis('off')
plt.show()

# Preoblikovanje slike na dimenzije 28x28 (MNIST dimenzije)
img = img.resize((28, 28))

# Skaliranje slike na [0, 1] interval
img_array = np.array(img).astype("float32") / 255

# Dodajemo dimenziju za batch (1, 28, 28, 1)
img_array = np.expand_dims(img_array, axis=-1)

# "Flatten" sliku u vektor dimenzije (784,)
img_array = img_array.reshape((1, 784))

# Predikcija sa modelom
predictions = model.predict(img_array)

# Prikazivanje predviđene oznake
predicted_label = np.argmax(predictions)
print(f"Predviđena oznaka: {predicted_label}")
