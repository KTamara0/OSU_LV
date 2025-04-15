import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1
unique_colors = np.unique(img_array, axis=0)
print(f"Broj različitih boja u slici: {len(unique_colors)}")

#2
K = 5 
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
kmeans.fit(img_array)

#3
# Zamijeni svaki piksel njegovim najbližim centroidom
labels = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[labels]

# Rekonstrukcija slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
axes[0].imshow(img)
axes[0].set_title("Originalna slika")
axes[0].axis('off')  

axes[1].imshow(img_aprox)
axes[1].set_title(f'Kvantizirana slika (K={K})')
axes[1].axis('off')  

plt.tight_layout()
plt.show()

#4
K_values = [2, 5, 10, 20]

fig, axes = plt.subplots(1, len(K_values), figsize=(18, 6))  
fig.suptitle('Kvantizirane slike za različite K vrijednosti', fontsize=16)

for i, K in enumerate(K_values):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(img_array)
    labels = kmeans.predict(img_array)
    img_array_aprox = kmeans.cluster_centers_[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    axes[i].imshow(img_aprox)
    axes[i].set_title(f'K={K}')
    axes[i].axis('off')  

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()

#5
# ucitaj sliku
img = Image.imread("imgs\\test_2.jpg")

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1
unique_colors = np.unique(img_array, axis=0)
print(f"Broj različitih boja u slici: {len(unique_colors)}")

#2
K = 5 
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
kmeans.fit(img_array)

#3
# Zamijeni svaki piksel njegovim najbližim centroidom
labels = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[labels]

# Rekonstrukcija slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
axes[0].imshow(img)
axes[0].set_title("Originalna slika")
axes[0].axis('off')  

axes[1].imshow(img_aprox)
axes[1].set_title(f'Kvantizirana slika (K={K})')
axes[1].axis('off')  

plt.tight_layout()
plt.show()

#4
K_values = [2, 5, 10, 20]

fig, axes = plt.subplots(1, len(K_values), figsize=(18, 6))  
fig.suptitle('Kvantizirane slike za različite K vrijednosti', fontsize=16)

for i, K in enumerate(K_values):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(img_array)
    labels = kmeans.predict(img_array)
    img_array_aprox = kmeans.cluster_centers_[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    axes[i].imshow(img_aprox)
    axes[i].set_title(f'K={K}')
    axes[i].axis('off')  

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()


# ucitaj sliku
img = Image.imread("imgs\\test_3.jpg")

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1
unique_colors = np.unique(img_array, axis=0)
print(f"Broj različitih boja u slici: {len(unique_colors)}")

#2
K = 5 
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
kmeans.fit(img_array)

#3
# Zamijeni svaki piksel njegovim najbližim centroidom
labels = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[labels]

# Rekonstrukcija slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
axes[0].imshow(img)
axes[0].set_title("Originalna slika")
axes[0].axis('off')  

axes[1].imshow(img_aprox)
axes[1].set_title(f'Kvantizirana slika (K={K})')
axes[1].axis('off')  

plt.tight_layout()
plt.show()

#4
K_values = [2, 5, 10, 20]

fig, axes = plt.subplots(1, len(K_values), figsize=(18, 6))  
fig.suptitle('Kvantizirane slike za različite K vrijednosti', fontsize=16)

for i, K in enumerate(K_values):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(img_array)
    labels = kmeans.predict(img_array)
    img_array_aprox = kmeans.cluster_centers_[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    axes[i].imshow(img_aprox)
    axes[i].set_title(f'K={K}')
    axes[i].axis('off')  

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()


# ucitaj sliku
img = Image.imread("imgs\\test_4.jpg")

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1
unique_colors = np.unique(img_array, axis=0)
print(f"Broj različitih boja u slici: {len(unique_colors)}")

#2
K = 5 
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
kmeans.fit(img_array)

#3
# Zamijeni svaki piksel njegovim najbližim centroidom
labels = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[labels]

# Rekonstrukcija slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
axes[0].imshow(img)
axes[0].set_title("Originalna slika")
axes[0].axis('off')  

axes[1].imshow(img_aprox)
axes[1].set_title(f'Kvantizirana slika (K={K})')
axes[1].axis('off')  

plt.tight_layout()
plt.show()

#4
K_values = [2, 5, 10, 20]

fig, axes = plt.subplots(1, len(K_values), figsize=(18, 6))  
fig.suptitle('Kvantizirane slike za različite K vrijednosti', fontsize=16)

for i, K in enumerate(K_values):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(img_array)
    labels = kmeans.predict(img_array)
    img_array_aprox = kmeans.cluster_centers_[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    axes[i].imshow(img_aprox)
    axes[i].set_title(f'K={K}')
    axes[i].axis('off')  

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()


# ucitaj sliku
img = Image.imread("imgs\\test_5.jpg")

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1
unique_colors = np.unique(img_array, axis=0)
print(f"Broj različitih boja u slici: {len(unique_colors)}")

#2
K = 5 
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
kmeans.fit(img_array)

#3
# Zamijeni svaki piksel njegovim najbližim centroidom
labels = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[labels]

# Rekonstrukcija slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
axes[0].imshow(img)
axes[0].set_title("Originalna slika")
axes[0].axis('off')  

axes[1].imshow(img_aprox)
axes[1].set_title(f'Kvantizirana slika (K={K})')
axes[1].axis('off')  

plt.tight_layout()
plt.show()

#4
K_values = [2, 5, 10, 20]

fig, axes = plt.subplots(1, len(K_values), figsize=(18, 6))  
fig.suptitle('Kvantizirane slike za različite K vrijednosti', fontsize=16)

for i, K in enumerate(K_values):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(img_array)
    labels = kmeans.predict(img_array)
    img_array_aprox = kmeans.cluster_centers_[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    axes[i].imshow(img_aprox)
    axes[i].set_title(f'K={K}')
    axes[i].axis('off')  

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()


# ucitaj sliku
img = Image.imread("imgs\\test_6.jpg")

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1
unique_colors = np.unique(img_array, axis=0)
print(f"Broj različitih boja u slici: {len(unique_colors)}")

#2
K = 5 
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
kmeans.fit(img_array)

#3
# Zamijeni svaki piksel njegovim najbližim centroidom
labels = kmeans.predict(img_array)
img_array_aprox = kmeans.cluster_centers_[labels]

# Rekonstrukcija slike
img_aprox = np.reshape(img_array_aprox, (w, h, d))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  
axes[0].imshow(img)
axes[0].set_title("Originalna slika")
axes[0].axis('off')  

axes[1].imshow(img_aprox)
axes[1].set_title(f'Kvantizirana slika (K={K})')
axes[1].axis('off')  

plt.tight_layout()
plt.show()

#4
K_values = [2, 5, 10, 20]

fig, axes = plt.subplots(1, len(K_values), figsize=(18, 6))  
fig.suptitle('Kvantizirane slike za različite K vrijednosti', fontsize=16)

for i, K in enumerate(K_values):
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(img_array)
    labels = kmeans.predict(img_array)
    img_array_aprox = kmeans.cluster_centers_[labels]
    img_aprox = np.reshape(img_array_aprox, (w, h, d))

    axes[i].imshow(img_aprox)
    axes[i].set_title(f'K={K}')
    axes[i].axis('off')  

plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()


#6
inertias = []
K_values = range(1, 21)
for K in K_values:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(img_array)
    inertias.append(kmeans.inertia_)

plt.figure()
plt.plot(K_values, inertias, marker='o')
plt.xlabel('Broj grupa (K)')
plt.ylabel('Inertia (J)')
plt.title('Elbow metoda za određivanje optimalnog K')
plt.grid(True)
plt.show()

#7
img = Image.imread("imgs\\test_1.jpg")
# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()
K = 5
kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
labels = kmeans.fit_predict(img_array)

# Kreiraj binarne slike za svaku grupu
fig, axes = plt.subplots(1, K, figsize=(18, 6))  # 1 red, K kolona
fig.suptitle(f'Binarne slike za svaku grupu (K={K})', fontsize=16)

for i in range(K):
    # Maskiraj samo piksele koji pripadaju grupi
    mask = (labels == i)
    
    # Stvori praznu sliku
    binary_image = np.zeros((w * h, 3))  # 3 boje (RGB)
    
    # Postavi piksele koji pripadaju grupi na bijelo (1.0)
    binary_image[mask] = 1.0

    # Pretvori natrag u oblik originalne slike
    binary_image_reshaped = np.reshape(binary_image, (w, h, 3))

    # Prikaz binarne slike za grupu
    axes[i].imshow(binary_image_reshaped)
    axes[i].set_title(f'Grupa {i + 1}')
    axes[i].axis('off')  # Skriva osi

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Prilagođava raspored
plt.show()