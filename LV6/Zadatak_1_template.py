import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

'''
Zadatak 6.5.1 Skripta zadatak_1.py ucitava ˇ Social_Network_Ads.csv skup podataka [2].
Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
Podaci o korisnicima su spol, dob i procijenjena placa. Razmatra se binarni klasi ´ fikacijski
problem gdje su dob i procijenjena placa ulazne veli ´ cine, dok je kupovina (0 ili 1) izlazna ˇ
velicina. Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija ˇ
plot_decision_region [1]. Podaci su podijeljeni na skup za ucenje i skup za testiranje modela ˇ
u omjeru 80%-20% te su standardizirani. Izgraden je model logisti ¯ cke regresije te je izra ˇ cunata ˇ
njegova tocnost na skupu podataka za u ˇ cenje i skupu podataka za testiranje. Potrebno je: ˇ

1. Izradite algoritam KNN na skupu podataka za ucenje (uz ˇ K=5). Izracunajte to ˇ cnost ˇ
klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje. Usporedite ˇ
dobivene rezultate s rezultatima logisticke regresije. Što primje ˇ cujete vezano uz dobivenu ´
granicu odluke KNN modela?
'''
# KNN model (K=5)
knn_model_5 = KNeighborsClassifier(n_neighbors=5)
knn_model_5.fit(X_train_n, y_train)

# Predikcija i evaluacija
y_train_knn5 = knn_model_5.predict(X_train_n)
y_test_knn5 = knn_model_5.predict(X_test_n)

print("KNN (K=5):")
print("Tocnost train: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn5)))
print("Tocnost test: " + "{:0.3f}".format(accuracy_score(y_test, y_test_knn5)))

# Granica odluke za K=5
plot_decision_regions(X_train_n, y_train, classifier=knn_model_5)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=5), Train Accuracy: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn5)))
plt.tight_layout()
plt.show()

#2. Kako izgleda granica odluke kada je K =1 i kada je K = 100?
# KNN model (K=1)
knn_model_1 = KNeighborsClassifier(n_neighbors=1)
knn_model_1.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=knn_model_1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=1)")
plt.tight_layout()
plt.show()

# KNN model (K=100)
knn_model_100 = KNeighborsClassifier(n_neighbors=100)
knn_model_100.fit(X_train_n, y_train)

plot_decision_regions(X_train_n, y_train, classifier=knn_model_100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN (K=100)")
plt.tight_layout()
plt.show()


'''
Zadatak 6.5.2 Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra ´ K
algoritma KNN za podatke iz Zadatka 1.
'''
# GridSearch za pronalazak najboljeg K
param_grid = {
    'n_neighbors': np.arange(1, 31)
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_n, y_train)

# Ispis rezultata
print("Najbolji K:", grid_search.best_params_['n_neighbors'])


'''
Zadatak 6.5.3 Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
ovih hiperparametara utjece na granicu odluke te pogrešku na skupu podataka za testiranje? ˇ
Mijenjajte tip kernela koji se koristi. Što primjecujete?
'''

# Funkcija za prikaz granice odluke
def plot_decision_boundary(X, y, classifier, resolution=0.02):
    plt.figure()
    # Postavljanje granica i mreže
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plotiranje podataka
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
    plt.title("SVM s RBF kernelom")
    plt.show()

# Promjena hiperparametara C i gamma
C_values = [0.1, 1, 10]
gamma_values = [0.1, 1, 10]

for C in C_values:
    for gamma in gamma_values:
        # Treniranje SVM modela s RBF kernelom
        svm_model = SVC(kernel='rbf', C=C, gamma=gamma)
        svm_model.fit(X_train_n, y_train)
        
        # Predikcija i točnost
        y_test_pred = svm_model.predict(X_test_n)
        print(f"Za C={C}, gamma={gamma} - Točnost na test skupu: {accuracy_score(y_test, y_test_pred):.3f}")
        
        # Prikaz granice odluke
        plot_decision_boundary(X_train_n, y_train, svm_model)

'''
Zadatak 6.5.4 Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra ´ C i γ
algoritma SVM za problem iz Zadatka 1.
'''

# Definiranje opsega vrijednosti za C i gamma
param_grid = {
    'C': [0.1, 1, 10, 100],  # Testiranje različitih vrijednosti C
    'gamma': [0.1, 1, 10, 'scale', 'auto']  # Testiranje različitih vrijednosti gamma
}

# Kreiranje SVM modela sa RBF kernelom
svm_model = SVC(kernel='rbf')

# GridSearchCV za pronalazak najboljih hiperparametara
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')

# Treniranje GridSearchCV modela
grid_search.fit(X_train_n, y_train)

# Ispis najboljih parametara i točnosti
print("Najbolji hiperparametri:", grid_search.best_params_)
print("Najbolja točnost (cross-validation):", "{:0.3f}".format(grid_search.best_score_))

# Evaluacija modela na test skupu s najboljim parametrima
best_svm_model = grid_search.best_estimator_
y_test_pred = best_svm_model.predict(X_test_n)
print("Tocnost na test skupu:", "{:0.3f}".format(accuracy_score(y_test, y_test_pred)))

