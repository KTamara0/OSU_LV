import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a)Prikažite podatke za učenje u x1−x2 ravnini matplotlib biblioteke pri čemu podatke obojite s obzirom na klasu. 
#Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi marker (npr. ’x’). 
#Koristite funkciju scatter koja osim podataka prima i parametre c i cmap kojima je moguće definirati boju svake klase.
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker="x")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

#b)Izgradite model logističke regresije pomoću scikit-learn biblioteke na temelju skupa podataka za učenje.
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

#c)Pronađite u atributima izgrađenog modela parametre modela. 
#Prikažite granicu odluke naučenog modela u ravnini x1 −x2 zajedno s podacima za učenje. 
#Napomena: granica odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.
print("Coefficients: ", log_reg_model.coef_)
print("Intercept: ", log_reg_model.intercept_)

th0 = log_reg_model.intercept_[0]
th1= log_reg_model.coef_[0, 0]
th2 = log_reg_model.coef_[0, 1]

print(th0, th1, th2)

x1 = X_train[:, 0]
x2 = X_train[:, 1]
line = (-th0-(th1*x1)) / th2

print(line)
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
plt.plot(X_train[:, 0], line)
plt.show()


#d)Provedite klasifikaciju skupa podataka za testiranje pomoću izgrađenog modela logističke regresije. 
#Izračunajte i prikažite matricu zabune na testnim podacima. 
#Izračunate točnost, preciznost i odziv na skupu podataka za testiranje.
y_test_p = log_reg_model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()

print("Accuracy: ", accuracy_score(y_test, y_test_p))
print("Precision: ", precision_score(y_test, y_test_p))
print("Recall: ", recall_score(y_test, y_test_p))


#e)Prikažite skup za testiranje u ravnini x1−x2. Zelenom bojom označite dobro klasificirane primjere dok pogrešno klasificirane primjere označite crnom bojom.
for i in range(len(y_test)):
    if y_test[i] == y_test_p[i]:
        plt.scatter(X_test[i, 0], X_test[i, 1], c="g")
    else:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='k')
plt.show()