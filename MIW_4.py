import os
import random
import numpy as np
import scipy.optimize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##Zadanie 1: Napisz program który automatycznie będzie wczytywał po koleji dane z plików daneXX.txt i wykonywał resztę zadań. Zaproponuj i zrealizuj podział tych danych na dane treningowe i dane testowe.

file = open("C:/Users/Артём/Python projects/Dane/dane2.txt","r")
X_y = np.loadtxt(file)
x = X_y[:,0]
y = X_y[:,1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

##Zadanie 2: Zaproponuj liniowy model parametryczny Model 1. Określ parametry modelu stosując metodę najmniejszych kwadratów dla danych treningowych,

a, b = np.polyfit(x_train, y_train, 1)

plt.scatter(x_train, y_train, label="Training data")
plt.scatter(x_test, y_test, label="Testing data")
plt.plot(x_train, a*x_train + b, label="Linear model")
plt.legend()
plt.show()

print("a = ", a)
print("b = ", b)

##Zadanie 3: Zweryfikuj poprawność Modelu 1

y_pred = a * x_test + b

mse_linear = np.mean((y_pred - y_test) ** 2)

print("Błąd średniokwadratowy: ", mse_linear)

##Zadanie 4: Zaproponuj bardziej złożony, minimum 3 stopnia, model regresji nieliniowej Model 2. Określ parametry modelu stosując metodę najmniejszych kwadratów dla danych treningowych

def quadratic_model(x, a, b, c):
    return a*x**2 + b*x + c

initial_guess = [1, 1, 1]

params, _ = scipy.optimize.curve_fit(quadratic_model, x_train, y_train, p0=initial_guess)

plt.scatter(x_train, y_train, label="Training data")
plt.scatter(x_test, y_test, label="Testing data")
plt.plot(x_train, quadratic_model(x_train, *params), label="Quadratic model")
plt.legend()
plt.show()

print("a = ", params[0])
print("b = ", params[1])
print("c = ", params[2])


##Zadanie 5: Zweryfikuj poprawność Model 2

y_pred = quadratic_model(x_test, *params)

mse_quadratic = np.mean((y_pred - y_test) ** 2)

print("Błąd średniokwadratowy: ", mse_quadratic)


##Zadanie 6: Porównaj oba modele

if mse_linear < mse_quadratic:
    print("Lepszy jest model liniowy")
else:
    print("Lepszy jest model kwadratowy")

