import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

# Zad 1

url = "https://api.zonda.exchange/rest/trading/candle/history/BTC-PLN/900?from=1540210129000&to=1543410329000"
headers = {'content-type': 'application/json'}

response = requests.get(url, headers=headers)

o = []
c = []
h = []
l = []
v = []

for j in response.json()['items']:
    print('-----------------------------')
    print('Kurs otwarcia: {}'.format(j[1]['o']))
    print('Kurs zamkniecia: {}'.format(j[1]['c']))
    print('Najwyższa wartość kursu: {}'.format(j[1]['h']))
    print('Najniższa wartość kursu: {}'.format(j[1]['l']))
    print('Wygenerowany wolumen: {}'.format(j[1]['v']))
    print(j)


    o.append(float(j[1]['o']))
    c.append(float(j[1]['c']))
    h.append(float(j[1]['h']))
    l.append(float(j[1]['l']))
    v.append(float(j[1]['v']))

X = list(zip(o, c, h, l))
y = v

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(len(response.json()['items']))
plt.plot(o)
plt.title('Kurs otwarcia')
plt.show()

plt.plot(c)
plt.title('Kurs zamknięcia')
plt.show()

plt.plot(h)
plt.title('Najwyższa wartość kursu')
plt.show()

plt.plot(l)
plt.title('Najniższa wartość kursu')
plt.show()

plt.plot(v)
plt.title('Wygenerowany wolumen')
plt.show()


# Zad 2

n_lags = 5  

X_train_ar = np.array([y_train[i:i+n_lags] for i in range(len(y_train)-n_lags)])
y_train_ar = np.array(y_train[n_lags:])

X_test_ar = np.array([y_test[i:i+n_lags] for i in range(len(y_test)-n_lags)])
y_test_ar = np.array(y_test[n_lags:])

model_ar = LinearRegression()
model_ar.fit(X_train_ar, y_train_ar)

y_pred_ar = model_ar.predict(X_test_ar)

mse_ar = mean_squared_error(y_test_ar, y_pred_ar)
print('Mean Squared Error (AR):', mse_ar)

plt.plot(y_test_ar, label='Actual values')
plt.plot(y_pred_ar, label='Predicted values')
plt.title('AR model')
plt.legend()
plt.show()



# Zad 3

model_lstm = Sequential()
model_lstm.add(LSTM(100, input_shape=(4, 1)))
model_lstm.add(Dropout(0.4))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')

X_train_lstm = np.reshape(X_train, (len(X_train), 4, 1))
X_test_lstm = np.reshape(X_test, (len(X_test), 4, 1))

y_train = np.array(y_train)
y_test = np.array(y_test)

model_lstm.fit(X_train_lstm, y_train, epochs=50, batch_size=32, verbose=1)

y_pred_lstm = model_lstm.predict(X_test_lstm)

mse_lstm = mean_squared_error(y_test, y_pred_lstm)
print('Mean Squared Error (LSTM):', mse_lstm)

plt.plot(y_test, label='Actual values')
plt.plot(y_pred_lstm, label='Predicted values')
plt.title('LSTM-model')
plt.legend()
plt.show()


# Zad 5

if mse_lstm < mse_ar:
    print('LSTM is better')
else:
    print('AR is better')
