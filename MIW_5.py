import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Zadanie 1: Wczytaj pełny zbiór danych iris. Zaproponuj i zrealizuj podział tych danych na dane treningowe i dane testowe oraz przeprowadź na nich normalizacje

def main():
  iris = datasets.load_iris()

  X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

  mean = np.mean(X_train, axis=0)
  std = np.std(X_train, axis=0)
  X_train = (X_train - mean) / std
  X_test = (X_test - mean) / std

  def to_categorical(y, num_classes=None):
      y = np.array(y, dtype='int').ravel()
      if not num_classes:
          num_classes = np.max(y) + 1
      n = y.shape[0]
      categorical = np.zeros((n, num_classes))
      categorical[np.arange(n), y] = 1
      return categorical

  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test)

  network = NeuronNetwork(layers=[4, 10, 20, 50, 3], acti=Sigmoid(), eta=0.001)
  batch(network, X_train, y_train, epoch=1)

  plt.plot(losses)
  plt.title('Training Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.show()

  plt.plot(accuracies)
  plt.title('Training Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.show()

#Zadanie 2: Zaproponuj optymalną sieć neuronową klasyfikującą wczytane dane. Użyj nieliniowej funkcji aktywacji w warstwie ukrytej (np. sigmoidalna). Stosując propagację wsteczną błędu wytrenuj sieć metodą wsadową

losses = []
accuracies = []

class Sigmoid():
    def acti(self, s):
        return 1 / (1 + np.exp(-s))
    def der(self, x):
        return x * (1 - x)
    
class Neuron():
  def __init__(self, input, acti, eta):
    self.W=np.random.rand(input)
    self.Wb=np.random.rand(1)[0]
    self.acti=acti
    self.eta=eta

  def predict(self, x):
    return self.acti.acti(np.dot(x, self.W) + self.Wb)

  def fit(self, x, e):
    y = self.predict(x)
    error = e - y
    delta = error * self.acti.der(y)
    self.W += self.eta * delta * x
    self.Wb += self.eta * delta
    return error
  
class Layer():
  def __init__(self, input, output, acti, eta):
    self.neurons=[]
    for i in range(output):
      self.neurons.append(Neuron(input, acti, eta))

  def predict(self, x):
    y = np.array([neuron.predict(x) for neuron in self.neurons])
    return y

  def fit(self, x, e):
    errors = []
    for neuron, e in zip(self.neurons, e):
        errors.append(neuron.fit(x, e))
    return errors
  
#Batch  
class NeuronNetwork():
    def __init__(self, layers, acti, eta):
        self.layers=[]
        for i in range(1, len(layers)):
            self.layers.append(Layer(layers[i-1], layers[i], acti, eta))

    def predict(self, x):
        for layer in self.layers:
            x = layer.predict(x)
        return x

    def fit(self, X, y, e):
        x = X
        for layer in self.layers:
            x = layer.fit(x, e)
        return x


def batch(NN, X, y, epoch=100):
    for i in range(epoch):
        e = 0
        for xe, ye in zip(X, y):
            p = NN.predict(xe)
            e += (p - ye)
            NN.fit(xe, ye, e)
        e /= len(X)

    loss = np.mean(np.abs(e))
    accuracy = accuracy_score(np.argmax(y, axis=1), np.argmax(NN.predict(X), axis=1))
    losses.append(loss)
    accuracies.append(accuracy)

  
#Online
# class NeuronNetwork():
#     def __init__(self, layers, acti, eta):
#         self.layers=[]
#         for i in range(1, len(layers)):
#             self.layers.append(Layer(layers[i-1], layers[i], acti, eta))

#     def predict(self, x):
#         # Получаем выход последнего слоя
#         for layer in self.layers:
#             x = layer.predict(x)
#         return x

#     def fit(self, X, y, epoch=100):
#         for i in range(epoch):
#             losses = []
#             accuracies = []
#             for X_sample, y_sample in zip(X, y):
#                 y_pred = self.predict(X_sample)
#                 delta = y_pred - y_sample
#                 for layer in reversed(self.layers):
#                     delta = np.dot(delta, layer.W.T) * layer.neurons[0].acti.der(layer.neurons[0].last_output)
#                 
#                 for layer in self.layers:
#                     layer.fit(delta)
#                 loss = np.sum((y_sample - y_pred) ** 2) / len(y_sample)
#                 losses.append(loss)
#                 accuracy = (np.argmax(y_pred) == np.argmax(y_sample))
#                 accuracies.append(accuracy)
#             losses.append(np.mean(losses))
#             accuracies.append(np.mean(accuracies))



if __name__ == '__main__':
    main()