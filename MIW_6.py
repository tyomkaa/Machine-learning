from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow import keras

#Dla danych “CIFAR10 small image classification” z biblioteki Keras zaproponuj i zrealizuj podział tych danych na dane treningowe i dane testowe

(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

y_train_full = to_categorical(y_train_full)
y_test = to_categorical(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

print("Size of training data:", X_train.shape)
print("Validation data size:", X_val.shape)
print("Size of test data:", X_test.shape)

for i in range(10):
  print(y_train[i])
  plt.imshow(X_train[i])
  plt.show()



#W oparciu o sieć konwolucyjną, zgodnie z wskazaniami prowadzącego, zaproponuj klasyfikator, klasyfikujący dwie z dziesięciu klas

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

history = model.fit(X_train, y_train[:, [0, 1]], epochs=20, batch_size=64, validation_data=(X_val, y_val[:, [0, 1]]))

model.save('miw_sXXXXX_f_{}_model_fit.h5'.format(1))


model = keras.models.load_model('miw_sXXXXX_f_{}_model_fit.h5'.format(1))
loss, acc = model.evaluate(X_train, y_train[:, [0, 1]], verbose=0)
print('Model with 1 convolutional layers:')
print('accuracy: {}'.format(acc))
print('loss: {}'.format(loss))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model2 = Sequential()

model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Flatten())
model2.add(Dense(2, activation='softmax'))

model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

history2 = model2.fit(X_train, y_train[:, [0, 1]], epochs=20, batch_size=64, validation_data=(X_val, y_val[:, [0, 1]]))

model2.save('miw_sXXXXX_f_{}_model_fit.h5'.format(1))


model2 = keras.models.load_model('miw_sXXXXX_f_{}_model_fit.h5'.format(1))
loss2, acc2 = model2.evaluate(X_train, y_train[:, [0, 1]], verbose=0)
print('Model with 2 convolutional layers:')
print('accuracy: {}'.format(acc2))
print('loss: {}'.format(loss2))

plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history2.history['binary_accuracy'])
plt.plot(history2.history['val_binary_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model3 = Sequential()

model3.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model3.add(MaxPooling2D((2, 2)))
model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))
model3.add(Conv2D(128, (3, 3), activation='relu'))
model3.add(MaxPooling2D((2, 2)))
model3.add(Flatten())
model3.add(Dense(2, activation='softmax'))

model3.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])

history3 = model3.fit(X_train, y_train[:, [0, 1]], epochs=20, batch_size=64, validation_data=(X_val, y_val[:, [0, 1]]))

model3.save('miw_sXXXXX_f_{}_model_fit.h5'.format(1))


model3 = keras.models.load_model('miw_sXXXXX_f_{}_model_fit.h5'.format(1))
loss3, acc3 = model3.evaluate(X_train, y_train[:, [0, 1]], verbose=0)
print('Model with 3 convolutional layers:')
print('accuracy: {}'.format(acc3))
print('loss: {}'.format(loss3))

plt.plot(history3.history['loss'])
plt.plot(history3.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history3.history['binary_accuracy'])
plt.plot(history3.history['val_binary_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


test_acc = [acc, acc2, acc3]
test_loss = [loss, loss2, loss3]

best_acc_index = test_acc.index(max(test_acc))
best_loss_index = test_loss.index(min(test_loss))

print("Best model based on accuracy: Model {}".format(best_acc_index+1))
print("Best model based on loss: Model {}".format(best_loss_index+1))