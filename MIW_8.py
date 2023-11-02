import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Dense

# Wczytanie danych mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
print(x_train.shape)
print(x_test.shape)

# Dodanie wymiaru kanału dla danych wejściowych
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Definicja modelu
input_shape = x_train.shape[1:]
input_img = Input(shape=input_shape)

# Encoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
encoded_flat = Flatten()(encoded)
classification = Dense(10, activation='softmax')(encoded_flat)

# Odseparowanie modelu encodera
encoder = Model(input_img, encoded)

# Decoder
x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(x)

# Zdefiniowanie modelu autokodera
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Trenowanie autokodera
autoencoder.fit(x_train, x_train,
                epochs=10,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# Testowanie na przykładowych danych
decoded_imgs = autoencoder.predict(x_test)

# Wyświetlenie przykładowych obrazów
for i in range(20):
    print(y_test[i])

    # Dane wejściowe
    plt.subplot(1, 3, 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.title('Dane wejściowe')

    # Dane zwrócone przez enkoder
    plt.subplot(1, 3, 2)
    encoded_img = encoder.predict(x_test[i][np.newaxis])
    plt.imshow(encoded_img.squeeze(), cmap='gray')
    plt.title('Dane zwrócone przez enkoder')

    # Dane wyjściowe
    plt.subplot(1, 3, 3)
    plt.imshow(decoded_imgs[i].squeeze(), cmap='gray')
    plt.title('Dane wyjściowe')

    plt.show()
