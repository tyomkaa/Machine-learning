import os
import cv2
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense

# Zad 1
data_dir = "D:/images"  
train_dir = os.path.join(data_dir, "train")


# Zad 2
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

num_classes = len(train_generator.class_indices)

model = Sequential()

model.add(ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
))
model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

model.fit(train_generator, epochs=10)

model.save('wma_fit.h5'.format(1))


#Zad 3

face_cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

model = keras.models.load_model('wma_fit.h5') 

def recognize_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    class_labels = train_generator.class_indices.keys()

    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0

        predictions = model.predict(face_img)
        class_index = np.argmax(predictions[0])
        class_label = list(class_labels)[class_index]
        confidence = predictions[0][class_index]

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f'{class_label} ({confidence:.2f})', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognize_faces(frame)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()