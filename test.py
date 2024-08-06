import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.utils import to_categorical

print("Starting the script...")

# Splitting data into test and train
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("Data loaded...")

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
print("Data reshaped...")

# Converting class vectors into binary
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("Class vectors converted to binary...")

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_test /= 255
print("Data normalized...")

batch_size = 128
num_classes = 10
epochs = 11

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print("Model created...")

model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(),
              metrics=['accuracy'])
print("Model compiled...")

print("Starting training...")
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
print("Training completed...")

score = model.evaluate(x_test, y_test, verbose=0)
print("Evaluation completed...")

print("loss", score[0])
print("accuracy", score[1])

model.save('mnist.h5')
print("Model saved...")
