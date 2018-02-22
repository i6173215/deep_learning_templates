import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 10

img_x, img_y = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert images to 4D tensor
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

# need to convert to floats for tensorflow
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize the pixel values
x_train /= 255
x_test /= 255

# convert target to sparse binary class matrix
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# build a sequential model
model = Sequential()
# add a convolution layer (28x28x1 -> 28x28x32)
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu', input_shape=input_shape))
# add a pooling layer (28x28x32 -> 14x14x32)
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# add another convolution layer (14x14x32 -> 14x14x64)
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
# add another pooling layer (14x14x64 -> 7x7x64)
model.add(MaxPooling2D(pool_size=(2, 2)))
# flatten the convolutions (7x7x64 -> 3136x1)
model.add(Flatten())
# add dense layer (1000 nodes)
model.add(Dense(1000, activation='relu'))
# and one more dense layer for prediction output
model.add(Dense(num_classes, activation='softmax'))

# compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# create a class to hold the accuracy history by utilizing model callbacks
class AccuracyHistory(keras.callbacks.Callback):
    # when training starts, create empty list for accuracies
    def on_train_begin(self, logs={}):
        self.acc = []

    # when an epoch ends, append the final accuracy to the list
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


# create the accuracy history object
history = AccuracyHistory()

# figut the model - call the history object as a callback
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
# compute the overall loss and accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: {}'.format(score[0]))
print('Test accuracy: {}'.format(score[1]))
# plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(range(1, 11), history.acc)
ax.grid()
ax.set(xlabel='Epochs', ylabel='Accuracy')
plt.show()
