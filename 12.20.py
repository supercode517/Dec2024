from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
model = Sequential()
model.add(Conv2D(512, (3,3), padding='same', activation='relu', input_shape=(256, 256, 3)))
model.summary()