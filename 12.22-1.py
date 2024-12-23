def inception_module(layer_in, f1, f2, f3):
conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
return layer_out