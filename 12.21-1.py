from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.utils import plot_model
def vgg_block(layer_in, n_filters, n_conv):
for _ in range(n_conv):
layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)
layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
return layer_in
visible = Input(shape=(256, 256, 3))
layer = vgg_block(visible, 64, 2)
layer = vgg_block(layer, 128, 2)
layer = vgg_block(layer, 256, 4)
model = Model(inputs=visible, outputs=layer)
model.summary()
plot_model(model, show_shapes=True, to_file='multiple_vgg_blocks.png')