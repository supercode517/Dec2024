from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import plot_model
def naive_inception_module(layer_in, f1, f2, f3):
    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out
visible = Input(shape=(256, 256, 3))
layer = naive_inception_module(visible, 64, 128, 32)
model = Model(inputs=visible, outputs=layer)
model.summary()
plot_model(model, show_shapes=True, to_file='naive_inception_module.png')
