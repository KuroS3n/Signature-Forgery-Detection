# capsnet.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense

def build_capsnet(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (9, 9), activation='relu')(inputs)
    x = Conv2D(32, (9, 9), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model