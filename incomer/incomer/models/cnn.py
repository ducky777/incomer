import tensorflow as tf
import tensorflow.keras.backend as k
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Flatten,
    Input,
    MaxPooling1D,
    SpatialDropout1D,
)


class TradeCNN:
    def __init__(
        self,
    ):
        pass

    def res_block(x, filters, size, stride, downsample=False):
        y = Conv1D(filters, size, (1 if not downsample else 2), padding="same")(x)
        y = Activation("relu")(y)
        y = BatchNormalization()(y)
        y = SpatialDropout1D(0.3)(y)
        y = Conv1D(filters, size, 1, padding="same")(y)

        if downsample:
            x = Conv1D(filters=filters, kernel_size=1, strides=2, padding="same")(x)

        out = Add()([x, y])
        out = Activation("relu")(out)
        out = BatchNormalization()(out)
        return out

    def build_model(self, lookbacks: int):

        start_model = Input(shape=(lookbacks - 1, 3))
        input_model = BatchNormalization()(start_model)
        # input_model = SpatialDropout1D(0.3)(start_model)
        input_model = Conv1D(
            kernel_size=7, strides=1, filters=32, padding="same", dilation_rate=5
        )(input_model)
        input_model = self.res_block(start_model, 64, 5, 2, downsample=True)
        input_model = self.res_block(input_model, 128, 3, 3, downsample=True)
        input_model = self.res_block(input_model, 256, 2, 3, downsample=True)
        input_model = MaxPooling1D()(input_model)
        input_model = Flatten()(input_model)
        input_model = BatchNormalization()(input_model)
        input_model = Dense(24, activation="relu")(input_model)
        input_model = BatchNormalization()(input_model)
        outputs = Dense(1, activation="linear")(input_model)

        model = Model([start_model], [outputs])

        model.compile(loss="mse", optimizer="adam")

        return model
