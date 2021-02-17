import json

import numpy as np
import tensorflow as tf

class CNNModel:
    def __init__(self, model_name, x_max):
        self.model_name = model_name
        self.model = tf.keras.models.load_model(model_name)
        self.x_max = x_max


    def _load_settings(self):
        with open('test.json', 'r') as f:
            data = json.load(f)
        return data

    def preprocess(self, op, hi, lo, cl):
        op = np.array(op)
        hi = np.array(hi)
        lo = np.array(lo)
        cl = np.array(cl)

        hightail = hi - op
        lowtail = lo - op
        body = cl - op

        x = np.stack((hightail, lowtail, body), axis=1)
        x = x/self.x_max
        x = x.reshape(-1, x.shape[0], x.shape[1])
        return x

    def predict(self, op, hi, lo, cl):
        x = self.preprocess(op, hi, lo, cl)
        return self.model.predict(x)[0][0]