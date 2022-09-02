import json

import numpy as np
import tensorflow as tf

class CNNModel:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
        self.model = tf.keras.models.load_model('%s/%s' %
                                                (self.model_path,
                                                 self.model_name))
        self.info = self._load_settings()

    def _load_settings(self):
        with open('%s/vars.json' % self.model_path, 'r') as f:
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
        x = x/self.info['x_max']
        # x = (x - self.info['x_mean']) /  self.info['x_std']
        x = x.reshape(-1, x.shape[0], x.shape[1])
        return x[::-1]

    def predict(self, op, hi, lo, cl):
        x = self.preprocess(op, hi, lo, cl)
        signal = self.model.predict(x)
        if signal[0][1] > 0.05:
            return 1
        elif signal[0][2] > 0.05:
            return -1
        return 0