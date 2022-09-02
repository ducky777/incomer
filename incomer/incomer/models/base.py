import numpy as np
from incomer.financials.backtest import BackTester
from tensorflow import keras


class BaseModel(BackTester):
    def __init__(self):
        super().__init__()

    def predict_all(self, save_path: str):
        model = keras.models.load_model(f"{save_path}.h5")

        sigs = self.get_signals(model)
        eq = self.get_eq(sigs, reversed=False)

        model = keras.models.load_model(f"{save_path}_reverse.h5")

        sigs = self.get_signals(model)
        eq2 = self.get_eq(sigs, reversed=True)

        model = keras.models.load_model(f"{save_path}_revsig.h5")

        sigs = self.get_signals(model)
        eq3 = self.get_eq(sigs, reversed=False, rev_signal=True)

        model = keras.models.load_model(f"{save_path}_revsig_reverse.h5")

        sigs = self.get_signals(model)
        eq4 = self.get_eq(sigs, reversed=True, rev_signal=True)

        eq = eq + eq2 + eq3 + eq4

        return np.array(eq)

    def train(
        self,
        x: np.array,
        y: np.array,
        xtest: np.array,
        xbase: np.array,
        ybase: np.array,
        op: np.array,
        epochs: int = 999999999,
        batch_size: int = 16,
        valid_splits: float = 0.9,
        save_path: str = "data/best.h5",
    ):
        best_eq = 2
        best_rev = 2

        for ii in range(epochs):
            self.model.fit(
                x,
                y,
                epochs=1,
                validation_split=valid_splits,
                batch_size=batch_size,
                shuffle=True,
                verbose=0,
            )
            predictions = self.get_signals(self.model, xtest, xbase, ybase)
            eq = self.get_eq(predictions, op=op, verbose=False)
            try:
                dd = self.calculate_dd(eq)
            except Exception:
                continue
            if dd == 0:
                dd = 1
            score = np.sum(eq) / abs(dd)
            if score > best_eq:
                self.model.save(f"{save_path}.h5")
                best_eq = score
                print(f"BEST: {best_eq}")
            elif -score > best_rev:
                self.model.save(f"{save_path}_reverse.h5")
                best_rev = -score
                print(f"BEST REVERSE: {best_rev}")
            else:
                print(f"{ii} -> {score}")

            eq = self.get_eq(predictions, op=op, verbose=False, rev_signal=True)
            try:
                dd = self.calculate_dd(eq)
            except Exception:
                continue
            if dd == 0:
                dd = 1
            score = np.sum(eq) / abs(dd)
            if score > best_eq:
                self.model.save(f"{save_path}_revsig.h5")
                best_eq = score
                print(f"BEST: {best_eq}")
            elif -score > best_rev:
                self.model.save(f"{save_path}_revsig_reverse.h5")
                best_rev = -score
                print(f"BEST REVERSE: {best_rev}")
            else:
                print(f"{ii} -> {score}")
