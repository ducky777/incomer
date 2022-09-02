import numpy as np

from tensorflow import keras

from sklearn.metrics.pairwise import cosine_similarity


class BackTester:
    def calculate_dd(self, eq):
        end = np.argmax(np.maximum.accumulate(eq) - eq)
        start = np.argmax(eq[:end])
        return eq[end] - eq[start]

    def get_eq(self, predictions, op, verbose=1, reversed=False, min_elasped=20):
        add_balance = 50000

        trades = []
        eq = []
        store_trades = []
        entry_elasped = min_elasped

        for j, (pr, o) in enumerate(zip(predictions, op)):
            lots = round(add_balance / o)
            if reversed:
                lots = -lots
            # lots = 1
            if pr == 1 and entry_elasped >= min_elasped:
                if verbose:
                    print(f"Entered: {o} / {lots} @ {j}")
                trades.append([o, lots])
                entry_elasped = 0
            elif pr == -1 and len(trades) > 0:
                # closed.append(pnl)
                trades = []
                if verbose:
                    print(f"Closed: {o} / {lots} @ {j}")
                # entry_elasped = min_elasped
            pnl = 0
            for _, l in trades:
                pnl += (o - op[j - 1]) * l
            eq.append(pnl)
            store_trades.append(trades)
            entry_elasped += 1

        eq = np.array(eq)
        return eq

    def get_signals(self, model, x, y):
        extractor = keras.models.Model(model.input, model.layers[-3].output)
        predictions = extractor.predict(x, verbose=0)

        base = extractor.predict(x, verbose=0)

        cosine = cosine_similarity(predictions, base)

        max_signals = np.argmax(cosine, axis=1)

        signals = y[max_signals]

        csum = np.cumsum(signals, axis=1)
        argmax = np.argmax(csum, axis=1)
        argmin = np.argmin(csum, axis=1)

        sigs = []

        for ma, mi in zip(argmax, argmin):
            if mi == 0:
                sigs.append(1)
            elif ma == 0:
                sigs.append(-1)
            else:
                sigs.append(0)

        return np.array(sigs)
