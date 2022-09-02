from datetime import datetime as dt

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


class DataManager:
    def __init__(self, filename: str, cycles: list = None, num_bars: int = None):
        self.filename = filename
        self.df = pd.read_csv(
            filename,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
            header=0,
        )

        if num_bars:
            self.df = self.df.iloc[-num_bars:]

        self.df = self.combine_date_time(self.df)
        self.df = self.get_ohlc(self.df)

        if cycles is not None:
            for c in cycles:
                self.df = self.get_date_ordinal(self.df, c)

        self.highscaler = MinMaxScaler()
        self.highscaler.fit(np.array(self.df["hightail"]).reshape(-1, 1))

        self.lowscaler = MinMaxScaler()
        self.lowscaler.fit(np.array(self.df["lowtail"]).reshape(-1, 1))

        self.bodyscaler = MinMaxScaler()
        self.bodyscaler.fit(np.array(self.df["body"]).reshape(-1, 1))

        self.body = np.array(self.df["body"]).reshape(-1, 1)
        self.op = np.array(self.df.Open).reshape(-1, 1)

        self.cycles = cycles.copy()

    def combine_date_time(self, df):
        df["DateTime"] = df.Date + " " + df.Time
        return df

    def get_date_ordinal(self, df: pd.DataFrame, cycle: int):
        date_ordinal = np.array(
            [
                dt.strptime(t.replace(".", "/"), "%Y/%m/%d %H:%M").toordinal() % cycle
                for t in np.array(df.DateTime)
            ]
        )

        date_ordinal = (date_ordinal - np.min(date_ordinal)) / (
            np.max(date_ordinal) - np.min(date_ordinal)
        )

        df[f"C{cycle}"] = date_ordinal

        return df

    def get_ohlc(self, df: pd.DataFrame):
        df["hightail"] = (df.High - df.Open) / df.Open
        df["lowtail"] = (df.Low - df.Open) / df.Open
        df["body"] = (df.Open.shift(-1) - df.Open) / df.Open

        return df

    def add_lookback(self, df: pd.DataFrame, lookback: int):
        df[f"hightail{lookback}"] = df.hightail.shift(lookback)
        df[f"lowtail{lookback}"] = df.lowtail.shift(lookback)
        df[f"body{lookback}"] = df.body.shift(lookback)

        for c in self.cycles:
            df[f"C{c}{lookback}"] = df[f"C{c}"].shift(lookback)

        return df

    def get_x(self, lookbacks: int):
        df = self.df.copy()
        df = df.drop(
            ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "DateTime"],
            axis=1,
        )

        for l in tqdm(range(1, lookbacks)):
            df = self.add_lookback(df, l)

        return np.array(df).reshape(-1, lookbacks, len(self.cycles) + 3)

    def get_cumsum(self, forward_bars: int):
        y = pd.DataFrame()
        y["f0"] = self.df.body
        for f in range(1, forward_bars):
            y[f"f{f}"] = y["f0"].shift(-f)
        y = y.cumsum(axis=1, skipna=True)
        return y

    def get_y(self, forward_bars: int):
        y = self.get_cumsum(forward_bars)
        y = np.array(y)

        signals = []

        for s in y:
            if np.argmax(s) == 0:
                signals.append(-1)
            elif np.argmin(s) == 0:
                signals.append(1)
            else:
                signals.append(0)
        return np.array(signals).astype(float).reshape(-1, 1)

    def create_train_data(self, valid_split: float, lookbacks: int, forward_bars: int):
        x = self.get_x(lookbacks)[lookbacks:]
        y = self.get_y(forward_bars)[lookbacks:]

        valid_idx = int(round(len(x) * valid_split))
        xtrain = x[:valid_idx]
        ytrain = y[:valid_idx]

        xtest = x[valid_idx:]
        ytest = y[valid_idx:]

        oversample = SMOTE(sampling_strategy="not majority")
        x_over, y_over = oversample.fit_resample(
            xtrain.reshape(-1, (x.shape[1] * x.shape[2])), ytrain
        )
        x_over = x_over.reshape(-1, x.shape[1], x.shape[2])

        xtrain = np.array([*x_over, *xtrain])
        ytrain = np.array([*y_over, *ytrain])

        self.body = self.body[valid_idx + lookbacks :]
        self.op = self.op[valid_idx + lookbacks :]

        return (
            xtrain.astype(np.float32),
            ytrain.astype(np.float32),
            xtest.astype(np.float32),
            ytest.astype(np.float32),
        )
