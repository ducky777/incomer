import pandas as pd
import numpy as np


class PriceEngine:
    def __init__(self):
        pass

    def add_data(self, file_path: str):
        self.df = pd.read_csv(
            file_path,
            names=["Date", "Time", "Open", "High", "Low", "Close", "Volume"],
            header=0,
        )

        self.date, self.time, self.op, self.hi, self.low, self.cl, self.vol = np.array(
            self.df
        )
