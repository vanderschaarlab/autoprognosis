# stdlib
from typing import Any

# third party
import numpy as np
import pandas as pd


class PREDICT_prostate:
    def __init__(self, decoder: Any) -> None:
        self.decoder = decoder

    def predict_internal(self, raw_df: pd.DataFrame, time: int) -> pd.DataFrame:
        df = self.decoder(raw_df)

        time = np.asarray(time)

        idx_cores_available = (
            np.abs(df["Number of Cores Examined"] - 12.406168) > 1e-4
        )  # since filled with mean values..

        psa_col = "PSA (ng/ml)"
        tstage_col = "Clinical T stage"
        grade_col = "Histological grade group"

        # {2: 'T1c', 5: 'T2c', 3: 'T2a', 1: 'T1b', 7: 'T3b', 0: 'T1a', 6: 'T3a', 4: 'T2b', 8: 'T4'}
        pred1 = (
            0.0026005 * ((df["Age at Diagnosis"] / 10) ** 3 - 341.155151)
            + 0.185959 * (np.log((df[psa_col] + 1) / 100) + 1.636423432)
            + 0.1614922
            * (
                (df[tstage_col] == "T2a").astype(int)
                + (df[tstage_col] == "T2b").astype(int)
                + (df[tstage_col] == "T2c")
            ).astype(int)
            + 0.39767881
            * (
                (df[tstage_col] == "T3a").astype(int)
                + (df[tstage_col] == "T3b").astype(int)
            )
            + 0.6330977 * (df[tstage_col] == "T4").astype(int)
            + 0.2791641 * (df[grade_col] == 2).astype(int)
            + 0.5464889 * (df[grade_col] == 3).astype(int)
            + 0.7411321 * (df[grade_col] == 4).astype(int)
            + 1.367963 * (df[grade_col] == 5).astype(int)
            + 1.890134
            * idx_cores_available.astype(float)
            * (
                (
                    (
                        df["Number of Cores Positive"]
                        / df["Number of Cores Examined"]
                        * 100
                        + 0.1811159
                    )
                    / 100
                )
                ** 0.5
                - 0.649019
            )
        )

        cancer_risk = 1 - np.exp(
            -np.exp(pred1)
            * np.exp(-16.40532 + 1.653947 * (np.log(time)) + 1.89e-12 * (time**3))
        )

        return cancer_risk.to_numpy()

    def predict(self, df: pd.DataFrame, times: list) -> pd.DataFrame:
        results = []
        for t in times:
            results.append(self.predict_internal(df, t)[0])

        return pd.DataFrame([results])
