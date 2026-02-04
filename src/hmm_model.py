#状態遷移

# src/hmm_model.py

import numpy as np
from hmmlearn.hmm import GaussianHMM

def train_hmm(X, n_states):
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=200
    )
    model.fit(X)
    states = model.predict(X)

    return model, states
