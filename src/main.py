# src/main.py

import numpy as np
import torch
from config import *
from preprocess import load_and_preprocess
from graph_builder import build_graph
from gnn_model import GCN
from embedding import graph_to_embedding
from hmm_model import train_hmm
from download_data import download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def main():
    df = load_and_preprocess("data/cybersecurity_threat_detection_logs.csv", TIME_WINDOW)

    time_windows = sorted(df["time_window"].unique())

    gnn = GCN(in_dim=1, hidden_dim=8, out_dim=EMBED_DIM).to(device)

    embeddings = []

    #時間ごとにグラフを構築し、埋め込みを取得
    for t in time_windows:
        df_t = df[df["time_window"] == t]
        G = build_graph(df_t)
        emb = graph_to_embedding(G, gnn)
        embeddings.append(emb)

    X = np.vstack(embeddings)

    #HMMで時系列パターンを学習
    hmm, states = train_hmm(X, HMM_STATES)

    print("Hidden states:")
    print(states)

    print("\nTransition matrix:")
    print(hmm.transmat_)

if __name__ == "__main__":
    main()
