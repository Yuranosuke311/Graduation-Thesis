#グラフ埋め込み生成

# src/embedding.py

import torch
import numpy as np
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

def graph_to_embedding(G, model,device):
    data = from_networkx(G).to(device)
    data.x = torch.ones((data.num_nodes, 1),device=device)  # ノード特徴なし
    #ノード特徴を「1」から意味のあるものへ（bytes_sum / access_count / threat_label 由来）変更したい　02/08

    model.eval()
    with torch.no_grad():
        node_emb = model(data.x, data.edge_index)

    graph_emb = node_emb.mean(dim=0).numpy()
    return graph_emb
