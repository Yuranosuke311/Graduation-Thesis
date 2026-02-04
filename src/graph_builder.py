#グラフ構築

# src/graph_builder.py

import networkx as nx

def build_graph(df_window):
    G = nx.Graph()

    for _, row in df_window.iterrows():
        u = f"user_{row['user_id']}"
        r = f"res_{row['resource_id']}"

        G.add_node(u, node_type="user")
        G.add_node(r, node_type="resource")

        G.add_edge(u, r, weight=row["access_count"])

    return G
