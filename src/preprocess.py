#（ログ前処理）
# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(csv_path, time_window="1h"):
    df = pd.read_csv(csv_path)

    # timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    df["time_window"] = df["timestamp"].dt.floor(time_window)

    # === ユーザ / リソース定義 ===
    user_encoder = LabelEncoder()
    res_encoder = LabelEncoder()

    df["user_id"] = user_encoder.fit_transform(df["source_ip"])
    df["resource_id"] = res_encoder.fit_transform(df["dest_ip"])

    # === 集約（時間窓 × ユーザ × リソース）===
    agg = (
        df.groupby(["time_window", "user_id", "resource_id"])
          .agg(
              #通信量
              bytes_sum=("bytes_transferred", "sum"),
              #アクセス回数
              access_count=("bytes_transferred", "count"),
              #脅威ラベル（最頻値）
              threat_label=("threat_label", lambda x: x.mode()[0])
          )
          .reset_index()
    )

    return agg

