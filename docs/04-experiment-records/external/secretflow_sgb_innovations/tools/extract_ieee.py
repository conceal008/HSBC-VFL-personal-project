"""从官方 IEEE-CIS 全量表中抽出团队 vfl_official 的 PSI 交集行，按 split_ids.csv 的时间切分顺序写出，
供 SGB 复现使用。只读引用原始数据，不改动任何原文件。

    python3 extract_ieee.py <全量数据集目录> <split_ids.csv> <输出目录>
"""
import gzip, sys, time
from pathlib import Path
import pandas as pd

raw, split_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
t0 = time.time()
ids = pd.read_csv(split_csv)
keep = set(ids["TransactionID"].tolist())
parts = []
for chunk in pd.read_csv(raw / "train_transaction.csv", chunksize=50_000, low_memory=False):
    parts.append(chunk[chunk["TransactionID"].isin(keep)])
tx = pd.concat(parts).set_index("TransactionID").loc[ids["TransactionID"]].reset_index()
idn = pd.read_csv(raw / "train_identity.csv")
idn.columns = [c.replace("id-", "id_") for c in idn.columns]
idn = idn.set_index("TransactionID").loc[ids["TransactionID"]].reset_index()
assert (tx["TransactionID"].to_numpy() == ids["TransactionID"].to_numpy()).all()
assert (tx["isFraud"].to_numpy() == ids["isFraud"].to_numpy()).all()
n = len(tx); k = 4
for i in range(k):
    sl = slice(i * n // k, (i + 1) * n // k)
    with gzip.open(out / f"ieee_tx_part{i}.csv.gz", "wt", compresslevel=3) as f:
        tx.iloc[sl].to_csv(f, index=False)
with gzip.open(out / "ieee_identity.csv.gz", "wt", compresslevel=3) as f:
    idn.to_csv(f, index=False)
ids.to_csv(out / "split_ids.csv", index=False)
print(f"DONE tx_cols={tx.shape[1]} id_cols={idn.shape[1]} {time.time() - t0:.0f}s", flush=True)
