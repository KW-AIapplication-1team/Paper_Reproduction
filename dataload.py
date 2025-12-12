# This file contains code for loading datasets from the UCI Machine Learning Repository
# using the ucimlrepo and datasets libraries.
# pip install ucimlrepo
# pip install datasets

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
from datasets import load_dataset

def make_full_df(uci_id: int, target_name: str):
    ds = fetch_ucirepo(id=uci_id)
    X = ds.data.features
    y = ds.data.targets.squeeze()
    return pd.concat([X, y.rename(target_name)], axis=1)

path = './raw_datas'

os.makedirs(path, exist_ok=True)


# 1) Adult (id=2) [web:2]
df_adult = make_full_df(uci_id=2, target_name="income")
df_adult.to_csv(f"{path}/adult_full.csv", index=False)

# 2) Abalone (id=1) [web:18][web:25]
df_abalone = make_full_df(uci_id=1, target_name="rings")  # 필요하면 이름 변경
df_abalone.to_csv(f"{path}/abalone_full.csv", index=False)

# 3) Bank Marketing (id=222) [web:26][web:28]
df_bank = make_full_df(uci_id=222, target_name="y")
df_bank.to_csv(f"{path}/bank_full.csv", index=False)

# 4) Car Evaluation (id=19) [web:35][web:20]
df_car = make_full_df(uci_id=19, target_name="class")
df_car.to_csv(f"{path}/car_full.csv", index=False)

# 5) Yeast (id=110) [web:61][web:59]
df_yeast = make_full_df(uci_id=110, target_name="class")
df_yeast.to_csv(f"{path}/yeast_full.csv", index=False)

# 6) FICO HELOC dataset from Hugging Face Datasets
heloc = load_dataset("mstz/heloc")  # FICO HELOC dataset [web:39]
df_heloc = heloc["train"].to_pandas()
df_heloc.to_csv(f"{path}/heloc_full.csv", index=False)