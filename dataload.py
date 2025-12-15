# This file contains code for loading datasets from the UCI Machine Learning Repository
# using the ucimlrepo and datasets libraries.
# pip install ucimlrepo datasets kagglehub

import os
import glob
import pandas as pd
from ucimlrepo import fetch_ucirepo
from datasets import load_dataset
import kagglehub


# =========================
# 공통 유틸
# =========================
def make_full_df(uci_id: int, target_name: str):
    ds = fetch_ucirepo(id=uci_id)
    X = ds.data.features
    y = ds.data.targets.squeeze()
    return pd.concat([X, y.rename(target_name)], axis=1)


def load_single_csv_from_dir(dir_path: str) -> pd.DataFrame:
    csvs = glob.glob(os.path.join(dir_path, "*.csv"))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV found in {dir_path}")
    if len(csvs) > 1:
        print(f"[WARN] Multiple CSVs found, using: {csvs[0]}")
    return pd.read_csv(csvs[0])


# =========================
# 저장 경로
# =========================
path = "./dataset/raw_datas"
os.makedirs(path, exist_ok=True)


# =========================
# 1) UCI datasets
# =========================

# Adult
df_adult = make_full_df(uci_id=2, target_name="income")
df_adult.to_csv(f"{path}/adult_full.csv", index=False)

# Abalone
df_abalone = make_full_df(uci_id=1, target_name="rings")
df_abalone.to_csv(f"{path}/abalone_full.csv", index=False)

# Bank Marketing
df_bank = make_full_df(uci_id=222, target_name="y")
df_bank.to_csv(f"{path}/bank_full.csv", index=False)

# Car Evaluation
df_car = make_full_df(uci_id=19, target_name="class")
df_car.to_csv(f"{path}/car_full.csv", index=False)

# Yeast - 다중분류인데 정보가 없어서 사용의 어려움
# df_yeast = make_full_df(uci_id=110, target_name="class")
# df_yeast.to_csv(f"{path}/yeast_full.csv", index=False)

# 6) FICO HELOC dataset from Hugging Face Datasets
heloc = load_dataset("mstz/heloc")  # FICO HELOC dataset [web:39]
df_heloc = heloc["train"].to_pandas()
df_heloc.to_csv(f"{path}/heloc_full.csv", index=False)


# =========================
# 2) Kaggle datasets (ORD 논문용)
# =========================

# ---- Cardio (Kaggle) ----
cardio_dir = kagglehub.dataset_download(
    "sulianova/cardiovascular-disease-dataset"
)
df_cardio = load_single_csv_from_dir(cardio_dir)

# 논문에서는 target이 이진 (presence / absence of disease)
# 보통 컬럼명이 'cardio'
df_cardio.to_csv(f"{path}/cardio_full.csv", index=False)


# ---- Fintech (Kaggle) ----
fintech_dir = kagglehub.dataset_download(
    "niketdheeryan/fintech-users-data"
)
df_fintech = load_single_csv_from_dir(fintech_dir)

# 논문 기준: churn / purchase 여부 (이진)
df_fintech.to_csv(f"{path}/fintech_full.csv", index=False)


print("✅ All datasets saved as *_full.csv in", path)