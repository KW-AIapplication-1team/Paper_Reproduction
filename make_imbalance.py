import argparse
from pathlib import Path
import pandas as pd

SEED = 42

# 처리할 데이터셋(이 논문 controlled-imbalance 4개)
DATASETS = {
    "adult_origin":   {"target": "income",     "imb_ratio": 0.015},
    "heloc_origin":   {"target": "is_at_risk",  "imb_ratio": 0.02},
    "fintech_origin": {"target": "churn",      "imb_ratio": 0.013},
    "cardio_origin":  {"target": "cardio",     "imb_ratio": 0.018},
}


def ensure_binary_01(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    target이 0/1이 아닐 수 있으니 안전하게 0/1로 맞춘다.
    - 이미 {0,1}이면 그대로
    - {False, True}면 0/1로 변환
    - {0.0,1.0} 같은 경우 int로 캐스팅
    그 외는 에러(재현 파이프라인에서 조용히 망가지는 것 방지)
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe columns.")

    s = df[target]

    # bool -> int
    if s.dtype == bool:
        df[target] = s.astype(int)
        return df

    # 숫자형인데 0/1로 표현 가능하면 int 변환
    uniq = pd.Series(s.dropna().unique()).sort_values().tolist()
    try:
        uniq_set = set(int(float(u)) for u in uniq)
        if uniq_set <= {0, 1}:
            df[target] = s.astype(int)
            return df
    except Exception:
        pass

    raise ValueError(
        f"[{target}] must be binary 0/1 (or bool). Found unique values: {uniq}"
    )


def make_splits(
    real: pd.DataFrame,
    target: str,
    testsize: int,
    imb_ratio: float,
):
    """
    - test: balanced (testsize/2 each for class 0 and 1)
    - train(imbalanced_noord): keep ALL majority(0) from remaining,
      and sample minority(1) = int(imb_ratio * num_majority_remaining)
      where imb_ratio is minority/majority.
    """
    if testsize % 2 != 0:
        raise ValueError("testsize must be even because test is balanced (1:1).")

    TEST_CLASS = testsize // 2

    # class 1 is minority, class 0 is majority (요구사항)
    maj = real[real[target] == 0]
    mino = real[real[target] == 1]

    if len(maj) < TEST_CLASS or len(mino) < TEST_CLASS:
        raise ValueError(
            f"Not enough samples to create balanced test of size {testsize}.\n"
            f"Need >= {TEST_CLASS} per class. Got: "
            f"majority(0)={len(maj)}, minority(1)={len(mino)}"
        )

    # balanced test
    test0 = maj.sample(TEST_CLASS, random_state=SEED)
    test1 = mino.sample(TEST_CLASS, random_state=SEED)
    test = pd.concat([test0, test1])

    # remaining (disjoint)
    remaining = real[~real.index.isin(test.index)]

    imb0 = remaining[remaining[target] == 0]  # keep all majority
    min_count = int(imb_ratio * len(imb0))

    if min_count <= 0:
        raise ValueError(
            f"imb_ratio={imb_ratio} makes min_count={min_count}. "
            f"Increase imb_ratio or check majority size."
        )

    rem_min = remaining[remaining[target] == 1]
    if len(rem_min) < min_count:
        raise ValueError(
            f"Not enough minority left after test split.\n"
            f"Need minority={min_count} (= {imb_ratio} * {len(imb0)} majority), "
            f"but remaining minority={len(rem_min)}.\n"
            f"Tip: reduce testsize or reduce imb_ratio."
        )

    imb1 = rem_min.sample(min_count, random_state=SEED)
    imb_df = pd.concat([imb1, imb0])

    # shuffle + reset index
    imb_df = imb_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = test.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return test, imb_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin_dir", type=str, default="dataset/origin",
                        help="Folder containing raw csvs (adult.csv, heloc.csv, fintech.csv, cardio.csv)")
    parser.add_argument("--out_dir", type=str, default="dataset",
                        help="Output folder (dataset/<name>/...)")
    parser.add_argument("--testsize", type=int, default=4000,
                        help="Balanced test size (must be even). Default 4000 = 2000+2000")
    parser.add_argument("--ext", type=str, default=".csv",
                        help="File extension in origin_dir (default .csv)")
    args = parser.parse_args()

    origin_dir = Path(args.origin_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, cfg in DATASETS.items():
        target = cfg["target"]
        imb_ratio = cfg["imb_ratio"]

        src = origin_dir / f"{name}{args.ext}"
        if not src.exists():
            raise FileNotFoundError(f"Missing source file: {src}")

        df = pd.read_csv(src)
        df = ensure_binary_01(df, target)

        test_df, imb_df = make_splits(df, target, args.testsize, imb_ratio)

        # dataset/<name>/ 폴더 생성
        ddir = out_dir / name.replace("_origin", "")
        ddir.mkdir(parents=True, exist_ok=True)

        # 원본도 dataset/<name>/original.csv 로 저장(재현 편의)
        df.to_csv(ddir / "original.csv", index=False)
        test_df.to_csv(ddir / "test.csv", index=False)
        imb_df.to_csv(ddir / "imbalanced_noord.csv", index=False)

        # 로그 출력
        tc = test_df[target].value_counts().to_dict()
        ic = imb_df[target].value_counts().to_dict()
        print(f"[{name}] target={target}, imb_ratio={imb_ratio}")
        print(f"  saved: {ddir/'original.csv'}")
        print(f"  saved: {ddir/'test.csv'}             counts={tc}")
        print(f"  saved: {ddir/'imbalanced_noord.csv'} counts={ic}")
        print("-" * 70)


if __name__ == "__main__":
    main()
