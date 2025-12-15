import os
from pathlib import Path
import pandas as pd
import numpy as np

# ==================================================
# ✅ 전처리 함수들
# ==================================================
def preprocess_adult(
    df: pd.DataFrame,
    target_col: str = "income",
    drop_cols=("education",),
):
    df = df.copy()

    # 1) 불필요 컬럼 drop
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # 2) object 컬럼 정리 + '?' 포함 행 제거
    obj_cols = df.select_dtypes(include=["object", "category"]).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    mask_has_q = (df[obj_cols] == "?").any(axis=1)
    df = df.loc[~mask_has_q].reset_index(drop=True)

    # 3) target (income) -> binary
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    t = df[target_col].str.replace(".", "", regex=False).str.strip()
    mapping_target = {"<=50K": 0, ">50K": 1}
    df[target_col] = t.map(mapping_target)

    if df[target_col].isna().any():
        bad = sorted(t[df[target_col].isna()].unique())
        raise ValueError(f"Unmapped income values: {bad}")

    df[target_col] = df[target_col].astype("int32")

    # 4) feature label encoding
    feature_obj_cols = [c for c in obj_cols if c != target_col]
    mappings = {"__target__": mapping_target}

    for col in feature_obj_cols:
        uniq = sorted(df[col].unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        df[col] = df[col].map(mapping).astype("int32")
        mappings[col] = mapping

    return df, mappings


def preprocess_fintech(
    df: pd.DataFrame,
    target_col: str = "churn",
    drop_user_id: bool = True,
):
    df = df.copy()

    # 0) (옵션) user id drop
    if drop_user_id and "user" in df.columns:
        df.drop(columns=["user"], inplace=True)

    # 1) age 결측 → 행 삭제
    if "age" not in df.columns:
        raise KeyError("Column 'age' not found.")
    df = df.dropna(subset=["age"]).reset_index(drop=True)

    # 2) credit_score: indicator + 0 대치
    if "credit_score" not in df.columns:
        raise KeyError("Column 'credit_score' not found.")
    df["credit_score_missing"] = df["credit_score"].isna().astype("int32")
    df["credit_score"] = df["credit_score"].fillna(0)

    # 3) rewards_earned: 0 대치
    if "rewards_earned" not in df.columns:
        raise KeyError("Column 'rewards_earned' not found.")
    df["rewards_earned"] = df["rewards_earned"].fillna(0)

    # 4) target 확인/정리
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")
    df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype("int32")

    uniq_t = set(df[target_col].unique())
    if not uniq_t.issubset({0, 1}):
        raise ValueError(f"Target '{target_col}' must be binary 0/1. Got: {sorted(uniq_t)}")

    # 5) object 컬럼 label encoding
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    mappings = {}

    for col in obj_cols:
        s = df[col].astype(str).str.strip()
        uniq = pd.Index(s.unique()).sort_values()
        mapping = {v: i for i, v in enumerate(uniq)}
        df[col] = s.map(mapping).astype("int32")
        mappings[col] = mapping

    # 6) dtype 정리 (선택)
    for c in df.columns:
        if df[c].dtype == "int64":
            df[c] = df[c].astype("int32")

    return df, mappings


def preprocess_abalone(
    df: pd.DataFrame,
    rings_threshold: int = 14,
    target_col: str = "target"
):
    df = df.copy()

    # 1) Sex label encoding
    if "Sex" not in df.columns:
        raise KeyError("Column 'Sex' not found.")
    sex_values = sorted(df["Sex"].unique())
    sex_mapping = {v: i for i, v in enumerate(sex_values)}
    df["Sex"] = df["Sex"].map(sex_mapping).astype("int32")

    # 2) rings -> binary target
    if "rings" not in df.columns:
        raise KeyError("Column 'rings' not found.")
    df[target_col] = (df["rings"] >= rings_threshold).astype("int32")

    # 3) drop rings
    df.drop(columns=["rings"], inplace=True)

    mappings = {"Sex": sex_mapping, "rings_threshold": rings_threshold}
    return df, mappings


def preprocess_bank(
    df: pd.DataFrame,
    target_col: str = "y",
    add_missing_indicators: bool = True,
    missing_token: str = "__MISSING__",
):
    df = df.copy()

    # 1) target y -> 0/1
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    t = df[target_col].astype(str).str.strip().str.lower()
    df[target_col] = t.map({"no": 0, "yes": 1})

    if df[target_col].isna().any():
        bad = sorted(set(t[df[target_col].isna()].unique()))
        raise ValueError(f"Unmapped target values in '{target_col}': {bad}")

    df[target_col] = df[target_col].astype("int32")

    # 2) object 컬럼 결측 처리
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    obj_cols = [c for c in obj_cols if c != target_col]

    mappings = {"__target__": {"no": 0, "yes": 1}}

    # 결측 indicator 추가
    if add_missing_indicators:
        for col in obj_cols:
            if df[col].isna().any():
                df[f"{col}_missing"] = df[col].isna().astype("int32")

    # 결측을 토큰으로 채우고 라벨인코딩
    for col in obj_cols:
        s = df[col].where(~df[col].isna(), other=missing_token)
        s = s.astype(str).str.strip()

        uniq = pd.Index(s.unique()).sort_values()
        mapping = {v: i for i, v in enumerate(uniq)}
        df[col] = s.map(mapping).astype("int32")
        mappings[col] = mapping

    # 3) int64 -> int32 통일
    for c in df.columns:
        if df[c].dtype == "int64":
            df[c] = df[c].astype("int32")

    return df, mappings


def preprocess_car(
    df: pd.DataFrame,
    target_col: str = "target",
    n_majority: int = 1324,
    n_minority: int = 58,
    random_state: int = 42,
):
    df = df.copy()
    df["class"] = df["class"].astype(str).str.strip()

    # 1) 클래스 분리
    maj_df = df[df["class"].isin(["unacc", "acc"])]
    min_df = df[df["class"].isin(["good", "vgood"])]

    # 2) downsampling
    if len(maj_df) < n_majority or len(min_df) < n_minority:
        print(f"[CAR] 샘플 수 부족: majority={len(maj_df)}, minority={len(min_df)}")
        raise ValueError("Not enough samples to match paper counts.")

    maj_df = maj_df.sample(n_majority, random_state=random_state)
    min_df = min_df.sample(n_minority, random_state=random_state)

    # 3) target 생성
    maj_df[target_col] = 0
    min_df[target_col] = 1
    df_bin = pd.concat([maj_df, min_df]).reset_index(drop=True)

    # 4) feature label encoding
    feature_cols = [c for c in df_bin.columns if c not in ["class", target_col]]
    mappings = {}

    for col in feature_cols:
        uniq = sorted(df_bin[col].astype(str).unique())
        mapping = {v: i for i, v in enumerate(uniq)}
        df_bin[col] = df_bin[col].astype(str).map(mapping).astype("int32")
        mappings[col] = mapping

    df_bin.drop(columns=["class"], inplace=True)
    return df_bin, mappings


def preprocess_heart_2020_cleaned(
    df: pd.DataFrame,
    target_col: str = "HeartDisease",
    drop_sleep_time: bool = True,
):
    """
    heart_2020_cleaned 전처리
    - Yes/No -> 1/0 매핑 (여러 컬럼)
    - Diabetic은 4가지 케이스를 0/1로 매핑
    - Sex: Female=0, Male=1
    - GenHealth, AgeCategory: 순서형 인코딩
    - BMI: log1p 변환
    - SleepTime drop
    - Race: get_dummies(drop_first=True)
    - 중복행 제거
    """
    df = df.copy()

    # -------------------------
    # 1) 기본 Yes/No 매핑
    # -------------------------
    d_yesno = {"Yes": 1, "No": 0}
    d_sex = {"Female": 0, "Male": 1}

    yesno_cols = [
        "HeartDisease",
        "Smoking",
        "AlcoholDrinking",
        "Stroke",
        "DiffWalking",
        "PhysicalActivity",
        "Asthma",
        "KidneyDisease",
        "SkinCancer",
    ]

    # 컬럼 존재 체크 + 매핑
    for c in yesno_cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found.")
        df[c] = df[c].astype(str).str.strip().map(d_yesno)

        if df[c].isna().any():
            bad = sorted(df.loc[df[c].isna(), c].astype(str).unique())
            raise ValueError(f"[heart] Unmapped values in '{c}': {bad}")

        df[c] = df[c].astype("int32")

    # -------------------------
    # 2) Diabetic 매핑 (특수 케이스)
    # -------------------------
    if "Diabetic" not in df.columns:
        raise KeyError("Column 'Diabetic' not found.")

    diabetic_map = {
        "Yes": 1,
        "Yes (during pregnancy)": 1,
        "No": 0,
        "No, borderline diabetes": 0,
    }
    df["Diabetic"] = df["Diabetic"].astype(str).str.strip().map(diabetic_map)

    if df["Diabetic"].isna().any():
        bad = sorted(df.loc[df["Diabetic"].isna(), "Diabetic"].astype(str).unique())
        raise ValueError(f"[heart] Unmapped values in 'Diabetic': {bad}")

    df["Diabetic"] = df["Diabetic"].astype("int32")

    # -------------------------
    # 3) Sex 매핑
    # -------------------------
    if "Sex" not in df.columns:
        raise KeyError("Column 'Sex' not found.")
    df["Sex"] = df["Sex"].astype(str).str.strip().map(d_sex)

    if df["Sex"].isna().any():
        bad = sorted(df.loc[df["Sex"].isna(), "Sex"].astype(str).unique())
        raise ValueError(f"[heart] Unmapped values in 'Sex': {bad}")

    df["Sex"] = df["Sex"].astype("int32")

    # -------------------------
    # 4) 순서형 인코딩: AgeCategory, GenHealth
    # -------------------------
    age_order = [
        "18-24", "25-29", "30-34", "35-39", "40-44",
        "45-49", "50-54", "55-59", "60-64", "65-69",
        "70-74", "75-79", "80 or older"
    ]
    health_order = ["Poor", "Fair", "Good", "Very good", "Excellent"]

    if "AgeCategory" not in df.columns:
        raise KeyError("Column 'AgeCategory' not found.")
    if "GenHealth" not in df.columns:
        raise KeyError("Column 'GenHealth' not found.")

    df["AgeCategory"] = df["AgeCategory"].astype(str).str.strip().map({v: i for i, v in enumerate(age_order)})
    df["GenHealth"] = df["GenHealth"].astype(str).str.strip().map({v: i for i, v in enumerate(health_order)})

    if df["AgeCategory"].isna().any():
        bad = sorted(df.loc[df["AgeCategory"].isna(), "AgeCategory"].astype(str).unique())
        raise ValueError(f"[heart] Unmapped values in 'AgeCategory': {bad}")
    if df["GenHealth"].isna().any():
        bad = sorted(df.loc[df["GenHealth"].isna(), "GenHealth"].astype(str).unique())
        raise ValueError(f"[heart] Unmapped values in 'GenHealth': {bad}")

    df["AgeCategory"] = df["AgeCategory"].astype("int32")
    df["GenHealth"] = df["GenHealth"].astype("int32")

    # -------------------------
    # 5) BMI log1p
    # -------------------------
    if "BMI" not in df.columns:
        raise KeyError("Column 'BMI' not found.")
    df["BMI"] = pd.to_numeric(df["BMI"], errors="raise")
    df["BMI"] = np.log1p(df["BMI"])

    # -------------------------
    # 6) SleepTime drop
    # -------------------------
    if drop_sleep_time and "SleepTime" in df.columns:
        df.drop(columns=["SleepTime"], inplace=True)

    # -------------------------
    # 7) Race 원핫 (drop_first=True)
    # -------------------------
    if "Race" not in df.columns:
        raise KeyError("Column 'Race' not found.")
    race_dummies = pd.get_dummies(df["Race"].astype(str).str.strip(), prefix="Race", dtype="int32", drop_first=True)
    df = pd.concat([df, race_dummies], axis=1)
    df.drop(columns=["Race"], inplace=True)

    # -------------------------
    # 8) 중복 제거
    # -------------------------
    df = df.drop_duplicates().reset_index(drop=True)

    # int64 -> int32 통일 (선택)
    for c in df.columns:
        if df[c].dtype == "int64":
            df[c] = df[c].astype("int32")

    return df


# ==================================================
# ✅ 메인 처리 로직
# ==================================================
FULL_FILES = [
    "abalone_full.csv",
    "adult_full.csv",
    "bank_full.csv",
    "cardio_full.csv",
    "car_full.csv",
    "fintech_full.csv",
    "heloc_full.csv",
    "heart_2020_cleaned.csv",   # ✅ 추가
]

# 파일명(키) -> 전처리 함수 매핑
# (없는 애들은 전처리 없이 그대로 저장)
PREPROCESSORS = {
    "abalone_full.csv": lambda df: preprocess_abalone(df)[0],
    "adult_full.csv":   lambda df: preprocess_adult(df)[0],
    "bank_full.csv":    lambda df: preprocess_bank(df)[0],
    "car_full.csv":     lambda df: preprocess_car(df)[0],
    "fintech_full.csv": lambda df: preprocess_fintech(df)[0],
    "heart_2020_cleaned.csv": lambda df: preprocess_heart_2020_cleaned(df),  # ✅ 추가
    # cardio_full.csv, heloc_full.csv 는 일부러 없음(전처리 없이 저장)
}

def full_to_origin_name(filename: str) -> str:
    """
    abalone_full.csv -> abalone_origin.csv
    heart_2020_cleaned.csv -> heart_2020_cleaned_origin.csv  (그냥 _origin만 붙임)
    """
    p = Path(filename)
    stem = p.stem  # 확장자 제거
    # _full.csv 패턴이면 _origin으로 치환, 아니면 뒤에 _origin 붙임
    if stem.endswith("_full"):
        out_stem = stem.replace("_full", "_origin")
    else:
        out_stem = f"{stem}_origin"
    return out_stem + p.suffix

def build_origin_from_raw(
    raw_dir: str = "raw_datas",
    out_dir: str = "origin",
    files: list[str] | None = None,
    encoding: str | None = None,  # 필요하면 "utf-8", "cp949" 등
):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if files is None:
        files = FULL_FILES

    for fname in files:
        in_path = raw_dir / fname
        if not in_path.exists():
            print(f"[SKIP] 파일 없음: {in_path}")
            continue

        # -------------------------
        # 1) 로드 (파일별 sep 분기 포함)
        # -------------------------
        try:
            if "cardio_full" in str(in_path):
                df = pd.read_csv(in_path, encoding=encoding, sep=";")
            else:
                df = pd.read_csv(in_path, encoding=encoding)
        except UnicodeDecodeError:
            # 인코딩 지정 안 했는데 터지면, 흔한 케이스로 한 번 더 시도
            if "cardio_full" in str(in_path):
                df = pd.read_csv(in_path, encoding="cp949", sep=";")
            else:
                df = pd.read_csv(in_path, encoding="cp949")

        # -------------------------
        # 2) 전처리 (있으면 적용, 없으면 그대로)
        # -------------------------
        if fname in PREPROCESSORS:
            try:
                df_out = PREPROCESSORS[fname](df)
                print(f"[OK] 전처리 적용: {fname}  (rows={len(df)} -> {len(df_out)}, cols={df.shape[1]} -> {df_out.shape[1]})")
            except Exception as e:
                print(f"[FAIL] 전처리 실패: {fname}  | 에러: {e}")
                raise
        else:
            df_out = df
            print(f"[OK] 전처리 없음(그대로 저장): {fname}  (rows={len(df_out)}, cols={df_out.shape[1]})")

        # -------------------------
        # 3) 저장: origin 폴더에 *_origin.csv로
        # -------------------------
        out_name = full_to_origin_name(fname)
        out_path = out_dir / out_name
        df_out.to_csv(out_path, index=False)
        print(f"     -> 저장 완료: {out_path}")

    print("\n✅ 전체 완료!")

if __name__ == "__main__":
    build_origin_from_raw(
        raw_dir="dataset/raw_datas",
        out_dir="dataset/origin",
        files=FULL_FILES,
        encoding=None,
    )
