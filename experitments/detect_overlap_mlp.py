import pandas as pd
import argparse

from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def boundary_function_helper(X_train, y_train, X_test, y_test, test):
    """
    - MLP 학습 후 test에 대해 predict_proba 계산
    - selection용으로 p_majority=P(class=0), p_minority=P(class=1)만 붙여서 반환
    """

    # ✅ MLP는 스케일링이 거의 필수라 Pipeline로 묶음
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(256, 128),   # 기본값(검색으로 더 좋은 구조 선택 가능)
                activation="relu",               # ✅ 비선형성 학습: ReLU 명시
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                batch_size=256,
                max_iter=400,
                early_stopping=True,
                n_iter_no_change=15,
                validation_fraction=0.1,
                random_state=42,
            ))
        ]
    )

    # XGB(200 estimators) 대비 "조금 크게" + 비선형 잘 학습하도록 후보 몇 개만 둠
    param_dist = {
        "mlp__hidden_layer_sizes": [
            (256, 128),
            (256, 128, 64),
            (384, 192),
        ],
        "mlp__alpha": [1e-5, 1e-4, 1e-3],
        "mlp__learning_rate_init": [5e-4, 1e-3, 2e-3],
        "mlp__batch_size": [128, 256],
        "mlp__max_iter": [400],  # 고정(조금 크게)
    }

    rand_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=6,
        cv=5,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )

    # 라벨이 0/1이 아닐 가능성 대비(안전)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    rand_search.fit(X_train, y_train)
    best_model = rand_search.best_estimator_

    # 확률: shape (n_samples, n_classes)
    proba_test = best_model.predict_proba(X_test)

    # ✅ classes_ 순서가 항상 [0,1]이라고 가정하지 말고 인덱스를 안전하게 매핑
    classes = best_model.named_steps["mlp"].classes_.tolist()
    idx0 = classes.index(0)
    idx1 = classes.index(1)

    df1 = test.copy()
    df1["p_majority"] = proba_test[:, idx0].astype(float)  # P(class=0)
    df1["p_minority"] = proba_test[:, idx1].astype(float)  # P(class=1)

    return df1


def find_boundary(df, TARGET, overlap_num, RANDOM_STATE=42):
    """
    목적:
    - majority(label=0) 샘플들 중에서
      P(class=0)=p_majority 가 '낮은 순'으로 overlap_num개를 cond=1로 지정
    - 나머지 majority는 cond=0
    - minority(label=1)는 cond=2
    """

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    # 0: majority, 1: minority 가정
    df_class1 = df[df[TARGET] == 1]  # minority
    df_class0 = df[df[TARGET] == 0]  # majority

    # majority 먼저, minority 나중 + ✅ 인덱스 리셋(중요)
    df = pd.concat([df_class0, df_class1], axis=0).reset_index(drop=True)

    # majority 인덱스를 반으로 쪼개 2 split (앞쪽 len(df_class0) 구간이 majority)
    start = [0, len(df_class0) // 2]
    end = [len(df_class0) // 2, len(df_class0)]

    for i in range(2):
        print(f"Split {i+1}")

        # 이번 fold의 majority 구간만 test로 떼고 나머지는 train
        maj_test_idx = list(range(start[i], end[i]))
        train = df.drop(maj_test_idx, axis=0)
        test = df.loc[maj_test_idx].copy()

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        X_train = train.drop(TARGET, axis=1)
        y_train = train[TARGET]
        X_test = test.drop(TARGET, axis=1)
        y_test = test[TARGET]

        if i == 0:
            df1 = boundary_function_helper(X_train, y_train, X_test, y_test, test)
        else:
            df2 = boundary_function_helper(X_train, y_train, X_test, y_test, test)

    # 두 fold 결과 합치기
    bnd = pd.concat([df1, df2], axis=0, ignore_index=True)

    # ✅ [핵심] majority(0) 중에서 P(class=0)=p_majority 낮은 순으로 overlap_num개 선택
    maj_all = bnd[bnd[TARGET] == 0].copy()
    maj_sorted = maj_all.sort_values("p_majority", ascending=True)
    selected = maj_sorted.index[:overlap_num]

    print("Total majority samples (in bnd):", len(maj_sorted))
    print("Requested overlap_num:", overlap_num)
    print("Selected overlap majority:", len(selected))
    print("-" * 100)

    # cond 기본값: 모두 clear majority(0)
    bnd["cond"] = 0
    # 선택된 overlap_num개 majority만 cond=1
    bnd.loc[selected, "cond"] = 1

    # selection에 쓰던 컬럼 제거 + TARGET 제거
    bnd = bnd.drop(["p_majority", "p_minority", TARGET], axis=1)

    # minority 쪽은 항상 cond = 2
    df_class1 = df_class1.copy()
    df_class1.loc[:, "cond"] = 2
    df_class1 = df_class1.drop(TARGET, axis=1)

    # majority(0/1) + minority(2) 합치기
    bnd = pd.concat([bnd, df_class1], axis=0, ignore_index=True)

    return bnd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataname", type=str, default="adult")
    parser.add_argument("--overlap_num", type=int, default=218, help="선택할 overlap majority 개수")
    parser.add_argument("--target", type=str, default="target")
    args = parser.parse_args()

    DATANAME = args.dataname
    OVERLAP_NUM = args.overlap_num
    TARGET = args.target

    path = f"data/{DATANAME}/imbalanced_noord.csv"
    df = pd.read_csv(path)

    bndry = find_boundary(df, TARGET, overlap_num=OVERLAP_NUM)

    out_path = f"data/{DATANAME}/imbalanced_ord.csv"
    bndry.to_csv(out_path, index=False)

    print(f"Saved Ternary Target successfully (MLP, overlap_num={OVERLAP_NUM}) → {out_path}")
