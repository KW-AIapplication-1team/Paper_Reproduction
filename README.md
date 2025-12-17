# Paper_Reproduction

본 저장소는  
**Synthetic Tabular Data Generation for Imbalanced Classification: The Surprising Effectiveness of an Overlap Class**  
논문의 방법론을 재현하고, ORD 및 CTabSyn 기반 합성 데이터 생성 기법의 효과를 실험적으로 검증하기 위한 코드와 실험 결과를 포함한다.

---

## Data Collection

- 실험에 사용된 실제 tabular 데이터 수집 및 전처리 코드는  
  `data_collect/` 폴더에서 확인할 수 있다.
- 공개 데이터셋을 기반으로 논문에서 정의한 문제 설정에 맞게 데이터를 구성하였다.

---

## Model Training

- ORD 및 CTabSyn 기반 생성 모델의 학습 과정과 로그는  
  `ORD/ctabsyn/ORD_modeltrain.ipynb` 에서 확인할 수 있다.
- VAE 학습과 잠재 공간 diffusion 학습을 포함한 전체 생성 파이프라인이 정리되어 있다.

---

## Model Checkpoints

학습된 모델 체크포인트는 아래 경로에서 확인 가능하다.

- **CTabSyn (Diffusion model)**  
  [ctabsyn_checkpoint](https://drive.google.com/drive/folders/1RulvPr79VHGNKPsc1voRcX9nzs2X7Zwu?usp=drive_link)

- **VAE model**  
  [vae_checkpoint](https://drive.google.com/drive/folders/1kpNwg9SoCEfRk5lQ6ruxu2nASUWnBbW4?usp=drive_link)

---

## Model Evaluation

- 생성 데이터 및 ORD 적용 결과에 대한 분류 성능 평가는  
  `ORD/ORD_evaluation.ipynb` 에서 확인할 수 있다.
- XGBoost 및 다수의 기본 분류 모델을 사용한 성능 비교 결과가 포함되어 있다.

---

## Experiments and Extensions

- ORD 단계에서 사용하는 모델 변경 실험
- overlap 샘플 수 변화에 따른 성능 분석
- 기존 선형 오버샘플링 기법과의 비교 실험  
  위 확장 실험들은 `experiments/` 폴더에서 확인할 수 있다.
