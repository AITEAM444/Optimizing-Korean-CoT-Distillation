# 언어적 불일치 해소를 통한 한국어 CoT Distillation 최적화

**논리 기반 다단계 데이터 필터링의 효용성 분석**

> Optimizing Korean CoT Distillation by Resolving Linguistic Mismatch:
> A Multi-Stage Logic-Based Filtering Pipeline

생성형 AI 기술 및 응용 · 4팀 — 정현우 · 박겸 · 이승연 · 이다혜

---

## 1. 연구 배경

최근 Frontier LLM의 고도화된 reasoning 능력을 SLM으로 전이시키는 **CoT Distillation**은 자원 제약 환경에서 모델의 지능을 극대화하는 핵심 패러다임으로 자리 잡았다. Teacher 모델이 생성한 reasoning trace를 학습 데이터로 활용하는 이 방식은 데이터의 품질이 student 모델의 최종 성능을 결정짓는 핵심 변수가 된다. 이를 최적화하기 위해 영어 도메인에서는 PPL, IFD 등 다양한 데이터 큐레이션 방법론이 제안되고 그 효용성이 검증된 바 있다 [2].

그러나 이러한 큐레이션 기법들은 대부분 영어 중심의 데이터 환경에서 설계되었으며, 한국어와 같은 형태론적 복잡성이 높은 언어 [7]나 사고 언어 간의 불일치(Language Mismatch) [6]가 발생하는 다국어 환경에서의 검증은 여전히 미비한 상태이다. 특히 영어와 한국어 사이의 자원 불균형으로 인해 발생하는 **잠재적 번역체** 및 **논리적 비약** 문제는 기존의 영어 지향적 필터링 기법만으로는 포착하기 어렵다.

본 연구에서는 이러한 학술적 공백을 메우기 위해, 한국어 CoT 데이터에 특화된 최적의 큐레이션 파이프라인을 탐색한다. 실험의 객관성을 확보하고 추론의 논리적 무결성을 정밀하게 측정하기 위해, ground-truth가 명확히 정의된 수학 및 일반상식 도메인을 중심으로 각 필터링 방법론의 성능과 비용 효율성을 비교 분석하고자 한다.

---

## 2. 연구 목표

본 연구의 최종 목적은 한국어 CoT 데이터의 지능적 선별을 통해, sLLM이 최소한의 데이터셋으로 최적의 추론 성능을 확보할 수 있는 전략적 필터링 파이프라인을 규명하는 데 있다. 이를 위해 다음과 같은 세부 목표를 설정한다.

### 2.1. 한국어 특화 필터링 파이프라인 설계

영어 중심의 기존 필터링 기법을 그대로 적용하는 것의 한계를 분석하고, Frontier LLM의 한국어 synthetic data가 가지는 논리적 비약과 번역체 노이즈를 제어할 수 있는 필터링을 제안한다.

### 2.2. 데이터 효율성의 극대화

전체 중 2~5%인 고품질 데이터만을 선별하여 학습 [1]시켰을 때, 더 많은 데이터를 사용한 baseline 대비 성능 유지율 및 향상도를 측정해 **Performance per Token**을 정량화하여 비용 효율성을 평가한다.

---

## 3. 연구 가설

### Q1. 기존 영어 통계 지표의 한국어 변별력 한계

영어 도메인에서 유효성이 검증된 통계적 큐레이션 기법(PPL, IFD 등)은 한국어 CoT 데이터에서 **동일한 수준의 변별력을 보이지 못할 것이다**.

> 이는 글로벌 Teacher 모델의 사고 언어와 출력 언어 간의 불일치(Language Mismatch) [6]로 인해, 논리적 난이도와 언어적 노이즈가 혼재되어 지표의 신뢰도를 저하시키기 때문이다.

### Q2. NLI + 한국어 언어 필터 조합의 우위

단계별 논리적 함축(entailment)을 검증하는 **NLI 기반 필터링**과 한국어 특유의 비문/번역투를 제어하는 **언어 필터링의 조합**이 기존 기법 대비 student 모델의 추론 성능 향상에 더 크게 기여할 것이다.

> 이는 한국어의 형태론적 복잡성 [7]과 번역체를 명시적으로 제거함으로써 모델의 학습 효율을 극대화하기 때문이다.

---

## 4. 필터링 방법론

### 4.1. Baseline Filtering

- **구조적 필터링**: 데이터의 형식, 최소/최대 길이 등 추론 데이터로서의 기본 규격 검증
- **결과 기반 필터링 (Outcome-based)**: ground truth와 모델의 도출 답안이 일치하지 않는 샘플은 즉시 제거

### 4.2. Model-based Metric Filtering

- **PPL**: 문장의 자연스러움 측정. 너무 낮으면 단순 반복으로 간주하고 버리고, 너무 높으면 비문 혹은 hallucination으로 판단해서 제거하는 **band-pass 방식** 적용
- **IFD (Instruction-Following Difficulty)**: student 모델이 느끼는 학습 난이도를 정량화하여, 모델의 현재 수준에 부합하는 고품질 샘플을 선별

### 4.3. Advanced Filtering

- **LLM-as-a-Judge**: Frontier 모델을 활용해 추론 과정의 유창성과 타당성을 정량적 평가
- **PRM (Process Reward Model)**: 추론의 각 단계별로 보상을 부여해, 답만 맞고 과정이 틀린 샘플 제거

### 4.4. 핵심 제안: 한국어 논리 기반 다단계 파이프라인

#### Pre — 언어 기반 필터링

> 한국어의 번역체 문제 / 형태론적 복잡성 사전 해결

- 한국어 특유의 번역체를 분석한 작은 **rule-based** 필터 적용
- **K-PPL**: PPL을 측정할 때 한국어로만 사전 학습된 모델 사용 → 번역투 문장에 대해 높은 PPL을 출력하므로 잘 잡아낼 수 있음 (글로벌 모델은 상대적으로 관대함)
- 전체 토큰 수 대비 **형태소 개수**를 측정해 의미 없는 문자 나열 제거 → 형태론적으로 풍부한 데이터 선별

#### Mid — 논리 기반 필터링

> 한국어의 논리적 모순 발견 (논리적 비약 제거)

- **NLI 기반 step-level 검증**: KLUE-RoBERTa-Large, KoBigBird를 NLI 태스크로 fine-tuning한 모델 사용. 정답은 맞았지만 논리적으로 말이 안 되는 샘플 제거
- **PRM**: 위 결과에서 남은 고난도 수학 문제들에 대해서만 실행. 수학 성능을 추가로 개선하면서, 기존 모델을 사용함으로써 비용 절감

#### Tail — 데이터 선별

- **IFD** 적용해서 적은 양의 고품질 데이터만 선별

---

## 5. 실험 설계

### 5.1. 데이터셋

- **Yi-Sang** (KO-REAson 선행 논문 데이터셋)
  - [https://huggingface.co/datasets/KOREAson/YiSang-STEM_Code-Unfiltered](https://huggingface.co/datasets/KOREAson/YiSang-STEM_Code-Unfiltered) — 필터링 전 raw CoT 데이터, 해당 도메인만 골라서 사용
- 다른 Teacher 모델로 같은 데이터에 대해서 CoT 데이터 추출하여 사용 (multi-source)
- 추가 수학 데이터셋: GSM8K, MATH, Date, Commonsense reasoning

### 5.2. 사용 모델

- **Teacher**: 공개 데이터 활용 (위의 raw 데이터셋 기반) / multi-teacher 구현을 위해 추가 Frontier 모델 사용 (Llama 3.1 70B, GPT-4o-mini 등)
- **Student**: 1B–3B SLM (**Phi-3.5 mini**)
  - 여유가 있다면 한국어 특화 모델 추가 사용

### 5.3. 평가 지표

#### A. 논리적 추론 평가

- **LogicKor** — 한국어 논리 평가
- **MGSM** — 다국어 수학 추론

#### B. 학습 효율 평가

- **큐레이션 비용 vs 성능 향상** — 정량화 가능성 검토
- **데이터 양 vs 품질 trade-off 곡선** — 전체 데이터에서 몇 %를 필터링해서 남겨야 효율 상승인지 확인
- **Performance per Token** — 토큰당 성능 정량화

#### C. 한국어 문장 품질

- **FLORES-200** (Meta multilingual)
- **KLUE** — 한국어 NLU 벤치마크

---

## 6. 실험 과정

### Step 1. Multi-source Synthetic Generation

Yi-Sang 기반의 raw CoT 데이터 + 서로 다른 Teacher 모델(Llama 3.1 70B, GPT-4o-mini 등)을 사용해 동일 질문에 대한 다양한 reasoning trace 확보.

### Step 2. Pre-Filtering (전처리)

[필터링 방법론] 부분의 **Pre 과정**을 이용해 문맥적으로 어색하거나 비문 제거 + outcome-based filtering을 통한 전처리.

### Step 3. Mid-stage Ablation 비교

다음 순서대로 적용해서 성능을 비교하는 ablation 실험:

- **Baseline** (Pre만 적용)
  - 기본적인 언어 전처리가 완료된 경우(한국어 비문, 번역체 해결)

- **Baseline → NLI**
  - [6]에서 지적하듯 다국어 모델의 추론 저하는 언어적 노이즈에서 기인하기에 NLI를 통해 언어적 무결성을 확보함으로써 sLLM의 학습 효율을 높이고자 함

- **Baseline → NLI → PRM**
  - 위의 방식에서 부족한 수학적 깊이를 난이도가 높은 문제들에 한해서 적용해 성능을 높이고자 함

### Step 4. Distillation & Multi-facet Evaluation

선별된 고품질의 작은 데이터로 student 모델 학습 후, **LogicKor, KLUE, MGSM** 평가.

---

## 7. 참고 문헌

[1] **Pushing on Multilingual Reasoning Models with Language-Mixed Chain-of-Thought.** arXiv:2510.04230, 2025.
[https://arxiv.org/abs/2510.04230](https://arxiv.org/abs/2510.04230)
→ 비영어권 언어의 추론 능력 극대화에는 Language-Mixed CoT 기법이 실험적으로 좋다.

[2] **The Quest for Efficient Reasoning: A Data-Centric Benchmark to CoT Distillation (DC-CoT).** arXiv:2505.18759, 2025.
[https://arxiv.org/abs/2505.18759](https://arxiv.org/abs/2505.18759)
→ CoT 증류 과정에서 데이터 중심 접근법의 효과를 평가하는 벤치마크 제안 (영어 데이터를 가지고 어떻게 증강·선별·혼합하냐에 따라 sLLM의 추론 성능이 달라짐을 분석).

[3] **KORMo: Korean Open Reasoning Model for Everyone.** arXiv:2510.09426, 2025.
[https://arxiv.org/abs/2510.09426](https://arxiv.org/abs/2510.09426)
→ 고품질의 합성 데이터가 뒷받침된다면, CoT 데이터가 부족한 비영어권에서도 충분히 추론 모델을 만들 수 있음을 실증한 연구.

[4] **The Signal is in the Steps: Local Scoring for Reasoning Data Selection.** arXiv:2510.03988, 2025.
[https://arxiv.org/abs/2510.03988](https://arxiv.org/abs/2510.03988)
→ 추론 데이터 선별 시 전체 답변의 확률보다 개별 단계의 신뢰도가 더 중요함을 입증. 본 연구에서 NLI를 사용하는 것이 더 정확한 성능을 보일 것이라는 근거.

[5] **Hugging Face Blog: Synthetic data — save money, time and carbon with open source.**
[https://huggingface.co/blog/synthetic-data-save-costs](https://huggingface.co/blog/synthetic-data-save-costs)

[6] **When Models Reason in Your Language: Controlling Thinking Language Comes at the Cost of Accuracy.** arXiv:2505.22888, 2025.
[https://arxiv.org/abs/2505.22888](https://arxiv.org/abs/2505.22888)
→ LRM들이 비영어권 언어로는 깊게 사고하지 못하는 본질적 한계 지적. 질문이 비영어권인 경우, 사고 과정을 해당 언어로 유지하지 못하고 영어로 생각하거나 여러 언어가 파편화되어 섞이는 현상이 빈번. 한국어 기반 SLM에 한국어 reasoning 데이터가 필요한 이유 (모델이 한국어로 질문을 받아도 번역 과정 없이 한국어 개념 체계 안에서 즉각적으로 추론할 수 있도록) — SLM의 추론 속도 증가 및 최적화 / 복잡한 추론에서도 성능 증가.

[7] **Why do language models perform worse for morphologically complex languages?** arXiv:2411.14198, 2024.
[https://arxiv.org/abs/2411.14198](https://arxiv.org/abs/2411.14198)
→ 언어의 구조적 차이로 인해서, 같은 크기의 데이터여도 안의 정보량은 다르다. 따라서 언어별 특성을 반영하여 토큰 효율을 극대화하는 설계가 필요. 언어별로 다른 구조의 필터링이 필요하다는 것의 근거.

[8] **Perplexed by Perplexity: Perplexity-Based Data Pruning With Small Reference Models.** arXiv:2405.20541, 2024.
[https://arxiv.org/abs/2405.20541](https://arxiv.org/abs/2405.20541)
→ 작은 모델로 데이터를 정제해서 큰 모델을 학습시켜도 성능 향상이 일어남. 데이터셋의 특징에 따라 필터링해야 할 기준이 달라짐. 필터링 과정이 충분히 저자원으로도 가능함을 시사.

---

<sub>생성형 AI 기술 및 응용 · 4팀 · 2026.05</sub>
