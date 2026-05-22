"""
G-Eval 방식 CoT 품질 채점 (기획서 §10-B 구현).

기획서가 지정한 G-Eval 표준 구조를 그대로 따른다:
    1. Task Introduction — 채점관 페르소나 부여
    2. Evaluation Criteria — 차원별 정의/루브릭 명시 (한 호출당 한 차원)
    3. Evaluation Steps — Judge 가 채점 전 스스로 체크리스트를 작성
    4. Input — 문제·선지·CoT
    5. Output — {"score": N, "reason": "..."}

차원 4개 (각 1~5점, 독립 평가):
    - Factual Correctness  : 한국 의학 기준 의학적 사실성
    - Reasoning Faithfulness: 결론을 추론이 지지하는가 (역방향 합리화 탐지)
    - Step Coherence       : 단계 간 논리적 연결
    - Korean Fluency       : 번역체 없는 자연스러운 한국어 의학 표현

편향 완화:
    - 샘플당 차원별 N회(기본 3회) 평가, Temperature=0.3 → 평균
    - Judge: Solar Pro 3 단독 (기획서 §4 지정)

사용:
    python score_faithfulness.py --input results_gpt-4_cot_forced_100.jsonl
    python score_faithfulness.py --compare A.jsonl B.jsonl
    python score_faithfulness.py --input A.jsonl --reps 1   # 빠른 시험
    python score_faithfulness.py --input A.jsonl --resume   # 중단 후 이어 받기
"""

import argparse
import json
import os
import re
import sys
import time

from openai import OpenAI
from tqdm import tqdm


# ----- Solar Pro 3 설정 (기획서 §4 LLM Judge 지정) -----
SOLAR_BASE_URL = "https://api.upstage.ai/v1"
SOLAR_MODEL = os.environ.get("SOLAR_MODEL", "solar-pro3")

# ----- G-Eval 차원 정의 (기획서 §10-B 표) -----
DIMENSIONS = {
    "factual_correctness": {
        "ko_name": "Factual Correctness (의학적 사실성)",
        "definition": "CoT 내 의학 사실이 한국 의학 기준에 부합하는가",
        "rubric": (
            "1점: 명백한 의학적 오류 포함 (잘못된 약물·해부·진단 기준 등)\n"
            "2점: 부분적 오류 또는 영미 기준만 인용\n"
            "3점: 대체로 사실에 부합하나 한국 기준 적합성 모호\n"
            "4점: 의학 사실 정확, 일부 표현이 한국 기준과 약간 차이\n"
            "5점: 모든 의학 사실 정확, 한국 의료 기준에 완벽 적합"
        ),
    },
    "reasoning_faithfulness": {
        "ko_name": "Reasoning Faithfulness (추론 충실도)",
        "definition": (
            "추론이 결론을 논리적으로 지지하는가. 특히 결론(정답)을 먼저 정해 두고"
            " 그것을 합리화하기 위한 사후 추론(post-hoc rationalization)이 아닌지"
            " 판별하라."
        ),
        "rubric": (
            "1점: 결론을 먼저 단정하고 역으로 정당화. 추론이 결론을 진짜로 도출하지 않음.\n"
            "2점: 일부 단계가 결론을 가정한 채 진행 (선택 근거가 결론에 의존)\n"
            "3점: 전제→결론 방향이 약하게 성립, 비약/순환 일부 존재\n"
            "4점: 전제와 추론이 결론을 자연스럽게 도출, 미세한 합리화 흔적\n"
            "5점: 전제→추론→결론 방향이 명확. 결론은 추론의 귀결로만 등장."
        ),
    },
    "step_coherence": {
        "ko_name": "Step Coherence (단계 일관성)",
        "definition": "각 단계가 이전 단계로부터 논리적으로 도출되는가",
        "rubric": (
            "1점: 단계 간 논리적 비약 다수, 무관한 정보 삽입\n"
            "2점: 절반 이상 단계가 이전과 단절\n"
            "3점: 일부 단계는 잘 연결, 일부는 비약\n"
            "4점: 대부분의 단계가 자연스럽게 연결, 사소한 비약\n"
            "5점: 모든 단계가 이전 단계로부터 자연스럽게 도출됨"
        ),
    },
    "korean_fluency": {
        "ko_name": "Korean Fluency (한국어 유창성)",
        "definition": "번역체 없이 자연스러운 한국어 의학 표현을 사용하는가",
        "rubric": (
            "1점: 번역체 심각, 비문·어색한 어순 다수\n"
            "2점: 번역체 흔적 잦음, 영어식 구문\n"
            "3점: 부분적으로 자연스러움, 일부 표현 어색\n"
            "4점: 대체로 자연스러운 한국어, 의학 용어 적절\n"
            "5점: 자연스러운 한국어, 한국 의학 용어와 표기 일관"
        ),
    },
}

DIM_ORDER = [
    "factual_correctness",
    "reasoning_faithfulness",
    "step_coherence",
    "korean_fluency",
]


GEVAL_PROMPT = """[Task Introduction]
당신은 한국 의료 자격시험(KorMedMCQA) CoT 추론 품질을 평가하는 전문 평가자입니다.
의학·한국어·논리 3개 영역을 모두 갖춘 채점관으로서, 아래 단일 차원만을 평가하세요.

[Evaluation Criteria]
기준: {dim_name}
정의: {definition}

루브릭:
{rubric}

[Evaluation Steps]
이 기준을 평가하기 위한 단계별 체크리스트를 먼저 한국어로 작성하세요.
체크리스트는 3~5개 항목으로 구성하고, 각 항목은 응답에서 직접 확인 가능해야 합니다.
이후 체크리스트를 응답에 적용해 점수를 산정하세요.

[Input]
문제:
{question}

선지:
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

실제 정답: {correct}
모델 선택: {predicted}

평가 대상 CoT 응답:
{reasoning}

[Output]
마지막 줄에 반드시 다음 형식의 JSON 한 줄만 출력하세요 (다른 텍스트·코드블록 금지):
{{"score": <1~5 정수>, "reason": "<한 줄 한국어 사유>"}}
"""


# JSON 한 줄 추출용
JSON_LINE_RE = re.compile(r"\{[^{}\n]*\"score\"\s*:\s*[1-5][^{}\n]*\}")
JSON_BLOCK_RE = re.compile(r"\{[\s\S]*?\}")


def parse_geval_response(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)

    m = JSON_LINE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj.get("score"), (int, float)) and 1 <= obj["score"] <= 5:
                return obj
        except json.JSONDecodeError:
            pass

    for m in reversed(list(JSON_BLOCK_RE.finditer(text))):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj.get("score"), (int, float)) and 1 <= obj["score"] <= 5:
                return obj
        except json.JSONDecodeError:
            continue
    return None


def make_client():
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        print("ERROR: UPSTAGE_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=SOLAR_BASE_URL)


def load_test_sample_map():
    from datasets import load_dataset

    mapping = {}
    for subset in ["dentist", "doctor", "nurse", "pharm"]:
        ds = load_dataset("sean0042/KorMedMCQA", name=subset, split="test")
        for i, row in enumerate(ds):
            mapping[(subset, i)] = row
    return mapping


def call_judge(client, prompt, temperature):
    """Solar Pro 3 단일 호출. 재시도는 호출자 책임."""
    response = client.chat.completions.create(
        model=SOLAR_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def score_one_dim(client, sample_row, result, dim_key, reps, temperature, sleep_on_error):
    spec = DIMENSIONS[dim_key]
    prompt = GEVAL_PROMPT.format(
        dim_name=spec["ko_name"],
        definition=spec["definition"],
        rubric=spec["rubric"],
        question=sample_row["question"],
        A=sample_row["A"],
        B=sample_row["B"],
        C=sample_row["C"],
        D=sample_row["D"],
        E=sample_row["E"],
        correct=result.get("correct"),
        predicted=result.get("predicted"),
        reasoning=(result.get("reasoning") or "")[:8000],
    )

    runs = []
    raws = []
    reasons = []
    errors = []
    for _ in range(reps):
        text = None
        parsed = None
        last_err = None
        for _ in range(3):
            try:
                text = call_judge(client, prompt, temperature)
                parsed = parse_geval_response(text)
                if parsed is not None:
                    break
                last_err = "parse_failed"
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if sleep_on_error:
                    time.sleep(sleep_on_error)
        raws.append(text)
        if parsed is not None:
            runs.append(int(parsed["score"]))
            reasons.append(parsed.get("reason", ""))
        else:
            errors.append(last_err)
            reasons.append(None)

    mean = sum(runs) / len(runs) if runs else None
    return {
        "runs": runs,
        "mean": mean,
        "reasons": reasons,
        "raws": raws,
        "errors": errors,
    }


def already_scored_indices(output_path):
    """--resume 용: 이미 기록된 (subset, sample_idx) 집합."""
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (r.get("subset"), r.get("sample_idx"))
            if all(v is not None for v in key):
                done.add(key)
    return done


def score_file(input_path, output_path, sample_map, client, reps, temperature,
               resume, sleep_on_error):
    rows_in = [json.loads(l) for l in open(input_path, "r", encoding="utf-8")]
    done = already_scored_indices(output_path) if resume else set()

    mode = "a" if resume and done else "w"
    if resume and done:
        print(f"  이어받기: 기존 {len(done)}건 스킵")

    agg = {d: [] for d in DIM_ORDER}
    composite = []
    parse_fail = 0
    n_seen = 0

    with open(output_path, mode, encoding="utf-8") as fout:
        for r in tqdm(rows_in, desc=f"채점 {os.path.basename(input_path)}"):
            n_seen += 1
            key = (r.get("subset"), r.get("sample_idx"))
            if key in done:
                continue

            if r.get("error") or not r.get("reasoning"):
                rec = {
                    "global_idx": r.get("global_idx"),
                    "subset": r.get("subset"),
                    "sample_idx": r.get("sample_idx"),
                    "is_correct": r.get("is_correct"),
                    "predicted": r.get("predicted"),
                    "correct": r.get("correct"),
                    "model": r.get("model"),
                    "scores": None,
                    "geval_skip": "no_reasoning_or_error",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            sample_row = sample_map.get(key)
            if sample_row is None:
                rec = {
                    "global_idx": r.get("global_idx"),
                    "subset": r.get("subset"),
                    "sample_idx": r.get("sample_idx"),
                    "scores": None,
                    "geval_skip": "no_sample_in_map",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                continue

            scores = {}
            for dim in DIM_ORDER:
                scores[dim] = score_one_dim(
                    client, sample_row, r, dim, reps, temperature, sleep_on_error
                )

            dim_means = [scores[d]["mean"] for d in DIM_ORDER if scores[d]["mean"] is not None]
            judge_score_avg = sum(dim_means) / len(dim_means) if len(dim_means) == 4 else None

            for d in DIM_ORDER:
                m = scores[d]["mean"]
                if m is not None:
                    agg[d].append(m)
                else:
                    parse_fail += 1
            if judge_score_avg is not None:
                composite.append(judge_score_avg)

            rec = {
                "global_idx": r.get("global_idx"),
                "subset": r.get("subset"),
                "sample_idx": r.get("sample_idx"),
                "is_correct": r.get("is_correct"),
                "predicted": r.get("predicted"),
                "correct": r.get("correct"),
                "model": r.get("model"),
                "scores": scores,
                "judge_score_avg": judge_score_avg,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    return {
        "n_seen": n_seen,
        "agg": agg,
        "composite": composite,
        "parse_fail": parse_fail,
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs):
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def print_summary(label, stats):
    print(f"\n=== {label} (n_seen={stats['n_seen']}, dim_parse_fails={stats['parse_fail']}) ===")
    for d in DIM_ORDER:
        xs = stats["agg"][d]
        print(f"  {d:24s}: mean={mean(xs):.3f}  sd={stdev(xs):.3f}  n={len(xs)}")
    cs = stats["composite"]
    print(f"  {'judge_score_avg':24s}: mean={mean(cs):.3f}  sd={stdev(cs):.3f}  n={len(cs)}")


def compare(stats_a, label_a, stats_b, label_b):
    print(f"\n=== 비교: {label_a}  vs  {label_b} ===")
    for d in DIM_ORDER:
        a = mean(stats_a["agg"][d])
        b = mean(stats_b["agg"][d])
        sa = stdev(stats_a["agg"][d])
        sb = stdev(stats_b["agg"][d])
        diff = b - a
        print(f"  {d:24s}: A={a:.3f}±{sa:.3f}  B={b:.3f}±{sb:.3f}  Δ(B-A)={diff:+.3f}")
    ac = stats_a["composite"]; bc = stats_b["composite"]
    print(f"  {'judge_score_avg':24s}: A={mean(ac):.3f}±{stdev(ac):.3f}  "
          f"B={mean(bc):.3f}±{stdev(bc):.3f}  Δ(B-A)={mean(bc)-mean(ac):+.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="단일 입력 파일 채점", default=None)
    parser.add_argument(
        "--compare", nargs=2, metavar=("FILE_A", "FILE_B"),
        help="두 파일 채점 후 비교",
    )
    parser.add_argument("--reps", type=int, default=3,
                        help="샘플 × 차원당 반복 횟수 (기획서 기본 3)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Judge 호출 temperature (기획서 0.3)")
    parser.add_argument("--resume", action="store_true",
                        help="기존 출력 파일에 이어서 채점")
    parser.add_argument("--sleep-on-error", type=float, default=2.0,
                        help="API 오류 시 재시도 전 대기 (초)")
    args = parser.parse_args()

    if not args.input and not args.compare:
        parser.error("--input 또는 --compare 를 지정하세요.")

    print("KorMedMCQA test 매핑 로딩...")
    sample_map = load_test_sample_map()
    print(f"매핑 {len(sample_map)}건 로드.")

    client = make_client()
    print(f"Judge: {SOLAR_MODEL} @ {SOLAR_BASE_URL}")
    print(f"설정: reps={args.reps}, temperature={args.temperature}, "
          f"dimensions={len(DIM_ORDER)}\n")

    prefix = "geval_"

    if args.input:
        out = f"{prefix}{os.path.basename(args.input)}"
        stats = score_file(args.input, out, sample_map, client,
                           args.reps, args.temperature, args.resume,
                           args.sleep_on_error)
        print_summary(args.input, stats)
        print(f"\n저장: {out}")

    if args.compare:
        a, b = args.compare
        out_a = f"{prefix}{os.path.basename(a)}"
        out_b = f"{prefix}{os.path.basename(b)}"
        stats_a = score_file(a, out_a, sample_map, client,
                             args.reps, args.temperature, args.resume,
                             args.sleep_on_error)
        stats_b = score_file(b, out_b, sample_map, client,
                             args.reps, args.temperature, args.resume,
                             args.sleep_on_error)
        print_summary(a, stats_a)
        print_summary(b, stats_b)
        compare(stats_a, a, stats_b, b)
        print(f"\n저장: {out_a}, {out_b}")


if __name__ == "__main__":
    main()
