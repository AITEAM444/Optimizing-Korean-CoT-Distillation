"""
Reasoning faithfulness 채점 - Solar Pro 3 LLM-as-judge.

eval_cot.py가 만든 results_*.jsonl을 입력으로 받아 각 응답의 단계별 추론과
최종 정답이 얼마나 일관·충실한지 1~5점으로 채점한다. 결과는
faithfulness_*.jsonl 로 저장되고, 마지막에 모델 간 비교 요약을 출력한다.

faithfulness 의미:
    추론(Step 1~4)이 (a) 문제 상황을 정확히 반영하고
    (b) 의학적 근거가 결론과 모순되지 않으며
    (c) 마지막 "정답: X"가 Step 4의 결론과 일치하는 정도.

판단은 모델의 답이 "정답인가"와는 독립이다. (틀린 답이라도 추론이 일관되면
faithfulness 는 높을 수 있고, 맞은 답이라도 추론이 결론을 지지하지 못하면
낮을 수 있다.)

사용:
    # Solar Pro 3 (기본)
    python score_faithfulness.py --compare A.jsonl B.jsonl

    # GPT-5 로 교차 검증
    python score_faithfulness.py --judge gpt5 --compare A.jsonl B.jsonl

출력 파일은 judge 이름을 prefix 로 붙여 충돌을 방지한다.
"""

import argparse
import json
import os
import re
import sys

from openai import OpenAI
from tqdm import tqdm


# Upstage Solar Pro 3 (OpenAI-호환). 환경변수 UPSTAGE_API_KEY 사용.
SOLAR_BASE_URL = "https://api.upstage.ai/v1"
SOLAR_MODEL = os.environ.get("SOLAR_MODEL", "solar-pro3")  # 실제 ID가 다르면 환경변수로 덮어쓰기

# OpenAI GPT-5 judge 설정. 환경변수 OPENAI_API_KEY 사용.
GPT5_MODEL = os.environ.get("GPT5_MODEL", "gpt-5")
GPT5_REASONING_EFFORT = os.environ.get("GPT5_REASONING_EFFORT", "low")


JUDGE_PROMPT = """당신은 의학 문제 풀이의 추론 충실도(reasoning faithfulness)를 평가하는 채점관입니다.
아래 모델이 한국 의료 자격시험 문제에 대해 4단계 CoT([Step 1]~[Step 4])와 "정답: X"로 답했습니다.

당신의 임무는 **모델이 정답을 맞췄는지가 아니라, 추론이 결론과 얼마나 충실히 연결되는지**를 평가하는 것입니다.

[원본 문제]
{question}

[선택지]
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

[실제 정답] {correct}
[모델 선택] {predicted}

[모델 응답 전체]
{reasoning}

다음 5개 항목을 평가하고 마지막에 JSON만 출력하세요.

1. step_completeness (0 또는 1): 모델이 [Step 1], [Step 2], [Step 3], [Step 4], "정답: X" 라인을 빠짐없이 출력했는가?
2. situation_grounding (1~5): Step 1이 문제의 핵심 임상 정보·조건을 정확히 반영하는가? (5=완전 반영, 1=문제와 무관)
3. knowledge_relevance (1~5): Step 2의 의학 지식이 이 문제와 직접 관련되어 있고 사실에 부합하는가? (5=정확·관련, 1=무관·오류)
4. option_grounding (1~5): Step 3에서 실제 선택지(A~E)의 내용을 다루며 적합/부적합 판단을 했는가? (5=각 선택지 명시적 검토, 1=선택지 무시)
5. conclusion_consistency (1~5): Step 4의 결론과 마지막 "정답: X" 라인이 일치하며, Step 1~3의 추론이 그 결론을 지지하는가? (5=완전 일치·지지, 1=결론이 추론과 모순)
6. overall_faithfulness (1~5): 종합 충실도. 위 항목들과 사고 흐름의 일관성을 종합하여 매기되, "정답 일치 여부"는 고려하지 마세요.

반드시 아래 JSON 한 줄만 출력하세요. 다른 텍스트·코드블록 금지.

{{"step_completeness": 0|1, "situation_grounding": 1-5, "knowledge_relevance": 1-5, "option_grounding": 1-5, "conclusion_consistency": 1-5, "overall_faithfulness": 1-5, "brief_reason": "한 줄 한국어 사유"}}
"""


def make_judge_client(backend):
    if backend == "solar":
        api_key = os.environ.get("UPSTAGE_API_KEY")
        if not api_key:
            print("ERROR: UPSTAGE_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
            sys.exit(1)
        return OpenAI(api_key=api_key, base_url=SOLAR_BASE_URL)
    if backend == "gpt5":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY 환경변수가 필요합니다.", file=sys.stderr)
            sys.exit(1)
        return OpenAI()
    raise ValueError(f"unknown backend: {backend}")


def judge_label(backend):
    if backend == "solar":
        return f"{SOLAR_MODEL} @ {SOLAR_BASE_URL}"
    if backend == "gpt5":
        return f"{GPT5_MODEL} (reasoning_effort={GPT5_REASONING_EFFORT})"
    return backend


def load_test_sample_map():
    """원본 문제/선택지를 채점 프롬프트에 다시 주입하기 위해 KorMedMCQA test를 로드."""
    from datasets import load_dataset

    mapping = {}
    for subset in ["dentist", "doctor", "nurse", "pharm"]:
        ds = load_dataset("sean0042/KorMedMCQA", name=subset, split="test")
        for i, row in enumerate(ds):
            mapping[(subset, i)] = row
    return mapping


JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_judge_response(text):
    if not text:
        return None
    text = text.strip()
    # ```json 블록 떼기
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_one(client, backend, sample_row, result):
    prompt = JUDGE_PROMPT.format(
        question=sample_row["question"],
        A=sample_row["A"],
        B=sample_row["B"],
        C=sample_row["C"],
        D=sample_row["D"],
        E=sample_row["E"],
        correct=result.get("correct"),
        predicted=result.get("predicted"),
        reasoning=result.get("reasoning", "")[:8000],
    )
    try:
        if backend == "solar":
            response = client.chat.completions.create(
                model=SOLAR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600,
            )
        elif backend == "gpt5":
            response = client.chat.completions.create(
                model=GPT5_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4000,
                reasoning_effort=GPT5_REASONING_EFFORT,
            )
        else:
            raise ValueError(f"unknown backend: {backend}")
        text = response.choices[0].message.content
        parsed = parse_judge_response(text)
        return {"judge_raw": text, "judge": parsed}
    except Exception as e:
        return {"judge_raw": None, "judge": None, "judge_error": f"{type(e).__name__}: {e}"}


def score_file(input_path, output_path, sample_map, client, backend):
    n = 0
    aggregated = {
        "step_completeness": [],
        "situation_grounding": [],
        "knowledge_relevance": [],
        "option_grounding": [],
        "conclusion_consistency": [],
        "overall_faithfulness": [],
    }
    parse_failed = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        lines = fin.readlines()
        for line in tqdm(lines, desc=f"채점 {os.path.basename(input_path)}"):
            n += 1
            result = json.loads(line)
            if result.get("error") or not result.get("reasoning"):
                fout.write(json.dumps({**result, "judge": None, "judge_skip": "no_reasoning"}, ensure_ascii=False) + "\n")
                continue
            key = (result.get("subset"), result.get("sample_idx"))
            sample_row = sample_map.get(key)
            if sample_row is None:
                fout.write(json.dumps({**result, "judge": None, "judge_skip": "no_sample"}, ensure_ascii=False) + "\n")
                continue
            j = judge_one(client, backend, sample_row, result)
            merged = {**result, **j}
            fout.write(json.dumps(merged, ensure_ascii=False) + "\n")
            fout.flush()

            parsed = j.get("judge")
            if not parsed:
                parse_failed += 1
                continue
            for k in aggregated:
                v = parsed.get(k)
                if isinstance(v, (int, float)):
                    aggregated[k].append(v)

    return {"n": n, "parse_failed": parse_failed, "aggregated": aggregated}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def print_summary(label, stats):
    agg = stats["aggregated"]
    print(f"\n=== {label} (n={stats['n']}, judge_parse_fail={stats['parse_failed']}) ===")
    for k in [
        "step_completeness",
        "situation_grounding",
        "knowledge_relevance",
        "option_grounding",
        "conclusion_consistency",
        "overall_faithfulness",
    ]:
        xs = agg[k]
        print(f"  {k:24s}: mean={mean(xs):.3f}  (n={len(xs)})")


def compare(stats_a, label_a, stats_b, label_b):
    print(f"\n=== 비교: {label_a} vs {label_b} ===")
    for k in [
        "step_completeness",
        "situation_grounding",
        "knowledge_relevance",
        "option_grounding",
        "conclusion_consistency",
        "overall_faithfulness",
    ]:
        a = mean(stats_a["aggregated"][k])
        b = mean(stats_b["aggregated"][k])
        diff = a - b
        arrow = "↑" if diff > 0 else ("↓" if diff < 0 else "=")
        print(f"  {k:24s}: {label_a}={a:.3f}  {label_b}={b:.3f}  Δ={diff:+.3f} {arrow}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="단일 입력 파일 채점", default=None)
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FILE_A", "FILE_B"),
        help="두 파일 채점 후 비교",
    )
    parser.add_argument(
        "--judge",
        choices=["solar", "gpt5"],
        default="solar",
        help="채점에 쓸 LLM judge 백엔드 (기본: solar)",
    )
    args = parser.parse_args()

    if not args.input and not args.compare:
        parser.error("--input 또는 --compare 를 지정하세요.")

    print("KorMedMCQA test 매핑 로딩...")
    sample_map = load_test_sample_map()
    print(f"매핑 {len(sample_map)}건 로드.")

    client = make_judge_client(args.judge)
    print(f"Judge ({args.judge}): {judge_label(args.judge)}\n")

    prefix = f"faithfulness_{args.judge}_"

    if args.input:
        out = f"{prefix}{os.path.basename(args.input)}"
        stats = score_file(args.input, out, sample_map, client, args.judge)
        print_summary(args.input, stats)
        print(f"\n저장: {out}")

    if args.compare:
        a, b = args.compare
        out_a = f"{prefix}{os.path.basename(a)}"
        out_b = f"{prefix}{os.path.basename(b)}"
        stats_a = score_file(a, out_a, sample_map, client, args.judge)
        stats_b = score_file(b, out_b, sample_map, client, args.judge)
        print_summary(a, stats_a)
        print_summary(b, stats_b)
        compare(stats_a, a, stats_b, b)
        print(f"\n저장: {out_a}, {out_b}")


if __name__ == "__main__":
    main()
