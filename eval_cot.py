"""
KorMedMCQA 평가 - 동일한 형식 강제 CoT 프롬프트로 GPT-4 / GPT-4o 비교.

KorMedMCQA fewshot split의 전문가 CoT(차근차근 생각해봅시다 → 상황 정리 →
의학 지식 → 결론 → 정답) 구조를 명시적인 [Step 1]~[Step 4] + "정답: X"
4단계 라벨로 변환해 형식을 강제한다. GPT-5 평가 때 발생했던
"단계별 추론을 제공할 수 없습니다" 같은 형식 거부를 막기 위한 것이다.

사용:
    python eval_cot.py --model gpt-4 --total 100
    python eval_cot.py --model gpt-4o --total 100
"""

import argparse
import json
import re

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


SUBSETS = ["dentist", "doctor", "nurse", "pharm"]

PRICING = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
}


PROMPT_TEMPLATE = """다음은 한국 의료 자격시험(KorMedMCQA) 객관식 문제입니다.
당신은 의사·치과의사·약사·간호사 수준의 의료 전문가입니다.
아래 [출력 형식]을 반드시 그대로 따라 한국어로 답하세요.

[문제]
{question}

[선택지]
A. {A}
B. {B}
C. {C}
D. {D}
E. {E}

[출력 형식 — 반드시 아래 4개 Step + 마지막 줄 형태로 출력]
[Step 1] 환자/문제 상황 정리: (문제에서 주어진 핵심 임상 정보·조건을 한국어로 요약)
[Step 2] 관련 의학적 원리·지식: (이 문제에 적용되는 해부·생리·병태생리·약리·진단·치료 원리를 서술)
[Step 3] 선택지별 검토: A부터 E까지 각 선택지를 한 줄씩이라도 검토하여 적합/부적합 사유를 명시
[Step 4] 결론 도출: 위 근거를 종합하여 가장 적절한 답을 고른 이유를 한 문단으로 정리
정답: X

[제약 — 위반 금지]
1. 위 4개의 Step 라벨([Step 1], [Step 2], [Step 3], [Step 4])과 "정답: X" 라인을 모두 빠짐없이 출력합니다.
2. "단계별 추론을 제공할 수 없다", "답할 수 없다", "간단한 근거" 등 형식 거부·축약 응답은 금지입니다. 불확실해도 가장 가능성이 높은 선택지를 골라 위 형식대로 작성하세요.
3. 마지막 줄은 정확히 "정답: X" (X는 A, B, C, D, E 중 하나)여야 합니다. 괄호·따옴표·마침표·추가 텍스트를 붙이지 마세요.
4. 모든 추론은 한국어로 작성합니다.
"""


def build_prompt(sample):
    return PROMPT_TEMPLATE.format(
        question=sample["question"],
        A=sample["A"],
        B=sample["B"],
        C=sample["C"],
        D=sample["D"],
        E=sample["E"],
    )


_ANSWER_LINE_RE = re.compile(r"정답\s*[::]\s*\(?\s*([A-E])\s*\)?")


def parse_answer(response_text):
    if not response_text:
        return None
    matches = _ANSWER_LINE_RE.findall(response_text)
    if matches:
        return matches[-1]
    last_chunk = response_text.strip()[-80:]
    for char in ["A", "B", "C", "D", "E"]:
        if char in last_chunk:
            return char
    return None


def check_format(response_text):
    """형식 강제가 잘 지켜졌는지 점검 (faithfulness 채점의 구조적 부분)."""
    if not response_text:
        return {
            "has_step1": False,
            "has_step2": False,
            "has_step3": False,
            "has_step4": False,
            "has_answer_line": False,
            "all_steps_present": False,
        }
    has_s1 = "[Step 1]" in response_text
    has_s2 = "[Step 2]" in response_text
    has_s3 = "[Step 3]" in response_text
    has_s4 = "[Step 4]" in response_text
    has_ans = bool(_ANSWER_LINE_RE.search(response_text))
    return {
        "has_step1": has_s1,
        "has_step2": has_s2,
        "has_step3": has_s3,
        "has_step4": has_s4,
        "has_answer_line": has_ans,
        "all_steps_present": all([has_s1, has_s2, has_s3, has_s4, has_ans]),
    }


def label_to_letter(label):
    return chr(ord("A") + int(label) - 1)


def evaluate_one(client, model, sample):
    prompt = build_prompt(sample)
    correct_letter = label_to_letter(sample["answer"])

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
        )
        response_text = response.choices[0].message.content
        predicted = parse_answer(response_text)
        fmt = check_format(response_text)

        return {
            "predicted": predicted,
            "correct": correct_letter,
            "is_correct": predicted == correct_letter,
            "reasoning": response_text,
            "format_check": fmt,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    except Exception as e:
        return {
            "predicted": None,
            "correct": correct_letter,
            "is_correct": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "input_tokens": 0,
            "output_tokens": 0,
        }


def samples_per_subset(total_samples, subsets):
    base = total_samples // len(subsets)
    remainder = total_samples % len(subsets)
    return {
        subset: base + (1 if i < remainder else 0)
        for i, subset in enumerate(subsets)
    }


def load_samples(total_samples):
    requested = samples_per_subset(total_samples, SUBSETS)
    all_samples = []

    print("데이터 로딩 중...")
    for subset in SUBSETS:
        ds = load_dataset("sean0042/KorMedMCQA", name=subset, split="test")
        count = min(requested[subset], len(ds))
        selected = ds.select(range(count))
        for sample_idx, sample in enumerate(selected):
            all_samples.append(
                {"subset": subset, "sample_idx": sample_idx, "sample": sample}
            )
        print(f"- {subset}: {count}건")

    return all_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="예: gpt-4, gpt-4o")
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    model = args.model
    total = args.total
    output_file = args.output or f"results_{model}_cot_forced_{total}.jsonl"

    client = OpenAI()

    print(f"=== KorMedMCQA all fields - {model} 평가 (형식 강제 CoT) ===\n")

    samples = load_samples(total)
    print(f"\n총 {len(samples)}건 평가\n")

    correct_count = 0
    error_count = 0
    format_ok_count = 0
    total_input = 0
    total_output = 0
    per_subset = {s: {"total": 0, "correct": 0, "errors": 0} for s in SUBSETS}

    with open(output_file, "w", encoding="utf-8") as f:
        for global_idx, item in enumerate(tqdm(samples, desc="평가 중")):
            subset = item["subset"]
            sample = item["sample"]
            result = evaluate_one(client, model, sample)
            result["global_idx"] = global_idx
            result["subset"] = subset
            result["sample_idx"] = item["sample_idx"]
            result["question"] = sample["question"][:100] + "..."
            result["model"] = model

            per_subset[subset]["total"] += 1
            if result.get("error"):
                error_count += 1
                per_subset[subset]["errors"] += 1
            else:
                if result["is_correct"]:
                    correct_count += 1
                    per_subset[subset]["correct"] += 1
                if result.get("format_check", {}).get("all_steps_present"):
                    format_ok_count += 1

            total_input += result.get("input_tokens", 0)
            total_output += result.get("output_tokens", 0)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    valid = len(samples) - error_count
    acc = (correct_count / valid * 100) if valid > 0 else 0
    fmt_rate = (format_ok_count / valid * 100) if valid > 0 else 0

    price = PRICING.get(model, {"input": 0, "output": 0})
    cost = (total_input * price["input"] + total_output * price["output"]) / 1_000_000

    print(f"\n=== {model} 결과 ===")
    print(f"정답률: {correct_count}/{valid} = {acc:.1f}%")
    print(f"형식 준수율 (4 step + 정답 라인 모두): {format_ok_count}/{valid} = {fmt_rate:.1f}%")
    if error_count:
        print(f"API 오류: {error_count}건")

    print("\n분야별 정답률:")
    for subset, stats in per_subset.items():
        v = stats["total"] - stats["errors"]
        a = (stats["correct"] / v * 100) if v else 0
        print(f"- {subset}: {stats['correct']}/{v} = {a:.1f}%")

    print(f"\n입력 토큰: {total_input:,}")
    print(f"출력 토큰: {total_output:,}")
    print(f"예상 비용: ${cost:.4f}")
    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
