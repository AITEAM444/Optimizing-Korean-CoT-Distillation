"""
Evaluate GPT-5 on KorMedMCQA across all fields.

The script samples about 100 total test questions across dentist, doctor,
nurse, and pharm, then stores the model's chain-of-thought style output.
"""

import json

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm


# === Settings ===
TOTAL_SAMPLES = 100
MODEL = "gpt-5"
SUBSETS = ["dentist", "doctor", "nurse", "pharm"]
OUTPUT_FILE = f"results_{MODEL}_all_fields_{TOTAL_SAMPLES}_cot.jsonl"

client = OpenAI()


def build_prompt(sample):
    """Build a KorMedMCQA prompt."""
    return f"""다음은 한국 의료 자격시험 객관식 문제입니다. 정답은 A, B, C, D, E 중 하나로만 답하세요.

[문제]
{sample['question']}

[선택지]
A. {sample['A']}
B. {sample['B']}
C. {sample['C']}
D. {sample['D']}
E. {sample['E']}

먼저 단계별로 추론하고, 마지막 줄에 "정답: X" 형식으로 답하세요. (X는 A~E 중 하나)
"""


def parse_answer(response_text):
    if not response_text:
        return None

    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        for char in ["A", "B", "C", "D", "E"]:
            if f"정답: {char}" in line or f"정답:{char}" in line:
                return char
            if f"답: {char}" in line or f"답:{char}" in line:
                return char

    last_chars = response_text.strip()[-50:]
    for char in ["A", "B", "C", "D", "E"]:
        if char in last_chars:
            return char
    return None


def label_to_letter(label):
    """KorMedMCQA: 1 -> A, 2 -> B, ..., 5 -> E."""
    return chr(ord("A") + int(label) - 1)


def evaluate_one(sample):
    prompt = build_prompt(sample)
    correct_letter = label_to_letter(sample["answer"])

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=12000,
            reasoning_effort="medium",
        )
        response_text = response.choices[0].message.content
        predicted = parse_answer(response_text)

        reasoning_tokens = 0
        if hasattr(response.usage, "completion_tokens_details"):
            details = response.usage.completion_tokens_details
            if details is not None:
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        return {
            "predicted": predicted,
            "correct": correct_letter,
            "is_correct": predicted == correct_letter,
            "reasoning": response_text,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
    except Exception as e:
        return {
            "predicted": None,
            "correct": correct_letter,
            "is_correct": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }


def samples_per_subset(total_samples, subsets):
    base = total_samples // len(subsets)
    remainder = total_samples % len(subsets)
    return {
        subset: base + (1 if i < remainder else 0)
        for i, subset in enumerate(subsets)
    }


def load_samples():
    requested_counts = samples_per_subset(TOTAL_SAMPLES, SUBSETS)
    all_samples = []

    print("데이터 로딩 중...")
    for subset in SUBSETS:
        ds = load_dataset("sean0042/KorMedMCQA", name=subset, split="test")
        count = min(requested_counts[subset], len(ds))
        selected = ds.select(range(count))

        for sample_idx, sample in enumerate(selected):
            all_samples.append(
                {
                    "subset": subset,
                    "sample_idx": sample_idx,
                    "sample": sample,
                }
            )

        print(f"- {subset}: {count}건")

    return all_samples


def main():
    print(f"=== KorMedMCQA all fields - {MODEL} 평가 ===\n")

    samples = load_samples()
    print(f"\n총 {len(samples)}건 평가\n")

    correct_count = 0
    error_count = 0
    total_input = 0
    total_output = 0
    per_subset = {
        subset: {"total": 0, "correct": 0, "errors": 0}
        for subset in SUBSETS
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for global_idx, item in enumerate(tqdm(samples, desc="평가 중")):
            subset = item["subset"]
            sample = item["sample"]
            result = evaluate_one(sample)
            result["global_idx"] = global_idx
            result["subset"] = subset
            result["sample_idx"] = item["sample_idx"]
            result["question"] = sample["question"][:100] + "..."

            per_subset[subset]["total"] += 1
            if result.get("error"):
                error_count += 1
                per_subset[subset]["errors"] += 1
            elif result["is_correct"]:
                correct_count += 1
                per_subset[subset]["correct"] += 1

            total_input += result.get("input_tokens", 0)
            total_output += result.get("output_tokens", 0)

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

    valid_count = len(samples) - error_count
    accuracy = (correct_count / valid_count * 100) if valid_count > 0 else 0
    cost = (total_input * 1.25 + total_output * 10.00) / 1_000_000

    print(f"\n=== {MODEL} 결과 (all fields) ===")
    print(f"정답률: {correct_count}/{valid_count} = {accuracy:.1f}%")
    if error_count > 0:
        print(f"오류: {error_count}건")

    print("\n분야별 결과:")
    for subset, stats in per_subset.items():
        subset_valid = stats["total"] - stats["errors"]
        subset_accuracy = (
            stats["correct"] / subset_valid * 100 if subset_valid > 0 else 0
        )
        print(
            f"- {subset}: {stats['correct']}/{subset_valid} = "
            f"{subset_accuracy:.1f}%"
        )

    print(f"\n입력 토큰: {total_input:,}")
    print(f"출력 토큰: {total_output:,}")
    print(f"예상 비용: ${cost:.4f}")
    print(f"\n결과 저장: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
