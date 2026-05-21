"""
eval_cot.py 가 만든 results_*.jsonl 안의 error 행만 다시 호출해 같은 자리에
덮어쓰는 보강 스크립트.

eval_cot.py 의 build_prompt / parse_answer / check_format / label_to_letter 를
재사용하므로 프롬프트가 100% 동일하다.

사용:
    python retry_errors.py --input results_gpt-4_cot_forced_100.jsonl --model gpt-4
"""

import argparse
import json
import os
import shutil

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from eval_cot import (
    PRICING,
    SUBSETS,
    build_prompt,
    check_format,
    label_to_letter,
    parse_answer,
)


def load_test_sample_map():
    mapping = {}
    for subset in SUBSETS:
        ds = load_dataset("sean0042/KorMedMCQA", name=subset, split="test")
        for i, row in enumerate(ds):
            mapping[(subset, i)] = row
    return mapping


def retry_one(client, model, sample):
    prompt = build_prompt(sample)
    correct_letter = label_to_letter(sample["answer"])
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        predicted = parse_answer(text)
        return {
            "predicted": predicted,
            "correct": correct_letter,
            "is_correct": predicted == correct_letter,
            "reasoning": text,
            "format_check": check_format(text),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="기존 결과 jsonl")
    parser.add_argument("--model", required=True, help="재호출에 쓸 모델 (원래와 동일하게)")
    args = parser.parse_args()

    in_path = args.input
    bak_path = in_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(in_path, bak_path)
        print(f"백업 생성: {bak_path}")

    with open(in_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    error_indices = [i for i, r in enumerate(rows) if r.get("error")]
    print(f"오류 {len(error_indices)}건 발견 / 총 {len(rows)}건")
    if not error_indices:
        print("재실행할 행이 없습니다.")
        return

    print("KorMedMCQA test 매핑 로딩...")
    sample_map = load_test_sample_map()

    client = OpenAI()

    retried_ok = 0
    still_failed = 0
    added_input = 0
    added_output = 0

    for i in tqdm(error_indices, desc=f"재실행 ({args.model})"):
        r = rows[i]
        key = (r.get("subset"), r.get("sample_idx"))
        sample = sample_map.get(key)
        if sample is None:
            print(f"  skip: 매핑 없음 idx={i} subset={r.get('subset')} sample_idx={r.get('sample_idx')}")
            continue

        new_result = retry_one(client, args.model, sample)

        # 메타 보존 (global_idx, subset, sample_idx, question, model)
        meta_keys = ["global_idx", "subset", "sample_idx", "question", "model"]
        merged = {k: r[k] for k in meta_keys if k in r}
        merged.update(new_result)
        rows[i] = merged

        if new_result.get("error"):
            still_failed += 1
        else:
            retried_ok += 1
            added_input += new_result.get("input_tokens", 0)
            added_output += new_result.get("output_tokens", 0)

    with open(in_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    valid_total = sum(1 for r in rows if not r.get("error"))
    correct_total = sum(1 for r in rows if r.get("is_correct"))
    fmt_total = sum(
        1 for r in rows if r.get("format_check", {}).get("all_steps_present")
    )

    price = PRICING.get(args.model, {"input": 0, "output": 0})
    added_cost = (added_input * price["input"] + added_output * price["output"]) / 1_000_000

    print(f"\n=== 재실행 결과 ===")
    print(f"성공: {retried_ok}건  /  여전히 실패: {still_failed}건")
    print(f"추가 비용 (재호출 토큰): ${added_cost:.4f}")
    print(f"\n=== 갱신 후 전체 통계 ({in_path}) ===")
    print(f"유효 응답: {valid_total}/{len(rows)}")
    if valid_total:
        print(f"정답률: {correct_total}/{valid_total} = {correct_total/valid_total*100:.1f}%")
        print(f"형식 준수율: {fmt_total}/{valid_total} = {fmt_total/valid_total*100:.1f}%")


if __name__ == "__main__":
    main()
