"""
merge_feedback.py — Gộp human feedback vào train_dataset.json để re-train.

Cách dùng:
    python src/merge_feedback.py

Output: data/train_dataset_v2.json (dataset gốc + feedback đã convert)

Sau khi chạy xong:
    1. Upload train_dataset_v2.json lên Kaggle Dataset (thay train_dataset.json cũ)
    2. Chạy lại finetune-gliner-kaggle.ipynb (load_best_model_at_end=True)
    3. Download checkpoint mới về models/
"""

import os
import re
import json
from collections import Counter

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

FEEDBACK_FILE   = os.path.join(_PROJECT_ROOT, "data", "feedback.jsonl")
TRAIN_FILE      = os.path.join(_PROJECT_ROOT, "data", "train_dataset.json")
OUTPUT_FILE     = os.path.join(_PROJECT_ROOT, "data", "train_dataset_v2.json")

VALID_LABELS = {"SKILL", "MAJOR", "EXPERIENCE"}

# Số lần duplicate mỗi feedback sample (để tăng weight của human corrections)
FEEDBACK_REPEAT = 3


# ---------------------------------------------------------------------------
# Tokenizer — giống hệt build_dataset_v2.py
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[\w']+|[.,!?;()&]")

def tokenize_text(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def find_token_indices(full_tokens: list[str], ent_tokens: list[str]) -> tuple[int, int]:
    n = len(ent_tokens)
    for i in range(len(full_tokens) - n + 1):
        if full_tokens[i:i + n] == ent_tokens:
            return i, i + n - 1
    return -1, -1


# ---------------------------------------------------------------------------
# Convert 1 feedback record → GLiNER training sample
# ---------------------------------------------------------------------------
def feedback_to_training(record: dict) -> dict | None:
    text     = record.get("text", "").strip()
    entities = record.get("entities", [])

    if not text or not entities:
        return None

    tokens = tokenize_text(text)
    if not tokens:
        return None

    ner: list[list] = []
    skipped = []

    for ent in entities:
        ent_text = ent.get("text", "").strip()
        label    = ent.get("label", "").upper()

        if label not in VALID_LABELS or not ent_text:
            continue

        ent_tokens = tokenize_text(ent_text)
        if not ent_tokens:
            continue

        start_idx, end_idx = find_token_indices(tokens, ent_tokens)
        if start_idx != -1:
            ner.append([start_idx, end_idx, label])
        else:
            skipped.append(ent_text)

    if skipped:
        print(f"  ⚠️  Không map được token: {skipped}")

    # Trả về None nếu không có entity nào map được
    if not ner:
        return None

    return {"tokenized_text": tokens, "ner": ner}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  GLiNER Feedback Merger")
    print("=" * 60)

    # ── Đọc feedback ──────────────────────────────────────────────
    if not os.path.isfile(FEEDBACK_FILE):
        print(f"❌ Không tìm thấy feedback file: {FEEDBACK_FILE}")
        print("   → Hãy dùng test_api.html để submit ít nhất 1 correction.")
        return

    feedback_records = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                feedback_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ⚠️  Dòng {i} parse lỗi: {e}")

    print(f"\n📥 Feedback records: {len(feedback_records)}")

    if not feedback_records:
        print("❌ Không có feedback record nào hợp lệ.")
        return

    # ── Convert feedback → training samples ───────────────────────
    converted = []
    label_counter: Counter = Counter()

    for rec in feedback_records:
        sample = feedback_to_training(rec)
        if sample is None:
            print(f"  ⚠️  Skip record (không map được entity nào): {rec.get('text','')[:60]}...")
            continue

        # Duplicate FEEDBACK_REPEAT lần để tăng weight
        for _ in range(FEEDBACK_REPEAT):
            converted.append(sample)

        for ner_item in sample["ner"]:
            label_counter[ner_item[2]] += 1

    print(f"\n✅ Converted: {len(feedback_records)} records → {len(converted)} samples "
          f"(×{FEEDBACK_REPEAT} repeat)")
    print(f"   Label distribution: {dict(label_counter)}")

    if not converted:
        print("❌ Không có sample nào được convert thành công.")
        return

    # ── Load existing training dataset ────────────────────────────
    if os.path.isfile(TRAIN_FILE):
        print(f"\n📂 Loading existing training data: {TRAIN_FILE}")
        with open(TRAIN_FILE, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        print(f"   Existing samples: {len(original_data):,}")
    else:
        print(f"\n⚠️  Không tìm thấy {TRAIN_FILE} → chỉ dùng feedback data")
        original_data = []

    # ── Merge & save ──────────────────────────────────────────────
    merged = original_data + converted
    print(f"\n📊 Merged dataset: {len(original_data):,} + {len(converted)} = {len(merged):,} samples")

    # Thống kê label trong dataset gộp
    all_labels: Counter = Counter()
    for item in merged:
        for ner_item in item["ner"]:
            all_labels[ner_item[2]] += 1
    print(f"   Total entities: {dict(all_labels)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n✅ XONG! Dataset mới được lưu tại:")
    print(f"   {OUTPUT_FILE}")
    print()
    print("📋 Bước tiếp theo:")
    print("   1. Upload train_dataset_v2.json lên Kaggle Dataset")
    print("      (đổi tên thành train_dataset.json hoặc sửa path trong notebook)")
    print("   2. Chạy lại finetune-gliner-kaggle.ipynb")
    print("   3. Download checkpoint mới về gliner-finetune/models/")
    print("   4. Restart api.py")


if __name__ == "__main__":
    main()
