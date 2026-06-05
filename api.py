"""
GLiNER NER API — sliding-window long-text support.

Key design decision:
  Chunking uses text.split() word count for boundary calculation only.
  Each chunk passed to GLiNER is the RAW ORIGINAL substring — NOT a
  space-joined version of tokens.  This ensures GLiNER receives natural
  text and its returned char offsets are directly correct relative to
  the chunk, requiring only a simple c_start addition for global offsets.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from gliner import GLiNER
import uvicorn
import os
import json
from datetime import datetime
import re

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = os.getenv("GLINER_MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "checkpoint-1400"))

# Số words (text.split()) tối đa mỗi chunk.
# Training dùng 384 tokenize_text() tokens (gồm cả dấu câu tách riêng),
# nên tương đương ~300-320 space-split words. Dùng 300 cho an toàn.
MAX_LEN = 300
OVERLAP  = 50   # ~1/6 chunk, đủ để không mất entity bị cắt biên


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GLiNER NER API",
    description="NER API with sliding-window support for long job descriptions",
    version="3.0.0",
)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# S3 Model Auto-Downloader
# ---------------------------------------------------------------------------
# If MinIO is configured in ENV, download the model first
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
GLINER_MINIO_PREFIX = os.getenv("GLINER_MINIO_PREFIX", "")

if MINIO_BUCKET:
    from s3_downloader import download_model_from_minio
    # Local path where we cache the downloaded model from MinIO
    local_dir_name = GLINER_MINIO_PREFIX.replace('/', '_') if GLINER_MINIO_PREFIX else MINIO_BUCKET
    local_cache_path = os.path.join(os.path.dirname(__file__), "models", local_dir_name)
    success = download_model_from_minio(MINIO_BUCKET, GLINER_MINIO_PREFIX, local_cache_path)
    if success:
        MODEL_PATH = local_cache_path

try:
    print(f"Loading model from {MODEL_PATH}...")
    model = GLiNER.from_pretrained(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None


# ---------------------------------------------------------------------------
# Feedback Memory Cache (Continuous Learning Simulation)
# ---------------------------------------------------------------------------
FEEDBACK_PATH = os.path.join(os.path.dirname(__file__), "data", "feedback.jsonl")
MEMORY_CACHE: Dict[str, str] = {}  # text.lower() -> label.upper()

def load_memory_cache():
    global MEMORY_CACHE
    if not os.path.isfile(FEEDBACK_PATH):
        return
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                for ent in rec.get("entities", []):
                    text_lower = ent.get("text", "").lower().strip()
                    label = ent.get("label", "").upper().strip()
                    if text_lower and label:
                        MEMORY_CACHE[text_lower] = label
    except Exception as e:
        print(f"Error loading memory cache: {e}")

load_memory_cache()

def apply_memory_cache(text: str, entities: List[Dict]) -> List[Dict]:
    """Ap dụng các từ đã được đánh nhãn trước đó (từ FEEDBACK) vào text nếu GLiNER bỏ sót."""
    intervals = [(ent["start"], ent["end"]) for ent in entities]
    intervals.sort()

    def is_overlap(s, e):
        for (i_s, i_e) in intervals:
            if max(s, i_s) < min(e, i_e):
                return True
        return False

    text_lower = text.lower()
    new_entities = list(entities)

    # Sort cache keys by length descending to match longest phrases first
    for phrase, label in sorted(MEMORY_CACHE.items(), key=lambda x: len(x[0]), reverse=True):
        idx = 0
        while True:
            idx = text_lower.find(phrase, idx)
            if idx == -1:
                break
            
            end_idx = idx + len(phrase)
            
            # Boundary check to avoid matching substrings of words
            left_ok = (idx == 0) or not text_lower[idx-1].isalnum()
            right_ok = (end_idx == len(text_lower)) or not text_lower[end_idx].isalnum()
            
            if left_ok and right_ok and not is_overlap(idx, end_idx):
                new_entities.append({
                    "text": text[idx:end_idx],
                    "label": label,
                    "start": idx,
                    "end": end_idx,
                    "score": 1.0,  # Hoàn toàn tự tin vì là human feedback
                    "is_manual": True
                })
                intervals.append((idx, end_idx))
                intervals.sort()
            idx = end_idx

    return sorted(new_entities, key=lambda x: x["start"])


# ---------------------------------------------------------------------------
# Sliding-window NER inference
#
# Tại sao KHÔNG dùng regex tokenizer để build chunk_text?
#   tokenize_text() loại bỏ \n, -, +, /, •, *, # ... → chunk_text bị
#   biến đổi khác original → GLiNER trả offset của chuỗi đã biến đổi →
#   map ngược về original sai → cắt giữa chữ → "brid Machine Lea", "g check"
#
# Giải pháp: dùng text.split() chỉ để ĐẾM WORD và xác định BIÊN CHUNK,
# rồi pass text[c_start:c_end] (original substring) vào GLiNER.
# GLiNER trả offset relative to chunk → cộng c_start = global offset đúng.
# ---------------------------------------------------------------------------
def _build_word_char_starts(text: str) -> List[int]:
    """
    Trả về char_start[i] = byte position của words[i] (text.split()) trong text.
    Dùng để xác định biên chunk theo word index.
    """
    positions: List[int] = []
    pos = 0
    for word in text.split():
        idx = text.find(word, pos)
        positions.append(idx)
        pos = idx + len(word)
    return positions


def predict_long_text(
    text: str,
    labels: List[str],
    threshold: float = 0.3,
    max_len: int = MAX_LEN,
    overlap: int = OVERLAP,
) -> List[Dict]:
    """
    Sliding-window NER inference.

    Luồng:
      1. Đếm words (text.split()), lấy char position của mỗi word.
      2. Nếu text ngắn (<= max_len words) → predict thẳng.
      3. Nếu dài → chia thành chunks theo word boundary.
         - Mỗi chunk = text[c_start : c_end] → original text, không biến đổi.
         - GLiNER trả (start, end) relative to chunk_text.
         - Global offset = c_start + pred["start"].
      4. Dedup (global_start, global_end, label) → giữ score cao nhất.
      5. Sắp xếp theo char position.
    """
    words   = text.split()
    n_words = len(words)

    # Văn bản ngắn → predict thẳng, không cần chunking
    if n_words <= max_len:
        return model.predict_entities(text, labels, threshold=threshold)

    word_char_starts = _build_word_char_starts(text)

    # key: (global_char_start, global_char_end, label) → entity
    best: Dict[tuple, Dict] = {}

    start_w = 0
    while start_w < n_words:
        end_w = min(start_w + max_len, n_words)

        # Lấy ORIGINAL TEXT SLICE — không join lại, không biến đổi
        c_start = word_char_starts[start_w]
        c_end   = word_char_starts[end_w] if end_w < n_words else len(text)
        chunk   = text[c_start:c_end]

        preds = model.predict_entities(chunk, labels, threshold=threshold)

        for pred in preds:
            # Offset relative to chunk → global offset trong original text
            gs    = c_start + pred["start"]
            ge    = c_start + pred["end"]
            key   = (gs, ge, pred["label"])
            score = float(pred.get("score", 1.0))

            if key not in best or best[key]["score"] < score:
                best[key] = {
                    "text":  text[gs:ge],   # lấy text từ original, không từ pred
                    "label": pred["label"],
                    "start": gs,
                    "end":   ge,
                    "score": score,
                }

        if end_w == n_words:
            break
        start_w += max_len - overlap

    return sorted(best.values(), key=lambda x: x["start"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., description="Job description text")
    labels: List[str] = Field(
        default=["skill", "major", "experience"],
        description="Entity labels to extract",
    )
    threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Confidence threshold. Lower = more entities (possibly noisier).",
    )


class EntityResponse(BaseModel):
    text: str
    label: str
    start: int   # char offset (byte) trong original text
    end: int     # char offset (byte) trong original text, exclusive
    score: float
    is_manual: bool = False


class PredictResponse(BaseModel):
    entities: List[EntityResponse]


# ---------------------------------------------------------------------------
# Feedback schemas (active learning)
# ---------------------------------------------------------------------------
class FeedbackEntity(BaseModel):
    text: str = Field(..., description="Entity text (verbatim from original)")
    label: str = Field(..., description="SKILL | MAJOR | EXPERIENCE")
    start: int = Field(..., description="Char start in original text")
    end: int   = Field(..., description="Char end in original text")


class FeedbackRequest(BaseModel):
    text: str                    = Field(..., description="Full original text")
    entities: List[FeedbackEntity] = Field(..., description="Corrected entity list (human-reviewed)")
    note: Optional[str]          = Field(None, description="Optional reviewer note")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "sliding_window": {"max_len_words": MAX_LEN, "overlap_words": OVERLAP},
        "chunk_method": "original text slice (no tokenizer transform)",
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Extract NER entities. Supports long text via sliding window.
    Offsets (start/end) are char positions in the original text.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    try:
        entities = predict_long_text(
            text=request.text,
            labels=request.labels,
            threshold=request.threshold,
        )

        # Apply continuous learning memory
        entities = apply_memory_cache(request.text, entities)

        return PredictResponse(
            entities=[
                EntityResponse(
                    text=ent["text"],
                    label=ent["label"],
                    start=ent["start"],
                    end=ent["end"],
                    score=ent["score"],
                    is_manual=ent.get("is_manual", False),
                )
                for ent in entities
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """
    Lưu corrections do human review.
    Dùng để xây dựng thêm training data (active learning).
    File: data/feedback.jsonl — mỗi dòng là 1 JSON record.
    """
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "text":      req.text,
        "entities":  [e.model_dump() for e in req.entities],
        "note":      req.note,
    }

    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Update in-memory cache
    for e in req.entities:
        text_lower = e.text.lower().strip()
        label = e.label.upper().strip()
        if text_lower and label:
            MEMORY_CACHE[text_lower] = label

    return {
        "status":          "saved",
        "entities_saved":  len(req.entities),
        "feedback_file":   FEEDBACK_PATH,
    }


@app.get("/feedback/stats")
def feedback_stats():
    """Thống kê số lượng feedback đã lưu."""
    if not os.path.isfile(FEEDBACK_PATH):
        return {"total_records": 0, "total_entities": 0}

    total_records  = 0
    total_entities = 0
    label_counts: Dict[str, int] = {}

    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                total_records  += 1
                total_entities += len(rec.get("entities", []))
                for ent in rec.get("entities", []):
                    lbl = ent.get("label", "UNKNOWN").upper()
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            except json.JSONDecodeError:
                continue

    return {
        "total_records":  total_records,
        "total_entities": total_entities,
        "by_label":       label_counts,
        "feedback_file":  FEEDBACK_PATH,
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7777, reload=True)
