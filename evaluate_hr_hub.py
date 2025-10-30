import os
import re
import json
import math
import argparse
import datetime as dt
from typing import List, Dict, Optional, Tuple

import pdfplumber
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import importlib.util


def load_app_module(app_path: str):
    spec = importlib.util.spec_from_file_location("hr_hub_app", app_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {app_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def _clean_cell(value: Optional[str]) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def extract_qas_from_pdf(pdf_path: str) -> List[Dict[str, str]]:
    qas: List[Dict[str, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []
            for tbl in tables:
                if not tbl or not any(tbl):
                    continue
                # Find header row
                header_idx = -1
                header_map: Dict[str, int] = {}
                for i, row in enumerate(tbl):
                    if not row:
                        continue
                    cells = [_clean_cell(c) for c in row]
                    joined = " ".join(cells).lower()
                    if "question" in joined and ("ideal answer" in joined or "ideal" in joined):
                        # Build column index map
                        for j, c in enumerate(cells):
                            cl = c.lower()
                            if "question" == cl:
                                header_map["question"] = j
                            if "ideal answer" == cl or cl.startswith("ideal"):
                                header_map["ideal_answer"] = j
                        if "question" in header_map and "ideal_answer" in header_map:
                            header_idx = i
                            break
                if header_idx == -1:
                    # Try positional assumption: 1st col = Question, 2nd col = Ideal Answer
                    header_map = {"question": 0, "ideal_answer": 1}
                    start_idx = 0
                else:
                    start_idx = header_idx + 1

                for r in range(start_idx, len(tbl)):
                    row = tbl[r]
                    if not row:
                        continue
                    cells = [_clean_cell(c) for c in row]
                    q = cells[header_map.get("question", 0)] if len(cells) > header_map.get("question", 0) else ""
                    a = cells[header_map.get("ideal_answer", 1)] if len(cells) > header_map.get("ideal_answer", 1) else ""
                    q, a = _clean_cell(q), _clean_cell(a)
                    if q:
                        qas.append({"question": q, "ideal_answer": a})
    # Dedupe while preserving order
    seen = set()
    deduped: List[Dict[str, str]] = []
    for item in qas:
        key = (item.get("question", ""), item.get("ideal_answer", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def build_index_from_app(app_mod):
    project_root = os.path.dirname(os.path.abspath(app_mod.__file__))
    chunks = app_mod.load_corpus(project_root)
    vectorizer, matrix = app_mod.build_index(chunks)
    return chunks, vectorizer, matrix


def answer_with_app(app_mod, question: str, chunks, vectorizer, matrix) -> Tuple[str, List[Tuple[any, float]]]:
    ranked = app_mod.rank_chunks(question, vectorizer, matrix, chunks, top_k=25)
    ai_answer = None
    try:
        ai_answer = app_mod.call_openai_generate(question, ranked, max_sentences=5)
    except Exception:
        ai_answer = None
    if ai_answer and ai_answer.strip():
        return ai_answer.strip(), ranked
    # Fallback extractive
    extractive_answer, _sources = app_mod.compose_answer(question, ranked)
    return extractive_answer.strip(), ranked


_STOPWORDS = {
    "the","a","an","and","or","of","to","in","for","on","by","with","as","at","from","is","are","was","were",
    "this","that","these","those","be","being","been","it","its","into","about","over","under","between","within",
    "we","you","they","he","she","them","their","our","your","i",
}


def tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z0-9_\-]+", text)
    return [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]


def top_keywords(text: str, max_k: int = 10) -> List[str]:
    toks = tokenize(text)
    if not toks:
        return []
    freq: Dict[str, int] = {}
    for t in toks:
        freq[t] = freq.get(t, 0) + 1
    # Sort by frequency then alphabetically
    items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in items[:max_k]]


def cosine_sim_12(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    try:
        X = vec.fit_transform([a, b])
        sim = float(cosine_similarity(X[0], X[1])[0][0])
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0


def compute_scores(app_mod, ideal: str, pred: str, ranked: List[Tuple[any, float]]) -> Dict[str, float]:
    # Similarity score
    sim = cosine_sim_12(ideal, pred)

    # Coverage of key points (keyword recall)
    ideal_kws = top_keywords(ideal, max_k=10)
    ans_toks = set(tokenize(pred))
    matched = sum(1 for k in ideal_kws if k in ans_toks)
    cov = 0.0 if not ideal_kws else matched / float(len(ideal_kws))

    # Support from documents: average of top-5 similarities scaled
    sims = [s for (_c, s) in ranked[:5]]
    mean_sim = sum(sims) / len(sims) if sims else 0.0
    support = max(0.0, min(1.0, mean_sim / 0.3))  # 0.3 ~ reasonable tf-idf cosine for good match

    # Clarity: 3-5 sentences preferred
    try:
        sents = app_mod.split_sentences(pred)
    except Exception:
        sents = re.split(r"(?<=[\.!?])\s+", pred)
    n_s = len([s for s in sents if s.strip()])
    if 3 <= n_s <= 5:
        clarity = 1.0
    elif 2 <= n_s <= 6:
        clarity = 0.7
    else:
        clarity = 0.4 if 1 <= n_s <= 8 else 0.0

    reliability = (0.4 * sim) + (0.3 * cov) + (0.2 * support) + (0.1 * clarity)
    return {
        "similarity": round(sim * 100, 2),
        "coverage": round(cov * 100, 2),
        "support": round(support * 100, 2),
        "clarity": round(clarity * 100, 2),
        "reliability": round(reliability * 100, 2),
    }


def is_unavailable_answer(text: str) -> bool:
    unavail_markers = [
        "I cannot find information in the provided chunks to answer this.",
        "No se encuentra en la información disponible.",
        "No encuentro información en los chunks proporcionados para responder a esto.",
    ]
    low = text.strip()
    return any(low.startswith(m) for m in unavail_markers)


def determine_correct(scores: Dict[str, float], answer_text: str) -> bool:
    if is_unavailable_answer(answer_text):
        return False
    # Majority of key points and reasonable similarity
    return (scores.get("coverage", 0.0) >= 60.0) and (scores.get("similarity", 0.0) >= 50.0)


def write_reports(rows: List[Dict[str, any]], out_base: str) -> Dict[str, str]:
    os.makedirs(os.path.dirname(out_base), exist_ok=True)

    # Results dataframe
    df = pd.DataFrame(rows)

    # Summary metrics
    N = len(rows)
    answered = sum(1 for r in rows if r["model_answer"].strip())
    correct = sum(1 for r in rows if r["correct"])
    avg_reliability = round(sum(r["reliability"] for r in rows) / N, 2) if N else 0.0
    acc_global = round((correct / N) * 100.0, 2) if N else 0.0
    summary_phrase = f"Based on {N} questions, the model is {acc_global}% accurate."

    summary = {
        "total_questions": N,
        "answered_pct": round((answered / N) * 100.0, 2) if N else 0.0,
        "correct_pct": round((correct / N) * 100.0, 2) if N else 0.0,
        "accuracy_global_pct": acc_global,
        "avg_reliability_pct": avg_reliability,
        "summary_phrase": summary_phrase,
    }

    # Excel with two sheets
    xlsx_path = out_base + ".xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, index=False, sheet_name="Summary")
        df.to_excel(writer, index=False, sheet_name="Results")

    # CSV and JSONL
    csv_path = out_base + ".csv"
    jsonl_path = out_base + ".jsonl"
    df.to_csv(csv_path, index=False)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    return {"xlsx": xlsx_path, "csv": csv_path, "jsonl": jsonl_path}


def main(pdf_path: str, app_path: str):
    app_mod = load_app_module(app_path)
    chunks, vectorizer, matrix = build_index_from_app(app_mod)

    qas = extract_qas_from_pdf(pdf_path)
    rows: List[Dict[str, any]] = []

    for idx, qa in enumerate(qas, start=1):
        q = qa.get("question", "").strip()
        ideal = qa.get("ideal_answer", "").strip()
        if not q:
            continue
        answer, ranked = answer_with_app(app_mod, q, chunks, vectorizer, matrix)
        scores = compute_scores(app_mod, ideal, answer, ranked)
        row = {
            "id": idx,
            "question": q,
            "ideal_answer": ideal,
            "model_answer": answer,
            "similarity": scores["similarity"],
            "coverage": scores["coverage"],
            "support": scores["support"],
            "clarity": scores["clarity"],
            "reliability": scores["reliability"],
        }
        row["correct"] = determine_correct(scores, answer)
        rows.append(row)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    out_base = os.path.join("eval_reports", f"hr_hub_eval_{timestamp}")
    paths = write_reports(rows, out_base)

    print("Generated reports:")
    print(paths["xlsx"])
    print(paths["csv"])
    print(paths["jsonl"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate H&R Hub chatbot against ideal answers from PDF.")
    parser.add_argument("pdf", help="Path to 'Evaluator questions.pdf'")
    parser.add_argument("--app", default=os.path.join(os.getcwd(), "H&R Hub.py"), help="Path to the Streamlit app file")
    args = parser.parse_args()
    main(args.pdf, args.app)


