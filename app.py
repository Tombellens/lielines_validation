"""
LieLines Validation — Annotation App
=====================================
Flask backend for binary (YES / NO) validation of LIE-predicted sentences.

Environment variables:
    APP_PASSWORD     Login password (default: lielines-validate)
    ANNOTATIONS_DIR  Where to write annotations.csv and assignments.json
                     (default: same directory as this file; set to a
                     Railway persistent volume path in production, e.g. /data)
    UNIQUE_BLOCK_SIZE  Unique rows assigned to each new annotator (default: 500)
    PORT             Server port (default: 5050)
    FLASK_DEBUG      Enable Flask debug mode (default: false)

Sequence logic
--------------
Each annotator sees an interleaved stream of sentences:
    seq_pos 0  → reliability row 0
    seq_pos 1  → their unique row 0
    seq_pos 2  → reliability row 1
    seq_pos 3  → their unique row 1
    ...
i.e. even seq_pos → reliability[seq_pos // 2]
     odd  seq_pos → unique[unique_start + seq_pos // 2]

When a new annotator registers the app assigns them the next unused block of
UNIQUE_BLOCK_SIZE rows from the unique pool in sample.csv.  Assignments are
persisted to assignments.json so they survive server restarts.
"""

import csv
import datetime
import functools
import json
import os
import secrets

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuration ─────────────────────────────────────────────────────────────
SAMPLE_CSV        = os.path.join(BASE_DIR, "sample.csv")
ANNOTATIONS_DIR   = os.environ.get("ANNOTATIONS_DIR", BASE_DIR)
ANNOTATIONS_CSV   = os.path.join(ANNOTATIONS_DIR, "annotations.csv")
ASSIGNMENTS_FILE  = os.path.join(ANNOTATIONS_DIR, "assignments.json")
UNIQUE_BLOCK_SIZE = int(os.environ.get("UNIQUE_BLOCK_SIZE", "500"))
PASSWORD          = os.environ.get("APP_PASSWORD", "lielines-validate")

valid_tokens: set[str] = set()

# ── Auth ──────────────────────────────────────────────────────────────────────

def require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("X-Auth-Token", "")
        if token not in valid_tokens:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/api/login", methods=["POST"])
def login():
    payload = request.get_json() or {}
    if payload.get("password") == PASSWORD:
        token = secrets.token_hex(32)
        valid_tokens.add(token)
        return jsonify({"token": token})
    return jsonify({"error": "Invalid password"}), 401


# ── Sample data ───────────────────────────────────────────────────────────────
_reliability_rows: list[dict] = []
_unique_rows:      list[dict] = []
_sample_fieldnames: list[str] = []


def load_sample() -> tuple[list[dict], list[dict]]:
    global _reliability_rows, _unique_rows, _sample_fieldnames
    if _reliability_rows or _unique_rows:
        return _reliability_rows, _unique_rows
    if not os.path.exists(SAMPLE_CSV):
        raise FileNotFoundError(
            f"sample.csv not found at {SAMPLE_CSV}. "
            "Run sample.py first to generate it."
        )
    with open(SAMPLE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        _sample_fieldnames = list(reader.fieldnames or [])
    _reliability_rows = [r for r in rows if r.get("sample_type") == "reliability"]
    _unique_rows      = [r for r in rows if r.get("sample_type") == "unique"]
    return _reliability_rows, _unique_rows


def out_fieldnames() -> list[str]:
    """Full list of columns written to annotations.csv."""
    load_sample()
    annotation_meta = [
        "annotator_name", "seq_pos", "is_reliability", "verdict", "timestamp",
    ]
    return _sample_fieldnames + annotation_meta


# ── Annotator assignments ─────────────────────────────────────────────────────

def load_assignments() -> dict:
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    if not os.path.exists(ASSIGNMENTS_FILE):
        return {}
    with open(ASSIGNMENTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def save_assignments(data: dict) -> None:
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    with open(ASSIGNMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_or_assign(normalized_name: str) -> dict:
    """Return existing assignment or create a new block for this annotator."""
    assignments = load_assignments()
    key = normalized_name.lower()
    if key in assignments:
        return assignments[key]

    _, unique_rows = load_sample()

    # Find the next free start index in the unique pool
    next_start = max(
        (v["unique_start"] + v["unique_size"] for v in assignments.values()),
        default=0,
    )
    unique_size = min(UNIQUE_BLOCK_SIZE, max(0, len(unique_rows) - next_start))

    assignment = {
        "display_name": normalized_name,
        "unique_start": next_start,
        "unique_size":  unique_size,
    }
    assignments[key] = assignment
    save_assignments(assignments)
    return assignment


# ── Sequence logic ────────────────────────────────────────────────────────────

def row_at_seq_pos(
    normalized_name: str, seq_pos: int
) -> tuple[dict | None, str]:
    """
    Return (sample_row, row_type) for this annotator's sequence position.
    row_type is "reliability", "unique", "reliability_exhausted",
    or "unique_exhausted".
    """
    reliability_rows, unique_rows = load_sample()
    assignment = get_or_assign(normalized_name)

    if seq_pos % 2 == 0:
        r_idx = seq_pos // 2
        if r_idx >= len(reliability_rows):
            return None, "reliability_exhausted"
        return reliability_rows[r_idx], "reliability"
    else:
        u_idx = assignment["unique_start"] + seq_pos // 2
        if u_idx >= len(unique_rows):
            return None, "unique_exhausted"
        return unique_rows[u_idx], "unique"


def max_seq_pos(normalized_name: str) -> int:
    """Last valid seq_pos for this annotator."""
    reliability_rows, unique_rows = load_sample()
    assignment = get_or_assign(normalized_name)
    max_r = 2 * (len(reliability_rows) - 1)
    max_u = 2 * (min(assignment["unique_size"], len(unique_rows) - assignment["unique_start"]) - 1) + 1
    return max(max_r, max_u)


# ── Annotations I/O ───────────────────────────────────────────────────────────

def annotated_positions(normalized_name: str) -> set[int]:
    """Return seq_pos values already annotated by this annotator (last write wins)."""
    if not os.path.exists(ANNOTATIONS_CSV):
        return set()
    norm = normalized_name.lower()
    done: set[int] = set()
    with open(ANNOTATIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("annotator_name", "").lower() == norm:
                try:
                    done.add(int(row["seq_pos"]))
                except (ValueError, KeyError):
                    pass
    return done


def existing_verdict(normalized_name: str, seq_pos: int) -> str | None:
    """Return the most recently saved verdict for this annotator + position."""
    if not os.path.exists(ANNOTATIONS_CSV):
        return None
    norm = normalized_name.lower()
    last: str | None = None
    with open(ANNOTATIONS_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                row.get("annotator_name", "").lower() == norm
                and row.get("seq_pos") == str(seq_pos)
            ):
                last = row.get("verdict")
    return last


def append_annotation(out_row: dict) -> None:
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    fields = out_fieldnames()
    file_exists = os.path.exists(ANNOTATIONS_CSV)
    with open(ANNOTATIONS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(out_row)


# ── Helper: normalise annotator name ─────────────────────────────────────────

def normalise(name: str) -> str:
    """First word, Title Case — so 'jan', 'JAN', 'Jan de Vries' → 'Jan'."""
    first = name.strip().split()[0] if name.strip() else ""
    return first.capitalize()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from flask import make_response
    resp = make_response(send_from_directory(BASE_DIR, "index.html"))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/register", methods=["POST"])
@require_auth
def register():
    """
    Register / resume an annotator session.
    Returns their assignment info, progress, and the next unannotated seq_pos.
    """
    payload = request.get_json() or {}
    raw_name = payload.get("name", "").strip()
    if not raw_name:
        return jsonify({"error": "Name is required"}), 400

    name = normalise(raw_name)
    if not name:
        return jsonify({"error": "Could not parse name"}), 400

    reliability_rows, _ = load_sample()
    assignment = get_or_assign(name)
    done = annotated_positions(name)

    # Find first unannotated seq_pos
    cap = max_seq_pos(name)
    next_pos = next((i for i in range(0, cap + 2) if i not in done), cap)

    r_done = sum(1 for p in done if p % 2 == 0)
    u_done = sum(1 for p in done if p % 2 == 1)

    return jsonify({
        "display_name":       name,
        "reliability_total":  len(reliability_rows),
        "unique_total":       assignment["unique_size"],
        "reliability_done":   r_done,
        "unique_done":        u_done,
        "annotated_positions": sorted(done),
        "next_seq_pos":       next_pos,
    })


@app.route("/api/row/<int:seq_pos>")
@require_auth
def get_row(seq_pos: int):
    """Return the sample row + context at this sequence position."""
    annotator = request.args.get("annotator", "").strip()
    if not annotator:
        return jsonify({"error": "annotator param required"}), 400

    name = normalise(annotator)
    row, row_type = row_at_seq_pos(name, seq_pos)

    if row is None:
        return jsonify({"error": row_type}), 404

    # Check for existing verdict (so frontend can show it on re-visit)
    verdict = existing_verdict(name, seq_pos)

    # Build context list: chronological order for display
    # before_3 (oldest) → before_2 → before_1 → [target] → after_1 → after_2 → after_3
    context_before = [
        row.get(f"context_before_{i}", "")
        for i in range(3, 0, -1)
    ]
    context_after = [
        row.get(f"context_after_{i}", "")
        for i in range(1, 4)
    ]

    return jsonify({
        "seq_pos":  seq_pos,
        "row_type": row_type,
        "data": {
            "sentence":       row.get("sentence", ""),
            "context_before": [s for s in context_before if s],
            "context_after":  [s for s in context_after  if s],
            "country":        row.get("country", ""),
            "date":           row.get("date", ""),
            "speaker":        row.get("speaker", ""),
            "source_dataset": row.get("source_dataset", ""),
            # lie_score intentionally excluded from display; kept in CSV output
        },
        "existing_verdict": verdict,
    })


@app.route("/api/annotate", methods=["POST"])
@require_auth
def annotate():
    payload  = request.get_json() or {}
    raw_name = payload.get("annotator_name", "").strip()
    seq_pos  = payload.get("seq_pos")
    verdict  = str(payload.get("verdict", "")).strip().upper()

    if not raw_name:
        return jsonify({"error": "annotator_name required"}), 400
    if seq_pos is None:
        return jsonify({"error": "seq_pos required"}), 400
    if verdict not in ("YES", "NO"):
        return jsonify({"error": "verdict must be YES or NO"}), 400

    name = normalise(raw_name)
    row, row_type = row_at_seq_pos(name, int(seq_pos))
    if row is None:
        return jsonify({"error": f"Position out of range: {row_type}"}), 404

    out_row = dict(row)
    out_row["annotator_name"] = name
    out_row["seq_pos"]        = seq_pos
    out_row["is_reliability"] = (int(seq_pos) % 2 == 0)
    out_row["verdict"]        = verdict
    out_row["timestamp"]      = datetime.datetime.utcnow().isoformat()

    append_annotation(out_row)
    return jsonify({"success": True, "seq_pos": seq_pos, "verdict": verdict})


@app.route("/api/download")
@require_auth
def download():
    if not os.path.exists(ANNOTATIONS_CSV):
        return jsonify({"error": "No annotations saved yet"}), 404
    return send_file(
        ANNOTATIONS_CSV,
        as_attachment=True,
        download_name="lielines_annotations.csv",
        mimetype="text/csv",
    )


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load sample data at startup so the first request is fast
    try:
        r, u = load_sample()
        print(f"Loaded sample.csv: {len(r)} reliability rows, {len(u)} unique rows.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")

    port  = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
