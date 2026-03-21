#!/usr/bin/env python3
"""
LieLines Validation — Stratified Sampling Script
=================================================

Usage:
    python sample.py --corpus /path/to/sentence_corpus_predicted.csv \\
                     [--output sample.csv] \\
                     [--reliability-size 200] \\
                     [--unique-pool-size 5000] \\
                     [--chunk-size 100000] \\
                     [--seed 42]

What it does:
  1. Streams the full corpus in chunks to collect all LIE-predicted rows.
  2. Performs stratified sampling across:
         country × source_dataset × lie_score_bin × year_period
     with proportional allocation and a floor of 1 per non-empty stratum.
  3. Shuffles the sample and labels the first <reliability-size> rows as
     "reliability" (shared across all annotators) and the rest as "unique".
  4. Does a second streaming pass to fetch ±3 context sentences for each
     sampled row, keeping only contiguous same-speech sentences.
  5. Writes sample.csv — the only file the annotation app ever reads.

Output columns (on top of all original corpus columns):
    lie_score_bin    low / mid / high  (< 0.6 / 0.6–0.8 / > 0.8)
    year_period      binned year label
    corpus_row_num   0-based row index in the original corpus
    context_before_3 third sentence before target (same speech), or ""
    context_before_2 second sentence before target, or ""
    context_before_1 sentence immediately before target, or ""
    context_after_1  sentence immediately after target, or ""
    context_after_2  second sentence after target, or ""
    context_after_3  third sentence after target, or ""
    sample_type      "reliability" or "unique"

Context ordering on screen: before_3 → before_2 → before_1 → [TARGET] →
                             after_1  → after_2  → after_3
"""

import argparse
import sys

import numpy as np
import pandas as pd

# ── Column names (must match the corpus) ─────────────────────────────────────
LIE_LABEL_COL = "lie_label"
LIE_SCORE_COL = "lie_score"
SENTENCE_COL  = "sentence"
SPEAKER_COL   = "speaker"
DATE_COL      = "date"
COUNTRY_COL   = "country"
DATASET_COL   = "source_dataset"
SPEECH_ID_COL = "source_speech_id"

CONTEXT_WINDOW = 3   # sentences before / after target to attempt


# ── Helper: stratification labels ────────────────────────────────────────────

def score_bin(score) -> str:
    try:
        s = float(score)
    except (ValueError, TypeError):
        return "unknown"
    if s < 0.6:
        return "low"
    if s < 0.8:
        return "mid"
    return "high"


def year_period(date_str) -> str:
    try:
        year = int(str(date_str).strip()[:4])
    except (ValueError, TypeError):
        return "unknown"
    if year < 2000:
        return "pre-2000"
    if year < 2010:
        return "2000-2009"
    if year < 2016:
        return "2010-2015"
    if year < 2020:
        return "2016-2019"
    return "2020+"


# ── Stratified sampling ───────────────────────────────────────────────────────

def stratified_sample(lies_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Draw n rows from lies_df using proportional stratified sampling.
    Strata = country × source_dataset × lie_score_bin × year_period.
    Guarantees at least 1 row per non-empty stratum; scales down if needed.
    """
    df = lies_df.copy()
    df["_stratum"] = (
        df[COUNTRY_COL].fillna("unknown").astype(str) + "||" +
        df[DATASET_COL].fillna("unknown").astype(str) + "||" +
        df["_score_bin"]  + "||" +
        df["_year_period"]
    )

    strata_counts = df["_stratum"].value_counts()
    n_strata = len(strata_counts)
    total    = len(df)
    print(f"  {n_strata:,} non-empty strata across {total:,} LIE predictions")

    # Proportional allocation with floor of 1 per stratum
    alloc: dict[str, int] = {}
    for stratum, count in strata_counts.items():
        raw = max(1, round(count / total * n))
        alloc[stratum] = min(raw, int(count))

    # Scale down if rounding pushed total over n
    while sum(alloc.values()) > n:
        biggest = max((s for s in alloc if alloc[s] > 1), key=alloc.get, default=None)
        if biggest is None:
            break
        alloc[biggest] -= 1

    # Sample from each stratum
    parts = []
    for stratum, n_take in alloc.items():
        subset = df[df["_stratum"] == stratum]
        n_actual = min(n_take, len(subset))
        if n_actual > 0:
            parts.append(subset.sample(n=n_actual, random_state=seed))

    sample = pd.concat(parts, ignore_index=True)

    # Top up to n if we're still short (small strata may have under-delivered)
    if len(sample) < n:
        taken = set(sample["_row_num"].tolist())
        remainder = df[~df["_row_num"].isin(taken)]
        extra_n = min(n - len(sample), len(remainder))
        if extra_n > 0:
            extra = remainder.sample(n=extra_n, random_state=seed)
            sample = pd.concat([sample, extra], ignore_index=True)

    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)

    return sample.drop(columns=["_stratum"]).reset_index(drop=True)


# ── Context fetching ──────────────────────────────────────────────────────────

def fetch_context(
    corpus_path: str,
    sample_df: pd.DataFrame,
    chunk_size: int,
) -> pd.DataFrame:
    """
    Stream the corpus a second time to collect ±CONTEXT_WINDOW rows for each
    sampled sentence.  Only contiguous sentences from the same speech are kept
    (determined by source_speech_id if present, speaker otherwise).
    """
    # Build the complete set of row numbers we need from the corpus
    needed: set[int] = set()
    for rn in sample_df["_row_num"].astype(int):
        for offset in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
            nr = rn + offset
            if nr >= 0:
                needed.add(nr)

    max_needed = max(needed)
    print(f"  Need to fetch {len(needed):,} rows (up to corpus row {max_needed:,})")

    # Stream and collect
    context_store: dict[int, dict] = {}
    current_row = 0

    for chunk in pd.read_csv(corpus_path, chunksize=chunk_size, low_memory=False):
        n_chunk = len(chunk)
        chunk["_row_num"] = range(current_row, current_row + n_chunk)
        relevant = chunk[chunk["_row_num"].isin(needed)]
        for _, row in relevant.iterrows():
            context_store[int(row["_row_num"])] = {
                "sentence":  str(row.get(SENTENCE_COL,  "") or ""),
                "speaker":   str(row.get(SPEAKER_COL,   "") or ""),
                "speech_id": str(row.get(SPEECH_ID_COL, "") or ""),
            }
        current_row += n_chunk
        if current_row > max_needed:
            break   # No need to scan the rest of a 149 M-row file

    print(f"  Retrieved {len(context_store):,} context rows from corpus")

    # Helper: does a candidate row belong to the same speech as the target?
    def same_speech(target_spk: str, target_sid: str, candidate: dict) -> bool:
        if target_sid and candidate["speech_id"]:
            return candidate["speech_id"] == target_sid
        return candidate["speaker"] == target_spk

    # Build per-row context columns
    col_names_before = [f"context_before_{i}" for i in range(CONTEXT_WINDOW, 0, -1)]
    col_names_after  = [f"context_after_{i}"  for i in range(1, CONTEXT_WINDOW + 1)]
    context_cols: dict[str, list] = {c: [] for c in col_names_before + col_names_after}

    for _, row in sample_df.iterrows():
        rn   = int(row["_row_num"])
        spk  = str(row.get(SPEAKER_COL,   "") or "")
        sid  = str(row.get(SPEECH_ID_COL, "") or "")

        # Before: scan outward (-1, -2, -3); stop at first speaker change
        # context_before_1 = immediately before, context_before_3 = furthest before
        before: dict[str, str] = {}
        for offset in range(-1, -(CONTEXT_WINDOW + 1), -1):
            col = f"context_before_{-offset}"   # offset -1 → col 1 (closest)
            nr  = rn + offset
            if nr >= 0 and nr in context_store and same_speech(spk, sid, context_store[nr]):
                before[col] = context_store[nr]["sentence"]
            else:
                break   # contiguity broken — don't go further back

        # After: scan outward (+1, +2, +3)
        after: dict[str, str] = {}
        for offset in range(1, CONTEXT_WINDOW + 1):
            col = f"context_after_{offset}"
            nr  = rn + offset
            if nr in context_store and same_speech(spk, sid, context_store[nr]):
                after[col] = context_store[nr]["sentence"]
            else:
                break

        for c in col_names_before:
            context_cols[c].append(before.get(c, ""))
        for c in col_names_after:
            context_cols[c].append(after.get(c, ""))

    for col, vals in context_cols.items():
        sample_df[col] = vals

    return sample_df


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified sampling for LieLines false-positive validation"
    )
    p.add_argument("--corpus",           required=True,
                   help="Path to sentence_corpus_predicted.csv")
    p.add_argument("--output",           default="sample.csv",
                   help="Output file path (default: sample.csv)")
    p.add_argument("--reliability-size", type=int, default=200,
                   help="Rows in the shared reliability pool (default: 200)")
    p.add_argument("--unique-pool-size", type=int, default=5000,
                   help="Rows in the unique annotation pool (default: 5000)")
    p.add_argument("--chunk-size",       type=int, default=100_000,
                   help="Streaming chunk size (default: 100000)")
    p.add_argument("--seed",             type=int, default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_n = args.reliability_size + args.unique_pool_size

    # ── Pass 1: Collect all LIE predictions ──────────────────────────────────
    print(f"\nPass 1  Streaming corpus to collect LIE-predicted rows…")
    print(f"        (corpus: {args.corpus})")

    parts: list[pd.DataFrame] = []
    current_row = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(args.corpus, chunksize=args.chunk_size, low_memory=False)
    ):
        n_chunk = len(chunk)
        chunk["_row_num"] = range(current_row, current_row + n_chunk)
        mask = chunk[LIE_LABEL_COL].astype(str).str.strip() == "LABEL_1"
        parts.append(chunk[mask].copy())
        current_row += n_chunk
        if (chunk_idx + 1) % 10 == 0:
            n_so_far = sum(len(p) for p in parts)
            print(f"  … {current_row:>12,} corpus rows scanned  |  {n_so_far:>8,} LIE rows collected")

    lies_df = pd.concat(parts, ignore_index=True)
    print(f"\n  Total corpus rows : {current_row:,}")
    print(f"  LIE predictions   : {len(lies_df):,}")

    if len(lies_df) == 0:
        sys.exit("ERROR: No LIE-labelled rows found. Check --corpus path and lie_label column.")

    if len(lies_df) < total_n:
        print(f"\nWARNING: Only {len(lies_df):,} LIE rows available — shrinking sample target.")
        total_n = len(lies_df)
        args.reliability_size = min(args.reliability_size, total_n // 2)
        args.unique_pool_size = total_n - args.reliability_size

    # ── Add stratification labels ─────────────────────────────────────────────
    print("\nComputing stratification labels…")
    lies_df["_score_bin"]   = lies_df[LIE_SCORE_COL].apply(score_bin)
    lies_df["_year_period"] = lies_df[DATE_COL].apply(year_period)

    # ── Stratified sample ─────────────────────────────────────────────────────
    print(f"\nSampling {total_n:,} rows ({args.reliability_size:,} reliability + "
          f"{args.unique_pool_size:,} unique)…")
    sample_df = stratified_sample(lies_df, total_n, args.seed)

    # Shuffle so reliability / unique are randomly interleaved before splitting
    sample_df = sample_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Rename internal columns to their final output names
    sample_df = sample_df.rename(columns={
        "_score_bin":   "lie_score_bin",
        "_year_period": "year_period",
        "_row_num":     "corpus_row_num",
    })

    # Label: first reliability_size rows → reliability, the rest → unique
    sample_df["sample_type"] = "unique"
    sample_df.loc[: args.reliability_size - 1, "sample_type"] = "reliability"

    n_rel  = (sample_df["sample_type"] == "reliability").sum()
    n_uniq = (sample_df["sample_type"] == "unique").sum()
    print(f"  Reliability pool : {n_rel:,}")
    print(f"  Unique pool      : {n_uniq:,}")

    # ── Pass 2: Fetch context sentences ──────────────────────────────────────
    print(f"\nPass 2  Fetching context sentences (±{CONTEXT_WINDOW} rows, same speech)…")
    # Restore _row_num temporarily for context fetch (it was renamed above)
    sample_df["_row_num"] = sample_df["corpus_row_num"]
    sample_df = fetch_context(args.corpus, sample_df, args.chunk_size)
    sample_df = sample_df.drop(columns=["_row_num"])

    # ── Write output ──────────────────────────────────────────────────────────
    sample_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nOutput written to: {args.output}  ({len(sample_df):,} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Sample summary " + "─" * 58)
    print(f"\nBy country ({COUNTRY_COL}):")
    print(sample_df[COUNTRY_COL].value_counts().to_string())
    print(f"\nBy source_dataset:")
    print(sample_df[DATASET_COL].value_counts().to_string())
    print(f"\nBy lie_score_bin:")
    print(sample_df["lie_score_bin"].value_counts().to_string())
    print(f"\nBy year_period:")
    print(sample_df["year_period"].value_counts().to_string())
    ctx_filled = sum(
        (sample_df[f"context_before_{i}"] != "").sum() +
        (sample_df[f"context_after_{i}"].sum() != "")
        for i in range(1, CONTEXT_WINDOW + 1)
    )
    print(f"\nContext coverage: {ctx_filled:,} / {len(sample_df) * CONTEXT_WINDOW * 2:,} "
          f"context slots filled")


if __name__ == "__main__":
    main()
