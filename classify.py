#!/usr/bin/env python3
"""
LieLines Validation — LLM Clarity Classifier
=============================================
Usage:
    python classify.py [--input sample.csv] [--output sample.csv]
                       [--lmstudio-url http://localhost:1234/v1]
                       [--model meta/llama-3.3-70b]
                       [--checkpoint classify_checkpoint.csv]
                       [--reliability-edge-pct 0.80]
                       [--reliability-size 200]
                       [--seed 42]

What it does:
  1. Reads sample.csv (produced by sample.py).
  2. Asks Llama 3.3 70B via LM Studio whether each flagged sentence is a
     CLEAR lying accusation or an EDGE case requiring judgment.
  3. Saves progress to a checkpoint file every 100 rows so the run is
     safely resumable.
  4. After all rows are classified, resamples the reliability batch:
         <reliability-edge-pct> × reliability-size  rows from EDGE pool
         (1 - reliability-edge-pct) × reliability-size rows from CLEAR pool
     Both sub-samples are stratified by country × source_dataset ×
     lie_score_bin × year_period.
  5. Overwrites --output with the new sample_type assignments and a
     `clarity` column (CLEAR / EDGE).
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

# ── Config ────────────────────────────────────────────────────────────────────

TEMPERATURE = 0
MAX_TOKENS  = 16   # response is a single word
SEED        = 42

CONTEXT_COLS_BEFORE = ["context_before_3", "context_before_2", "context_before_1"]
CONTEXT_COLS_AFTER  = ["context_after_1",  "context_after_2",  "context_after_3"]

STRAT_COLS = ["country", "source_dataset", "lie_score_bin", "year_period"]

CHECKPOINT_INTERVAL = 100

SYSTEM_PROMPT = """\
You are helping validate a parliamentary lie-detection model. Your task is binary classification.

A sentence has been flagged as a lying accusation in a parliamentary debate.
Decide whether it is a CLEAR or EDGE case.

CLEAR — an unambiguous, direct accusation that a specific person or group is \
lying or deliberately deceiving (e.g. "He is lying", "That is simply not true", \
"The minister has misled this House").

EDGE — anything requiring judgment: accusations of fraud or misconduct \
(not specifically lying), a speaker defending themselves against an accusation, \
figurative or rhetorical language, indirect accusations, or mentions of lying \
without a clear target.

Reply with a single word: CLEAR or EDGE"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_user_message(row: pd.Series) -> str:
    lines = []

    # Context before
    for col in CONTEXT_COLS_BEFORE:
        text = str(row.get(col, "") or "").strip()
        if text:
            lines.append(text)

    # Target sentence (marked)
    target = str(row.get("sentence", "") or "").strip()
    lines.append(f">>> {target} <<<")

    # Context after
    for col in CONTEXT_COLS_AFTER:
        text = str(row.get(col, "") or "").strip()
        if text:
            lines.append(text)

    # Metadata footer
    country = str(row.get("country", "") or "").strip()
    date    = str(row.get("date", "") or "").strip()[:4]
    dataset = str(row.get("source_dataset", "") or "").strip()
    lines.append(f"\nCountry: {country} | Year: {date} | Parliament: {dataset}")

    return "\n".join(lines)


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_clarity(text: str) -> str:
    """Return CLEAR or EDGE; default to EDGE on parse failure (conservative)."""
    upper = text.strip().upper()
    if "CLEAR" in upper:
        return "CLEAR"
    if "EDGE" in upper:
        return "EDGE"
    return "EDGE"


# ── Stratified subsample ──────────────────────────────────────────────────────

def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Draw n rows from df using proportional stratified sampling across
    country × source_dataset × lie_score_bin × year_period.
    """
    if len(df) == 0 or n == 0:
        return df.iloc[0:0].copy()

    n = min(n, len(df))
    df = df.copy()
    df["_stratum"] = (
        df["country"].fillna("unknown").astype(str) + "||" +
        df["source_dataset"].fillna("unknown").astype(str) + "||" +
        df["lie_score_bin"].fillna("unknown").astype(str) + "||" +
        df["year_period"].fillna("unknown").astype(str)
    )

    strata_counts = df["_stratum"].value_counts()
    total = len(df)

    alloc: dict[str, int] = {}
    for stratum, count in strata_counts.items():
        raw = max(1, round(count / total * n))
        alloc[stratum] = min(raw, int(count))

    while sum(alloc.values()) > n:
        biggest = max((s for s in alloc if alloc[s] > 1), key=alloc.get, default=None)
        if biggest is None:
            break
        alloc[biggest] -= 1

    parts = []
    for stratum, n_take in alloc.items():
        subset = df[df["_stratum"] == stratum]
        n_actual = min(n_take, len(subset))
        if n_actual > 0:
            parts.append(subset.sample(n=n_actual, random_state=seed))

    result = pd.concat(parts, ignore_index=True)

    if len(result) < n:
        taken = set(result["corpus_row_num"].tolist())
        remainder = df[~df["corpus_row_num"].isin(taken)]
        extra_n = min(n - len(result), len(remainder))
        if extra_n > 0:
            result = pd.concat([result, remainder.sample(n=extra_n, random_state=seed)],
                               ignore_index=True)

    if len(result) > n:
        result = result.sample(n=n, random_state=seed)

    return result.drop(columns=["_stratum"]).reset_index(drop=True)


# ── Reliability resampling ────────────────────────────────────────────────────

def resample_reliability(
    df: pd.DataFrame,
    reliability_size: int,
    edge_pct: float,
    seed: int,
) -> pd.DataFrame:
    """
    Rebuild sample_type labels so that the reliability batch is composed of
    round(reliability_size * edge_pct) EDGE rows +
    (reliability_size - that) CLEAR rows, both stratified.
    """
    n_edge  = round(reliability_size * edge_pct)
    n_clear = reliability_size - n_edge

    edge_pool  = df[df["clarity"] == "EDGE"].copy()
    clear_pool = df[df["clarity"] == "CLEAR"].copy()

    print(f"\n  EDGE pool  : {len(edge_pool):,}  (need {n_edge:,})")
    print(f"  CLEAR pool : {len(clear_pool):,}  (need {n_clear:,})")

    if len(edge_pool) < n_edge:
        print(f"  WARNING: only {len(edge_pool):,} EDGE rows — using all of them.")
        n_edge = len(edge_pool)
        n_clear = min(reliability_size - n_edge, len(clear_pool))

    rel_edge  = stratified_subsample(edge_pool,  n_edge,  seed)
    rel_clear = stratified_subsample(clear_pool, n_clear, seed)
    reliability = pd.concat([rel_edge, rel_clear], ignore_index=True)

    rel_ids = set(reliability["corpus_row_num"].tolist())

    df = df.copy()
    df["sample_type"] = df["corpus_row_num"].apply(
        lambda x: "reliability" if x in rel_ids else "unique"
    )

    n_rel  = (df["sample_type"] == "reliability").sum()
    n_uniq = (df["sample_type"] == "unique").sum()
    print(f"\n  New reliability batch : {n_rel:,} "
          f"({(df.loc[df['sample_type']=='reliability','clarity']=='EDGE').sum()} EDGE + "
          f"{(df.loc[df['sample_type']=='reliability','clarity']=='CLEAR').sum()} CLEAR)")
    print(f"  Unique pool           : {n_uniq:,}")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify LieLines sample sentences as CLEAR or EDGE via LM Studio"
    )
    p.add_argument("--input",                default="sample.csv")
    p.add_argument("--output",               default="sample.csv",
                   help="Overwrite this file with clarity column + new sample_type "
                        "(default: sample.csv)")
    p.add_argument("--lmstudio-url",         default="http://localhost:1234/v1")
    p.add_argument("--model",                default="meta/llama-3.3-70b")
    p.add_argument("--checkpoint",           default="classify_checkpoint.csv")
    p.add_argument("--reliability-edge-pct", type=float, default=0.80,
                   help="Fraction of reliability batch drawn from EDGE pool (default: 0.80)")
    p.add_argument("--reliability-size",     type=int,   default=200)
    p.add_argument("--seed",                 type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load sample ───────────────────────────────────────────────────────────
    print(f"Loading {args.input} …")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df):,} rows loaded")

    # ── Resume from checkpoint if present ────────────────────────────────────
    import os
    done: dict[int, str] = {}   # corpus_row_num → clarity
    if os.path.exists(args.checkpoint):
        ckpt = pd.read_csv(args.checkpoint, low_memory=False)
        done = dict(zip(ckpt["corpus_row_num"].astype(int),
                        ckpt["clarity"].astype(str)))
        print(f"  Resuming — {len(done):,} rows already classified (checkpoint loaded)")

    # ── Connect to LM Studio ─────────────────────────────────────────────────
    client = OpenAI(base_url=args.lmstudio_url, api_key="lm-studio")
    print(f"\nConnecting to LM Studio at {args.lmstudio_url} …")
    try:
        models = [m.id for m in client.models.list()]
        print(f"  Available models: {models}")
        if args.model not in models:
            print(f"  WARNING: '{args.model}' not in the list above — will attempt anyway")
    except Exception as e:
        print(f"  Could not list models: {e}")

    # ── Classify ──────────────────────────────────────────────────────────────
    print(f"\nClassifying {len(df):,} rows …")
    results: list[str] = []
    pending = [
        (i, row) for i, row in df.iterrows()
        if int(row["corpus_row_num"]) not in done
    ]
    print(f"  {len(pending):,} rows to classify  |  {len(done):,} already done\n")

    checkpoint_buf: list[dict] = []

    with tqdm(total=len(pending), unit="row") as pbar:
        for i, (idx, row) in enumerate(pending):
            user_msg = build_user_message(row)
            raw = ""
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    seed=SEED,
                )
                raw = resp.choices[0].message.content or ""
            except Exception as e:
                tqdm.write(f"  Error on row {int(row['corpus_row_num'])}: {e} — defaulting to EDGE")

            clarity = parse_clarity(raw)
            done[int(row["corpus_row_num"])] = clarity
            checkpoint_buf.append({"corpus_row_num": int(row["corpus_row_num"]),
                                   "clarity": clarity})
            pbar.update(1)

            # Save checkpoint every N rows
            if len(checkpoint_buf) >= CHECKPOINT_INTERVAL:
                ckpt_df = pd.DataFrame(checkpoint_buf)
                header = not os.path.exists(args.checkpoint)
                ckpt_df.to_csv(args.checkpoint, mode="a", index=False, header=header)
                checkpoint_buf.clear()

    # Flush remaining checkpoint buffer
    if checkpoint_buf:
        ckpt_df = pd.DataFrame(checkpoint_buf)
        header = not os.path.exists(args.checkpoint)
        ckpt_df.to_csv(args.checkpoint, mode="a", index=False, header=header)

    # ── Attach clarity column ────────────────────────────────────────────────
    df["clarity"] = df["corpus_row_num"].astype(int).map(done).fillna("EDGE")

    print(f"\n── Clarity distribution ──────────────────────────────────────────────")
    print(df["clarity"].value_counts().to_string())
    print(f"\n── Clarity by country ────────────────────────────────────────────────")
    print(df.groupby("country")["clarity"]
            .value_counts(normalize=True)
            .mul(100).round(1)
            .rename("pct")
            .to_string())

    # ── Resample reliability batch ────────────────────────────────────────────
    print(f"\n── Resampling reliability batch "
          f"({round(args.reliability_edge_pct*100):.0f}% EDGE + "
          f"{round((1-args.reliability_edge_pct)*100):.0f}% CLEAR) ──────────────")
    df = resample_reliability(df, args.reliability_size, args.reliability_edge_pct, args.seed)

    # ── Write output ──────────────────────────────────────────────────────────
    df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"\nOutput written to: {args.output}  ({len(df):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
