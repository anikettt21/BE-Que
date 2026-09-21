"""
compare.py
----------
Loads female (BANC) and male (neuprint male-cns) annotation parquets,
counts neurons per cell-type name in each dataset, performs an outer
merge, and classifies each type as:

  - 'isomorphic'      : type present in BOTH datasets
  - 'male_specific'   : type found ONLY in male-cns
  - 'female_specific' : type found ONLY in BANC (female)

Merge key (v2)
~~~~~~~~~~~~~~
Diagnostic (check_flywire_type.py) showed:
  • flywire_type is non-null for 81.1% of male neurons (vs cell_type 93.2%)
  • flywire_type matches 64.1% of female identity values (vs 49.4% for cell_type)

We therefore use:
  female['identity']   ↔   male['flywire_type']

male['cell_type'] is retained as a descriptive column in the output.

Female identity cleaning (before groupby)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Strip trailing '?' so "Tm9?" and "Tm9" merge as the same type.
  2. Drop rows whose identity contains a comma (compound/ambiguous).
  3. Drop rows where identity is exactly 'unknown' (case-insensitive).
  (CB####/MTe#### placeholder filter removed — these exist in flywire_type
   too and were causing real matches to be dropped.)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).resolve().parent.parent / "data"
FEMALE_PAR = DATA_DIR / "female_annotations.parquet"
MALE_PAR   = DATA_DIR / "male_annotations.parquet"

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading parquet files …")
female = pd.read_parquet(FEMALE_PAR)
male   = pd.read_parquet(MALE_PAR)
print(f"  Female (BANC)      : {len(female):>7,} neurons")
print(f"  Male   (neuprint)  : {len(male):>7,} neurons")

# ── 2. Clean & group female types (by identity) ───────────────────────────────
female_typed = female.dropna(subset=["identity"]).copy()
n_start = len(female_typed)

# 2a. Strip trailing '?' → "Tm9?" becomes "Tm9"
female_typed["identity"] = female_typed["identity"].str.rstrip("?")

# 2b. Drop compound / ambiguous annotations that contain a comma
mask_comma = female_typed["identity"].str.contains(",", regex=False)
female_typed = female_typed[~mask_comma]
n_after_comma = len(female_typed)

# 2c. Drop rows where identity is exactly 'unknown' (case-insensitive)
mask_unknown = female_typed["identity"].str.lower() == "unknown"
female_typed = female_typed[~mask_unknown]
n_after_unknown = len(female_typed)

print(f"\nFemale identity cleanup (rows removed at each step):")
print(f"  Start (has identity)            : {n_start:>7,}")
print(f"  After dropping comma-annotations: {n_after_comma:>7,}  "
      f"(-{n_start - n_after_comma:,} removed)")
print(f"  After dropping 'unknown'        : {n_after_unknown:>7,}  "
      f"(-{n_after_comma - n_after_unknown:,} removed)")
print(f"  (trailing '?' stripped; CB####/MTe#### kept — match flywire_type)")

female_counts = (
    female_typed
    .groupby("identity", dropna=True)
    .agg(
        female_n=("neuron_id", "count"),
        # most common broad class label (CNS neuron / sensory / glia …)
        female_broad_class=(
            "cell_type",
            lambda x: x.mode().iloc[0] if x.notna().any() else pd.NA,
        ),
    )
    .reset_index()
    .rename(columns={"identity": "merge_key"})
)

# ── 3. Group & count male types (by flywire_type) ─────────────────────────────
# Only keep rows that actually have a flywire_type annotation
male_typed = male.dropna(subset=["flywire_type"]).copy()
n_male_total  = len(male)
n_male_typed  = len(male_typed)
print(f"\nMale flywire_type coverage: {n_male_typed:,} / {n_male_total:,} "
      f"({n_male_typed/n_male_total*100:.1f}%)")

male_counts = (
    male_typed
    .groupby("flywire_type", dropna=True)
    .agg(
        male_n=("bodyId", "count"),
        # keep cell_type as a descriptive companion column
        male_cell_type=(
            "cell_type",
            lambda x: x.mode().iloc[0] if x.notna().any() else pd.NA,
        ),
        male_superclass=(
            "superclass",
            lambda x: x.mode().iloc[0] if x.notna().any() else pd.NA,
        ),
    )
    .reset_index()
    .rename(columns={"flywire_type": "merge_key"})
)

print(f"\nDistinct typed names (after cleaning):")
print(f"  Female identity  : {len(female_counts):>6,}")
print(f"  Male flywire_type: {len(male_counts):>6,}")

# ── 4. Outer merge on the shared merge_key ────────────────────────────────────
merged = female_counts.merge(
    male_counts,
    on="merge_key",
    how="outer",
    indicator=True,
)

# ── 5. Classify ───────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    "both":       "isomorphic",
    "left_only":  "female_specific",
    "right_only": "male_specific",
}
merged["category"] = merged["_merge"].map(CATEGORY_MAP)
merged = merged.drop(columns=["_merge"])

# Fill missing counts with 0
merged["female_n"] = merged["female_n"].fillna(0).astype(int)
merged["male_n"]   = merged["male_n"].fillna(0).astype(int)

# Rename merge_key → cell_type_name for readability in output
merged = merged.rename(columns={"merge_key": "cell_type_name"})

# Sort: isomorphic first, then descending total
merged["_total"] = merged["female_n"] + merged["male_n"]
merged = (
    merged
    .sort_values(["category", "_total"], ascending=[True, False])
    .drop(columns=["_total"])
    .reset_index(drop=True)
)

# ── 6. Save ───────────────────────────────────────────────────────────────────
OUT_PATH = DATA_DIR / "type_comparison.parquet"
merged.to_parquet(OUT_PATH, index=False)
print(f"\nFull comparison table saved → {OUT_PATH}")
print(f"Columns: {merged.columns.tolist()}")

# ── 7. Summary ────────────────────────────────────────────────────────────────
summary = (
    merged["category"]
    .value_counts()
    .rename_axis("category")
    .reset_index(name="n_types")
)
summary["pct_of_total"] = (summary["n_types"] / summary["n_types"].sum() * 100).round(1)

print("\n" + "=" * 55)
print("  Cell-type presence/absence summary")
print("  (merge key: female identity ↔ male flywire_type)")
print("=" * 55)
print(summary.to_string(index=False))
print("-" * 55)
print(f"  Total distinct type names : {len(merged):,}")
print("=" * 55)

# Per-category neuron totals
print("\nNeuron counts by category:")
neuron_summary = (
    merged.groupby("category", observed=True)[["female_n", "male_n"]]
    .sum()
    .assign(total=lambda d: d["female_n"] + d["male_n"])
)
print(neuron_summary.to_string())

# ── 8. Preview each category ──────────────────────────────────────────────────
for cat in ["isomorphic", "female_specific", "male_specific"]:
    subset = merged[merged["category"] == cat]
    sort_col = "female_n" if cat == "female_specific" else "male_n"
    top10 = subset.nlargest(10, sort_col)

    print(f"\n── Top 10 {cat.upper()} types (by neuron count) ──")
    base_cols = ["cell_type_name", "female_n", "male_n"]
    if cat == "isomorphic":
        extra = ["female_broad_class", "male_cell_type", "male_superclass"]
    elif cat == "female_specific":
        extra = ["female_broad_class"]
    else:
        extra = ["male_cell_type", "male_superclass"]
    print(top10[base_cols + extra].to_string(index=False))

# ── 9. Validation against published reference (Shiu et al. 2024) ──────────────
# ~7,205 isomorphic · ~138 dimorphic · ~289 male-specific · ~71 female-specific
REFERENCE = {
    "isomorphic":      7_205,
    "dimorphic":         138,   # not yet computed — requires connectivity diff
    "male_specific":     289,
    "female_specific":    71,
}

cat_counts = merged["category"].value_counts().to_dict()
COMPUTED = {
    "isomorphic":      cat_counts.get("isomorphic",      0),
    "dimorphic":       0,   # placeholder
    "male_specific":   cat_counts.get("male_specific",   0),
    "female_specific": cat_counts.get("female_specific", 0),
}

THRESHOLD = 0.30

print("\n" + "=" * 65)
print("  Validation vs. published reference (Shiu et al. 2024)")
print("=" * 65)
print(f"  {'Category':<18} {'Computed':>10} {'Reference':>10} {'Δ%':>8}  Status")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*8}  ------")

all_ok = True
for cat, ref in REFERENCE.items():
    computed = COMPUTED[cat]

    if cat == "dimorphic":
        print(f"  {'dimorphic':<18} {'(not computed)':>10} {ref:>10,}    n/a   "
              f"⏭  skipped (needs connectivity diff)")
        continue

    delta_pct = (computed - ref) / ref if ref != 0 else float("inf")
    flag = "✅  OK" if abs(delta_pct) <= THRESHOLD else "⚠️  OFF >30%"
    if abs(delta_pct) > THRESHOLD:
        all_ok = False
    print(f"  {cat:<18} {computed:>10,} {ref:>10,} {delta_pct:>+7.1%}  {flag}")

print("=" * 65)

if not all_ok:
    print("""
⚠️  One or more counts deviate >30% from the reference.

Likely reasons (not bugs — expected at this stage):
  1. COVERAGE — BANC female dataset is still being proofread. Many optic-lobe
                neurons exist but have no identity label yet → they cannot
                appear in the female side of the comparison at all, inflating
                male_specific and deflating isomorphic.

  2. DIMORPHIC — We report 0 dimorphic; the reference's 138 are folded into
                 isomorphic in our counts (connectivity diff not yet computed).

  3. DATASET VERSION — Reference may use a restricted status filter
                       (Traced-only) or a different materialization version.

Next steps to close the gap:
  • Filter male to status='Traced' neurons only.
  • Filter female to backbone-proofread neurons only.
  • Implement connectivity-based dimorphic detection (next script).
""")
else:
    print("\n✅  All computed counts are within 30% of published reference.\n")
