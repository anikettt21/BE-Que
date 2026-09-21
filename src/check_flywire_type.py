"""
check_flywire_type.py
---------------------
Diagnostic: loads data/male_annotations.parquet and reports coverage
of the flywire_type column — how many neurons have a non-null value,
what fraction of the dataset that represents, and 10 example values.
"""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
male = pd.read_parquet(DATA_DIR / "male_annotations.parquet")

total      = len(male)
non_null   = male["flywire_type"].notna().sum()
null_count = total - non_null
coverage   = non_null / total * 100

print(f"male_annotations.parquet — flywire_type coverage")
print(f"{'='*52}")
print(f"  Total rows          : {total:>8,}")
print(f"  Non-null flywire_type: {non_null:>8,}  ({coverage:.1f}%)")
print(f"  Null / missing       : {null_count:>8,}  ({100-coverage:.1f}%)")
print()

# Unique value counts
unique_vals = male["flywire_type"].dropna().unique()
print(f"  Unique flywire_type values : {len(unique_vals):,}")
print()

# 10 examples (most common)
print("  Top 20 most common flywire_type values:")
top = male["flywire_type"].value_counts().head(20)
for val, cnt in top.items():
    print(f"    {val:<35}  n={cnt:,}")

print()
print("  10 random non-null examples:")
sample = male["flywire_type"].dropna().sample(min(10, non_null), random_state=42).tolist()
for v in sample:
    print(f"    {v}")

print()
# Compare unique flywire_type vs unique cell_type
ft_unique  = set(male["flywire_type"].dropna().unique())
ct_unique  = set(male["cell_type"].dropna().unique())
print(f"  Unique cell_type values    : {len(ct_unique):,}")
print(f"  Unique flywire_type values : {len(ft_unique):,}")
overlap = ft_unique & ct_unique
print(f"  Overlap (fwt == ct)        : {len(overlap):,}")
only_ft = ft_unique - ct_unique
only_ct = ct_unique - ft_unique
print(f"  Only in flywire_type       : {len(only_ft):,}")
print(f"  Only in cell_type          : {len(only_ct):,}")
print()

# How many flywire_type values appear in the female identity column?
female = pd.read_parquet(DATA_DIR / "female_annotations.parquet")
f_ids = set(female["identity"].dropna().str.rstrip("?").unique())
ft_in_female = ft_unique & f_ids
ct_in_female = ct_unique & f_ids
print(f"  Female identity unique (after ?-strip): {len(f_ids):,}")
print(f"  flywire_type values found in female   : {len(ft_in_female):,}  "
      f"({len(ft_in_female)/len(ft_unique)*100:.1f}% of flywire_type)")
print(f"  cell_type values found in female      : {len(ct_in_female):,}  "
      f"({len(ct_in_female)/len(ct_unique)*100:.1f}% of cell_type)")
print()

if coverage >= 20:
    print(f"✅  Coverage {coverage:.1f}% ≥ 20% threshold.")
    print("   Recommend: use flywire_type as the merge key in compare.py")
else:
    print(f"⚠️  Coverage {coverage:.1f}% < 20% threshold.")
    print("   flywire_type too sparse — keep cell_type as merge key")
