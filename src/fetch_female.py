"""
fetch_female.py
---------------
Connects to the BANC (Brain-And-Nerve-Cord) datastack via CAVEclient,
pulls the cell_info annotation table, pivots it into per-neuron columns
(neuron_id, cell_type, region, side), and saves as data/female_annotations.parquet.

Authentication uses the pre-configured chunkedgraph secret at
~/.cloudvolume/secrets/cave-secret.json — no token prompt needed.
"""

import json
import os
from pathlib import Path

# pyrefly: ignore [missing-import]
import caveclient
import pandas as pd

# ── 1. Auth: read token from the pre-configured cave secret ──────────────────
SECRET_PATH = Path.home() / ".cloudvolume" / "secrets" / "cave-secret.json"

with open(SECRET_PATH) as f:
    token = json.load(f)["token"]

# ── 2. Connect to BANC (public datastack) ───────────────────────────────────
DATASTACK = "brain_and_nerve_cord_public"

print(f"Connecting to datastack: {DATASTACK}")
client = caveclient.CAVEclient(DATASTACK, auth_token=token)
print(f"  Server : {client.server_address}")
mat_version = client.materialize.most_recent_version()
print(f"  Materialization version: {mat_version}")

# ── 3. Fetch the cell_info annotation table ──────────────────────────────────
# cell_info is a long-format table:
#   tag  → annotation value  (e.g. "CNS neuron", "soma on left", "brain")
#   tag2 → annotation class  (e.g. "primary class", "soma side", "soma region")
print("\nFetching cell_info table …")
raw = client.materialize.query_table(
    "cell_info",
    materialization_version=mat_version,
)
print(f"  Raw rows fetched: {len(raw):,}")

# Keep only the columns we need
raw = raw[["pt_root_id", "tag", "tag2"]].copy()

# ── 4. Pivot long → wide: one row per neuron ─────────────────────────────────
# Map tag2 category names to the output column names we want.
# Inspect what categories are actually present:
tag2_counts = raw["tag2"].value_counts()
print("\nAnnotation categories in cell_info (tag2 counts):")
print(tag2_counts.to_string())

# Canonical mapping: tag2 category → output column name.
# 'soma region' (152k rows) is the main anatomical-region annotation.
CATEGORY_MAP = {
    "primary class": "cell_type",                        # CNS neuron, sensory neuron, glia, …
    "soma region": "region",                             # brain, VNC, optic lobe compartments, …
    "soma side": "side",                                 # soma on left / right / midline
    "neuron identity": "identity",                       # named neuron (e.g. DNge104)
    "sensory neuron": "sensory_subtype",                 # sensory subtype
    "anterior-posterior projection pattern": "ap_projection",
    "left-right projection pattern": "lr_projection",
    "body part innervated": "body_part",
    "body side innervated": "body_side",
    "fast neurotransmitter": "neurotransmitter",
    "freeform": "freeform",
}

# For each desired output column, extract from the relevant tag2 category.
# When a neuron has multiple entries in the same category, take the first.
def extract_category(df: pd.DataFrame, tag2_value: str) -> pd.Series:
    """Return a Series mapping pt_root_id → tag for a given tag2 category."""
    subset = (
        df[df["tag2"] == tag2_value]
        .drop_duplicates("pt_root_id")
        .set_index("pt_root_id")["tag"]
    )
    return subset


# Build a dict of Series, one per output column
series_dict = {}
for tag2_val, col_name in CATEGORY_MAP.items():
    if tag2_val in raw["tag2"].values:
        series_dict[col_name] = extract_category(raw, tag2_val)

# Combine into a single DataFrame (index = pt_root_id)
wide = pd.DataFrame(series_dict)
wide.index.name = "neuron_id"
wide = wide.reset_index()

# Ensure the four required columns exist (fill with NaN if absent)
for col in ["cell_type", "region", "side"]:
    if col not in wide.columns:
        wide[col] = pd.NA

# Reorder: neuron_id first, then the four key columns, then extras
key_cols = ["neuron_id", "cell_type", "region", "side"]
extra_cols = [c for c in wide.columns if c not in key_cols]
wide = wide[key_cols + extra_cols]

# ── 5. Save to parquet ───────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "female_annotations.parquet"

wide.to_parquet(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")

# ── 6. Summary ───────────────────────────────────────────────────────────────
print(f"\nDataFrame shape : {wide.shape}")
print(f"Columns         : {wide.columns.tolist()}")
print("\nFirst 5 rows:")
print(wide.head(5).to_string(index=False))
