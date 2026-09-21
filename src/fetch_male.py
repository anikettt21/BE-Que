"""
fetch_male.py
-------------
Connects to the male-cns:v1.0 dataset on neuprint.janelia.org using
neuprint-python's Client, pulls the full neuron / cell-type annotation
table (bodyId, type, instance, class hierarchy, soma side/neuromere,
predicted neurotransmitter, and fruitless / doublesex flags if available),
and saves it as data/male_annotations.parquet.

The auth token is read from the pre-existing test_neuprint.py config;
no interactive prompt is used.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from neuprint import Client, NeuronCriteria as NC

# ── 1. Auth & connection ─────────────────────────────────────────────────────
TOKEN   = "ab15fe59936e000268f2542e0a244091f7a24e4c6797365a6d2974c156cd087a"
SERVER  = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"

print(f"Connecting to {SERVER}  dataset={DATASET}")
client = Client(SERVER, dataset=DATASET, token=TOKEN)
print(f"  neuprint version : {client.fetch_version()}")

# ── 2. Pull neuron annotations via Cypher ────────────────────────────────────
# neuprint-python's fetch_neurons() sends one request per neuron for ROI info
# which is extremely slow on large datasets (~70 k neurons in male-cns).
# We use fetch_custom() with a single Cypher query to get everything in one
# round-trip, then handle the optional fruitless / doublesex fields gracefully.

print("\nFetching neuron annotation table …")

CYPHER = """
MATCH (n:Neuron)
RETURN
    n.bodyId          AS bodyId,
    n.type            AS cell_type,
    n.instance        AS instance,
    n.class           AS class,
    n.superclass      AS superclass,
    n.subclass        AS subclass,
    n.supertype       AS supertype,
    n.status          AS status,
    n.statusLabel     AS statusLabel,
    n.somaSide        AS soma_side,
    n.somaNeuromere   AS soma_neuromere,
    n.rootSide        AS root_side,
    n.entryNerve      AS entry_nerve,
    n.flywireType     AS flywire_type,
    n.predictedNt     AS predicted_nt,
    n.consensusNt     AS consensus_nt,
    n.predictedNtConfidence AS nt_confidence,
    n.pre             AS pre_synapses,
    n.post            AS post_synapses,
    n.upstream        AS upstream,
    n.downstream      AS downstream,
    n.fruitless       AS fruitless,
    n.doublesex       AS doublesex
"""

df = client.fetch_custom(CYPHER)
print(f"  Rows fetched : {len(df):,}")
print(f"  Columns      : {df.columns.tolist()}")

# ── 3. Report fruitless / doublesex coverage ─────────────────────────────────
fru_coverage = df["fruitless"].notna().sum()
dsx_coverage = df["doublesex"].notna().sum()
print(f"\nfruitless  non-null : {fru_coverage:,}  "
      f"({'present' if fru_coverage > 0 else 'not annotated in this dataset'})")
print(f"doublesex  non-null : {dsx_coverage:,}  "
      f"({'present' if dsx_coverage > 0 else 'not annotated in this dataset'})")

# ── 4. Light cleanup ──────────────────────────────────────────────────────────
# Cast bodyId to int64 (arrives as float if any NULLs exist in the column)
df["bodyId"] = pd.to_numeric(df["bodyId"], errors="coerce").astype("Int64")

# Drop rows with no bodyId (shouldn't happen, but be safe)
df = df.dropna(subset=["bodyId"])

# Sort by bodyId for reproducibility
df = df.sort_values("bodyId").reset_index(drop=True)

# ── 5. Save to parquet ────────────────────────────────────────────────────────
OUT_DIR  = Path(__file__).resolve().parent.parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "male_annotations.parquet"

df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved → {OUT_PATH}")

# ── 6. Summary ────────────────────────────────────────────────────────────────
print(f"\nDataFrame shape : {df.shape}")
print("\nFirst 5 rows:")
print(df.head(5).to_string(index=False))
