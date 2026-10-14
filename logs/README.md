# Experiment logs (2026-10-14)

Run logs for five rejected experiments, kept as evidence. Results and reasoning are in the root README (§17) and the knowledge graph (§5.4h, §5.4i).

| Log | Model / graphs | Experiment |
|---|---|---|
| `graph_builder.log`, `train*.log` | 11-feature graphs and models (step 1) | Payload node features: rebuild and retrain |
| `evaluate.log`, `evaluate_timing.log`, `evaluate_folds.log` | Step-1 models | Payload features, scoring unchanged |
| `evaluate*_payload.log` | Step-1 models | Step 2: per-ID payload baseline P (`--payload`) |
| `evaluate*_drivenorm.log` | Restored 9-feature models (current) | Step 3: per-drive score normalisation (`--drive-norm`) |
| `evaluate*_values.log` | Restored 9-feature models (current) | Per-ID value baseline V (`--values`): signal mean/min/max vs the ID's benign values |
| `evaluate*_gated.log` | Restored 9-feature models (current) | Step 3b: gated per-drive normalisation (history only takes windows below tau) |

All five were reverted. The step-1 graphs and models are archived (gitignored) under `graph-transformer/outputs/_payload_features_archive/`.
