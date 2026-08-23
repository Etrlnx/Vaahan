# Changelog

All notable changes and milestones across this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.4.0] - 2026-08-24 — Explainable AI (XAI) Layer & Visual Dashboards (MVP Complete)

### Added
- **Multi-Head Attention Extraction & Attention Rollout (`explainability/attention_analyzer.py`)**:
  - Hooks into `TransformerConv` across all 4 encoder layers to capture attention distributions.
  - Implements recursive Attention Rollout ($\mathbf{A}_{\text{rollout}} = \mathbf{A}_4 \cdot \mathbf{A}_3 \cdot \mathbf{A}_2 \cdot \mathbf{A}_1$) to isolate dominant message interaction paths.
- **Node & Physical Feature Attribution (`explainability/attribution.py`)**:
  - Decomposes per-node reconstruction residuals: $\mathbf{e}_v = (x_v - \hat{x}_v)^2 \odot \mathbf{w}$.
  - Ranks top anomalous CAN Arbitration IDs and calculates feature-level contributions across 9 signal dimensions.
- **Subgraph Localizer (`explainability/graph_localizer.py`)**:
  - Extracts $k$-hop interaction subgraphs surrounding anomalous IDs.
  - Annotates novel/rare transition edges with empirical negative log-likelihood ($G_{\text{error}}$).
- **Dual Incident Report Generator (`explainability/report_generator.py`)**:
  - Synthesizes all multi-source evidence into machine-readable JSON schemas and human-readable natural language triage reports with recommended gateway containment actions.
- **Visual Explanations Suite**:
  - `plot_component_radar.py`: Polar spider chart normalized against alert threshold $\tau_{\text{alert}}$.
  - `plot_id_attribution.py`: Dual-panel CAN ID error share and root-cause physical signal breakdown.
  - `plot_interactive_subgraph.py`: Standalone interactive HTML5 drag-and-drop network topology + static PNG snapshot.
  - `plot_xai_dashboard.py`: Unified 4-panel Matplotlib dashboard supporting interactive `--show` GUI inspection and clean file replacement on re-runs.
- **XAI Fidelity Benchmark (`explainability/evalxai.py`)**:
  - Validates model explanations against ROAD ground-truth attack targets and physical mechanisms.
  - Supports automatic hex/decimal cross-representation equivalence (`0x0D0` $\leftrightarrow$ `208`).

---

## [0.3.0] - 2026-08-23 — Multi-Deviation Anomaly Scoring & Zero-Day Benchmark

### Added
- **Zero-Day Detector (`attack-detection/detector.py`)**:
  - Fuses Reconstruction Error ($R$), Temporal Jitter ($T$), and Structural Sequence Penalty ($G$).
  - Calibrates empirical thresholds on held-out validation data ($\tau_{\text{suspicious}} = 0.7072$, $\tau_{\text{alert}} = 1.2440$).
- **Zero-Day Evaluation Benchmark (`attack-detection/evaluate.py`)**:
  - Rigorous evaluation on 905 held-out test windows across 17 distinct attack captures.
  - Performance: **ROC-AUC 0.9473**, **PR-AUC 0.8311**, **Precision 0.8562**, **Recall 0.7396**, **F1-Score 0.7937**, **FPR 2.85%**, **Detection Latency 0.00s–2.00s**.
- **Unit Test Suite (`tests/test_pipeline.py`)**:
  - Added 10 tests covering ROC-AUC/PR-AUC ranking, tie handling, detector contracts, and calibration edge cases.

---

## [0.2.0] - 2026-08-20 — Graph Transformer & Autoencoder Representation Learning

### Added
- Dynamic graph construction (`graph-transformer/graph_builder.py`) with 9D statistical node features, global ID vocabulary, and temporal directed adjacency edges.
- Coupled `GraphTransformerAutoencoder` (`graph-transformer/model.py`) with `TransformerConv` encoder and MLP decoder.
- Benign-only training pipeline (`graph-transformer/train.py`) with train-set-only feature normalization to prevent data leakage.

---

## [0.1.0] - 2026-08-15 — ROAD Dataset Ingestion & Temporal Windowing

### Added
- Temporal sliding-window preprocessor (`graph-transformer/preprocess.py`) for ORNL ROAD dataset.
- Ingestion of ambient captures and 17 signal-translated attack captures with metadata preservation.
