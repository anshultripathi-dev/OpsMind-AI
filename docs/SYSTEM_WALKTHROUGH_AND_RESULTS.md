# OpsMind AI: System Architecture, Major Functionalities & ML Evaluation Results

**Project Title:** A Machine Learning Framework for Incident Prediction and Root-Cause Analysis in Microservice Systems  
**Scope:** End-to-end AIOps platform for microservice observability, incident forecasting, and explainable diagnosis.
**Artifact Directory & High-Resolution Image Directory:** `docs/screenshots/` (All captures: 3360 × 2100 Retina resolution)

---

## Executive Overview

OpsMind AI is an end-to-end autonomous AIOps platform built to detect abnormal telemetry patterns in distributed microservices, forecast service-level objective (SLO) breaches **minutes before user impact occurs**, and pinpoint root-cause candidate components using game-theoretic feature attribution (SHAP) combined with topology graph walks.

This document compiles high-resolution visual evidence across all system layers—from the real-time operational dashboard and live chaos fault injector to the offline validation benchmarks and Model Lab evaluation scorecards.

```
+----------------------------------------------------------------------------------------------------+
|                                      OpsMind AI Pipeline Flow                                      |
+----------------------------------------------------------------------------------------------------+
|  Docker Testbed / Simulator                                                                        |
|    | (43 Prometheus Metrics @ 5s + Structured Container Logs)                                      |
|    v                                                                                               |
|  Preprocessing & Feature Extraction                                                                |
|    | 30s rolling sliding windows (mean, std, min, max, Δ) + Drain-3 log template frequency         |
|    v                                                                                               |
|  ML Core Engine                                                                                    |
|    +--> Module 2: Dimensionality Reduction (PCA vs Random Projection, 2D State Space)              |
|    +--> Module 3: Anomaly Detection (One-Class SVM vs PCA-Reconstruction vs Isolation Forest)      |
|    +--> Modules 4 & 5: Supervised Classification & Stacking Ensemble (6 Fault Classes)             |
|    +--> Module 6: Unsupervised Clustering (K-Means ARI 0.667 vs Ward vs DBSCAN)                   |
|    +--> Module 7: Early-Warning Sequential Model (LSTM / GRU lead time up to 120s)                 |
|    +--> Module 8: Root-Cause Attribution (SHAP explainability + Dependency Graph Walk)             |
|    v                                                                                               |
|  FastAPI Serving Engine (REST + WebSocket @ ws://localhost:8001/ws/live)                           |
|    v                                                                                               |
|  React + TypeScript + Tailwind Analytics Dashboard                                                |
+----------------------------------------------------------------------------------------------------+
```

---

## Table of Contents

1. [Live Operations & Continuous Monitoring](#1-live-operations--continuous-monitoring)
   - Figure 1: Live Operations Under Normal Steady-State Operation
   - Figure 2: Chaos Engineering Console & Fault Injection Controls
   - Figure 3: Real-Time Incident State: Early Warning, Degraded Topology & SHAP RCA
2. [Incident Management & Held-Out Test Evaluation](#2-incident-management--held-out-test-evaluation)
   - Figure 4: Incident Registry & Held-Out Episode Lead-Time Benchmarks
3. [Model Lab — Comprehensive Machine Learning Results](#3-model-lab--comprehensive-machine-learning-results)
   - Figure 5: Executive Success Criteria & Baseline Comparison Scorecard
   - Figure 6: Module 2 — Dimensionality Reduction & 2D PCA Latent Space
   - Figure 7: Module 3 — Anomaly Detection Under False-Alarm Budget
   - Figure 8: Modules 4 & 5 — Fault Classification & 6-Class Confusion Matrix
   - Figure 9: Module 6 — Unsupervised Clustering & Discovered Failure Signatures
   - Figure 10: Module 7 — Early-Warning LSTM Lead-Time Dynamics
   - Figure 11: Module 8 — Root-Cause Analysis (SHAP + Graph Walk Ablation)
   - Figure 12: Real Docker Testbed Campaigns (Campaign 1 vs Campaign 2)
4. [Testbed Architecture & Dataset Provenance](#4-testbed-architecture--dataset-provenance)
   - Figure 13: Distributed Microservices Architecture & Telemetry Pipeline
   - Figure 14: Dataset Class Balance & Drain Log Parsing Templates
5. [Developer Surface & API Architecture](#5-developer-surface--api-architecture)
   - Figure 15: FastAPI Interactive OpenAPI/Swagger Documentation

---

## 1. Live Operations & Continuous Monitoring

### Figure 1: Live Operations Under Normal Steady-State Operation

![Figure 1: Live Operations Normal](screenshots/01_live_operations_normal.png)

* **View / Component:** Live Operations Main Dashboard (`http://localhost:8001/`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:** 
  Presents real-time telemetry streaming through the entire inference pipeline under normal operating conditions.
* **Key Visual Elements:**
  * **Status Hero Banner:** Displays `ALL SYSTEMS NORMAL` with green indicator, confirming Gateway p95 latency is well within the 200 ms threshold and error rates are under 3%.
  * **KPI Summary Cards:**
    * **System Risk Gauge:** Indicates `0.02` risk probability with a green "HEALTHY" pill.
    * **Anomaly Score:** Reports `0.00` (threshold calibrated at alert budget $\le 0.5$ false alarms/hour).
    * **Early Warning:** `0.00` incident probability within the 120-second rolling horizon.
    * **SLO Status:** `100% HEALTHY` Gateway response latency and zero dropped requests.
  * **Service Topology Graph:** Interactive dependency graph showing the 6 core components (`gateway`, `auth`, `order`, `payment`, `postgres`, `redis`) with real-time request rates, call fan-out links, and node latencies.
  * **Metric Explorer & Log Stream:** Visualizes rolling 5-second scrapes for gateway RPS, database query latency, and Drain-parsed container logs with assigned template cluster identifiers.

---

### Figure 2: Chaos Engineering Console & Fault Injection Controls

![Figure 2: Chaos Engineering Console](screenshots/02_chaos_console.png)

* **View / Component:** Chaos Console Panel (`Live Operations` scrolled view)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Provides an interactive chaos-testing control center to simulate hardware degradations, network partitions, container crashes, and application failures directly against the runtime testbed.
* **Key Visual Elements:**
  * **Fault Catalogue Selectors:** Allows granular configuration of 6 discrete fault types:
    1. `db_slowdown`: Lock contention and slow read queries on PostgreSQL.
    2. `memory_leak`: Leaking memory allocations and GC thrashing on Order service.
    3. `network_latency`: Linux `tc netem` packet delay and drop on Payment service.
    4. `service_crash`: Abrupt container kill (`docker kill`) with crash-loop restart.
    5. `cache_failure`: Redis shutdown triggering cascading cache misses to database.
    6. `cpu_saturation`: Intensive `stress-ng` CPU spinning on Gateway or Auth.
  * **Injection Parameters:** Target component selector, duration slider (60 s to 900 s), and intensity control (20% to 100%).
  * **Benign Traffic Burst Trigger ("Spike"):** Simulates sudden 4× Locust user surges (240 concurrent users) to prove that OpsMind distinguishes benign traffic spikes from genuine system faults.
  * **Automated Scenarios:** Quick-launch buttons for preset multi-stage incidents (`Healthy surge`, `Database slowdown`, `Memory leak`, `Chaos playlist`).

---

### Figure 3: Real-Time Incident State: Early Warning, Degraded Topology & SHAP RCA

![Figure 3: Active Incident State](screenshots/03_live_incident_active_rca.png)

* **View / Component:** Live Operations during active `db_slowdown` injection on `postgres`
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Illustrates OpsMind's real-time detection, early-warning alert propagation, and root-cause localization while a database slowdown fault is active.
* **Key Visual Elements:**
  * **Critical Incident Banner:** Highlights `CRITICAL INCIDENT DETECTED` with active lifecycle state:
    $$\text{Early Warning (LSTM)} \longrightarrow \text{Anomaly Detected} \longrightarrow \text{User SLO Impact} \longrightarrow \text{Static Threshold Breach}$$
  * **Warning Lead Time Indicator:** Shows OpsMind alerted **+90 seconds before** the static threshold alert fired and before user-facing requests degraded.
  * **Topological Degradation:** `postgres` node changes to alert amber/red with an explicit `TOP-1 ROOT CAUSE` badge, while upstream callers (`order`, `auth`) show propagated queueing latency.
  * **SHAP Primary Drivers (Root-Cause Panel):** Ranks the top telemetry features responsible for the incident:
    1. `postgres_active_connections` (+0.38 attribution mass)
    2. `postgres_query_latency_ms` (+0.31 attribution mass)
    3. `order_upstream_db_wait_ms` (+0.19 attribution mass)
  * **Telemetry Charts:** Clear visualization of latency step-change with shaded fault injection bands and automatic alert markers.

---

## 2. Incident Management & Held-Out Test Evaluation

### Figure 4: Incident Registry & Held-Out Episode Lead-Time Benchmarks

![Figure 4: Incidents Management](screenshots/04_incidents_heldout_eval.png)

* **View / Component:** Incidents Page (`http://localhost:8001/incidents`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Evaluates OpsMind on held-out test runs (54 unseen test episodes, 43 of which reached user SLO impact), providing a side-by-side comparison of OpsMind's early-warning system versus conventional Prometheus-style static alert thresholds.
* **Key Visual Elements:**
  * **Executive Metrics Strip:**
    * **Held-out Episodes:** 54 total test runs (43 caused user SLO impact).
    * **OpsMind Early-Warning Recall:** **19 / 43 episodes** successfully alerted before any user SLO impact was recorded (mean lead time: **107 seconds**).
    * **Static Alert Baseline:** Only 9 / 43 alerted before SLO impact, with 30 alerts arriving *late* (after users already experienced errors).
    * **Root-Cause Accuracy:** **Top-1: 94% (51/54)**, **Top-3: 100% (54/54)**.
  * **Episode Ledger Table:** Itemizes every test episode by Run ID, Fault Type, Target Service, Lead Time (`+120s`, `+90s`, `+45s`), Root-Cause Ranking, and an instant `Replay` button to stream the historical run through the live pipeline.

---

## 3. Model Lab — Comprehensive Machine Learning Results

### Figure 5: Executive Success Criteria & Baseline Comparison Scorecard

![Figure 5: Model Lab Scorecard](screenshots/05_model_lab_scorecard.png)

* **View / Component:** Model Lab Executive Summary (`http://localhost:8001/lab#summary`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Presents the formal success-criteria scorecard comparing every module against its baseline on chronologically held-out test runs.
* **Key Quantitative Benchmarks:**
  * **Anomaly Detection F1:** One-Class SVM achieves **F1 0.852** vs Static Threshold baseline **F1 0.742** ($+14.8\%$ relative gain).
  * **False Alarm Rate:** **0.16 false alarms/hour** vs Static Alert baseline **0.61 FA/h** ($73.7\%$ reduction in false alerts).
  * **Early Warning Recall:** LSTM model achieves **44% early warning recall** vs Static Alert baseline **21%**, providing an average advance notice of **107 seconds**.
  * **Root Cause Top-3:** SHAP + Dependency Graph achieves **100% Top-3 recall** (and 94% Top-1) vs Highest Raw CPU baseline **57% Top-3** (and 31% Top-1).
  * **Supervised Classification:** Stacking ensemble reaches **macro-F1 0.979** across 6 fault classes.
  * **Unsupervised Clustering:** K-Means clustering achieves Adjusted Rand Index (**ARI 0.667**) against true fault labels without using ground-truth annotations during training.

---

### Figure 6: Module 2 — Dimensionality Reduction & 2D PCA Latent Space

![Figure 6: Dimensionality Reduction PCA](screenshots/06_model_lab_dimred.png)

* **View / Component:** Dimensionality Reduction Section (`http://localhost:8001/lab#dimred`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Validates dimensionality reduction techniques (Principal Component Analysis vs Gaussian Random Projection) on 299 engineered features, displaying variance preservation, reconstruction error, and 2D state projections.
* **Key Visual Elements:**
  * **2D Latent State Map:** Scatter plot of system states in the first two principal component axes ($PC_1$ vs $PC_2$), with normal operating regions centered and individual fault trajectories radiating outward.
  * **Technique Comparison Bar Chart:**
    * **PCA:** Retains $91.4\%$ cumulative explained variance at 18 components; downstream anomaly F1 remains $0.825$.
    * **Gaussian Random Projection:** Lower downstream F1 ($0.681$) and higher reconstruction MSE, demonstrating PCA's superiority for this telemetry distribution.

---

### Figure 7: Module 3 — Anomaly Detection Under False-Alarm Budget

![Figure 7: Anomaly Detection Comparison](screenshots/07_model_lab_anomaly.png)

* **View / Component:** Unsupervised Anomaly Detection (`http://localhost:8001/lab#anomaly`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Evaluates unsupervised anomaly detectors trained strictly on normal telemetry subject to an operational alert budget of $\le 0.5$ false alarms per hour.
* **Key Quantitative Benchmarks:**
  * **One-Class SVM (RBF Kernel):** **F1 = 0.852**, Precision = 0.881, Recall = 0.824, False Alarms = 0.16/h.
  * **PCA Reconstruction Error:** **F1 = 0.825**, Precision = 0.860, Recall = 0.793, False Alarms = 0.22/h.
  * **Isolation Forest (on PCA):** **F1 = 0.600** (collapses on full 299 features due to curse of dimensionality, but recovers to 0.600 when projected to PCA subspace).
  * **Static Threshold Rules (Prometheus baseline):** **F1 = 0.742**, False Alarms = **0.61/h** (severely penalised by benign nightly backups and traffic spikes).

---

### Figure 8: Modules 4 & 5 — Fault Classification & 6-Class Confusion Matrix

![Figure 8: Fault Classification & Confusion Matrix](screenshots/08_model_lab_classification.png)

* **View / Component:** Supervised Fault Classification (`http://localhost:8001/lab#classification`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Evaluates single classifiers and ensemble methods on classifying 6 discrete microservice failure modes, featuring a full 6×6 normalized confusion matrix.
* **Key Visual Elements:**
  * **Model Comparison:**
    * Stacking Classifier: **Macro-F1 0.979**
    * Logistic Regression (L2): **Macro-F1 0.977**
    * Random Forest (100 estimators): **Macro-F1 0.974**
    * XGBoost (gbtree): **Macro-F1 0.968**
    * Baseline (Majority class): **Macro-F1 0.047**
  * **Normalized Confusion Matrix:** Strong diagonal concentration ($> 0.96$ recall across all 6 classes: `db_slowdown`, `memory_leak`, `network_latency`, `service_crash`, `cache_failure`, `cpu_saturation`), demonstrating minimal cross-fault confusion.
  * **Classifier Veto Pipeline:** Demonstrates how piping the anomaly detector output into the classifier allows the classifier to veto benign blips, reducing false alarms from 0.16/h to 0.00/h.

---

### Figure 9: Module 6 — Unsupervised Clustering & Discovered Failure Signatures

![Figure 9: Unsupervised Clustering](screenshots/09_model_lab_clustering.png)

* **View / Component:** Clustering & Failure Pattern Discovery (`http://localhost:8001/lab#clustering`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Demonstrates unsupervised grouping of abnormal episodes without utilizing ground-truth labels, evaluating alignment with true fault types.
* **Key Visual Elements:**
  * **Algorithm Comparison:**
    * K-Means ($k=12$): **ARI = 0.667**, Silhouette = 0.412, NMI = 0.742
    * Ward Hierarchical: **ARI = 0.647**, Silhouette = 0.398
    * DBSCAN: **ARI = 0.549**, Silhouette = 0.320
    * Random Assignment Baseline: **ARI ≈ 0.000**
  * **Cluster Distribution Scatter:** Visual map of clustered fault partitions showing distinct groupings corresponding to database thread exhaustion, network latency ramps, and memory exhaustion.
  * **Discovered Pattern Cards:** Summarizes key metric deviations characterizing each cluster.

---

### Figure 10: Module 7 — Early-Warning LSTM Lead-Time Dynamics

![Figure 10: Early Warning LSTM](screenshots/10_model_lab_early_warning.png)

* **View / Component:** Early Warning Sequence Model (`http://localhost:8001/lab#warning`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Evaluates sequence models (LSTM, GRU) consuming 5-minute rolling windows to predict impending SLO breaches within a 120-second future horizon.
* **Key Visual Elements:**
  * **Lead-Time Histogram:** Displays the distribution of advance notice provided prior to SLO impact, centered at **mean lead time of 107 seconds**.
  * **Fault Physics Breakdown:**
    * **Gradual Faults:** High advance warning capability (Database Slowdown: **100% recall**, Network Latency: **86% recall**, Memory Leak: **75% recall**).
    * **Abrupt Faults:** Service crash and cache outage occur instantaneously with zero physical precursor; the model honestly reflects low early-warning recall on abrupt crashes while maintaining zero false alarms during healthy spikes.
  * **Model Comparison:** LSTM (19/43, mean lead 107s) vs Gradient Boosting (20/43, mean lead 94s) vs Static Thresholds (9/43 early, 30 late).

---

### Figure 11: Module 8 — Root-Cause Analysis (SHAP + Graph Walk Ablation)

![Figure 11: Root Cause Analysis](screenshots/11_model_lab_root_cause.png)

* **View / Component:** Root-Cause Attribution & Ablations (`http://localhost:8001/lab#rca`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Validates the root-cause ranking pipeline that aggregates SHAP feature attributions across component boundaries and walks the directed dependency call graph.
* **Key Ablation Results:**
  * **SHAP + Dependency Graph Walk (OpsMind):** **Top-1: 94.4%**, **Top-3: 100.0%**
  * **Largest Z-Score Deviation Baseline:** Top-1: 92.6%, Top-3: 98.1%
  * **Highest Raw Metric Baseline (e.g. Max CPU):** **Top-1: 31.5%**, **Top-3: 57.4%**
  * **Engineering Takeaway:** Naively blaming the container with highest CPU fails because victim services (like Gateway) buffer requests and spin CPU during downstream database outages. SHAP + dependency graph traversal correctly directs blame upstream to the dependency root cause.

---

### Figure 12: Real Docker Testbed Campaigns (Campaign 1 vs Campaign 2)

![Figure 12: Real Docker Testbed Evaluation](screenshots/12_model_lab_real_docker.png)

* **View / Component:** Model Lab Real Testbed Evaluation (`http://localhost:8001/lab` Real Docker Data toggle)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Demonstrates model performance when trained and tested on 96 real physical Docker testbed campaign runs (Campaign 1: abrupt faults; Campaign 2: ramped faults with container resets).
* **Key Results on Held-Out Real Telemetry (23 Unseen Real Runs):**
  * **PCA-Reconstruction Anomaly Detection:** **F1 = 0.979** (Precision 0.98, Recall 0.98).
  * **Episode Detection:** **21 / 21 fault episodes** successfully detected (mean detection delay: 30 seconds).
  * **Early Warning on Real Runs:** **3 / 9 SLO-breaching episodes warned in advance** with **mean lead time of 102 seconds** and zero false alarms.
  * **Zero-Shot vs Real-Trained:** Real testbed data exhibits container runtime noise and scrape jitter; training directly on real campaigns improved detection F1 from 0.644 (zero-shot transfer) to **0.979**.

---

## 4. Testbed Architecture & Dataset Provenance

### Figure 13: Distributed Microservices Architecture & Telemetry Pipeline

![Figure 13: Testbed Architecture](screenshots/13_testbed_architecture.png)

* **View / Component:** Testbed Architecture & Telemetry Pipeline (`http://localhost:8001/testbed`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Illustrates the physical architecture of the 10-container Docker Compose testbed, telemetry flow, and automated load injection stack.
* **Key Visual Elements:**
  * **Architecture Schematic (SVG):**
    * **Workload Layer:** Locust distributed traffic generator targeting API Gateway.
    * **Microservices Layer:** `api-gateway`, `auth-service`, `order-service`, `payment-service`.
    * **Storage & Cache Layer:** `postgres-db` and `redis-cache`.
    * **Chaos Layer:** Fault injector using Linux `stress-ng`, `Pumba`, and Docker socket control.
    * **Observability Layer:** Prometheus scraping cAdvisor and custom microservice metrics every 5 seconds.
  * **Telemetry Metric Catalog:** Summary of 43 Prometheus metrics collected across CPU, memory, thread pools, active DB connections, query latency, and network traffic.

---

### Figure 14: Dataset Class Balance & Drain Log Parsing Templates

![Figure 14: Testbed Dataset and Drain](screenshots/14_testbed_dataset_drain.png)

* **View / Component:** Dataset Statistics & Drain-3 Parser (`http://localhost:8001/testbed` scrolled view)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Details the composition of the training and evaluation dataset and documents the online Drain log parser results.
* **Key Visual Elements:**
  * **Dataset Statistics:**
    * Total Experiment Runs: **254 runs** (22 hours of normal operation, 210 fault episodes).
    * Windows Extracted: **45,720 labelled 30-second sliding windows**.
    * Raw Telemetry: **579,140 structured log lines**.
  * **Loghub HDFS Validation:** Results on the public Loghub benchmark showing **99.75% Drain grouping accuracy** (cluster template fidelity).
  * **Drain Template Table:** Itemizes parsed log message regex patterns, template IDs, associated services, and occurrence counts.

---

## 5. Developer Surface & API Architecture

### Figure 15: FastAPI Interactive OpenAPI/Swagger Documentation

![Figure 15: FastAPI Swagger Docs](screenshots/15_api_swagger_docs.png)

* **View / Component:** Interactive API Documentation (`http://localhost:8001/docs`)
* **Resolution:** 3360 × 2100 (Retina 2×)
* **Functionality:**
  Provides complete interactive documentation and testing interfaces for all REST endpoints and WebSocket protocols exposed by the OpsMind FastAPI service.
* **Key Exposed Endpoints:**
  * `GET /api/health` & `GET /api/status`: System state, model load verification, simulation clock.
  * `GET /api/config`: Microservices topology, dependency edges, metrics catalogue, SLO thresholds.
  * `POST /api/faults/inject`, `POST /api/faults/spike`, `POST /api/faults/clear`: Chaos console endpoints.
  * `POST /api/session/control`, `POST /api/session/speed`, `POST /api/session/replay`: Session pacing, time-dilation, and historical replay.
  * `POST /api/predict`: Stateless online inference scoring raw scrape batches.
  * `GET /api/benchmarks` & `GET /api/dataset`: Analytical data for the Model Lab and Incidents registries.
  * `WS /ws/live`: Real-time bi-directional WebSocket connection for snapshot streaming and 10s tick updates.

---

## Summary of Results

| Evaluation Metric | Baseline (Conventional Ops) | OpsMind AI (Simulated Test) | OpsMind AI (Real Docker Test) |
|---|---|---|---|
| **Anomaly Detection F1** | 0.742 (Static Thresholds) | **0.852** (One-Class SVM) | **0.979** (PCA Reconstruction) |
| **False Alarms per Hour** | 0.61 FA/h | **0.16 FA/h** (0.00 with veto) | 0.65 FA/h (with veto) |
| **Early Warning Lead Time** | 9/43 warned early (30 late) | **19/43 warned early** (mean lead **107 s**) | **3/9 warned early** (mean lead **102 s**) |
| **Classification Macro-F1** | 0.047 (Majority Class) | **0.979** (Stacking Ensemble) | **0.965** (Random Forest) |
| **Root-Cause Top-1 Accuracy** | 31.5% (Highest Raw Metric) | **94.4%** (SHAP + Graph Walk) | **95.2%** (SHAP + Graph Walk) |
| **Root-Cause Top-3 Accuracy** | 57.4% (Highest Raw Metric) | **100.0%** (SHAP + Graph Walk) | **100.0%** (SHAP + Graph Walk) |
| **Drain Log Grouping Accuracy** | N/A | **99.75%** (Loghub HDFS) | **99.81%** (Real Docker Logs) |
