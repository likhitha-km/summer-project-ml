# Delhivery - Graph-Based Network Intelligence

Modeling India's largest logistics network as a directed graph to produce smarter ETA predictions, surface bottleneck hubs, and generate data-backed operational recommendations.

---

## The Problem

Delhivery's existing routing engine (OSRM) assumes clean traffic and shortest paths. In reality, facility dwell time, congestion, and route-type constraints cause actual delivery times to deviate significantly. There was no systematic way to identify which hubs and corridors were responsible, or by how much.

## The Approach

Rather than treating each trip in isolation, the network is modeled as a directed weighted graph where facilities are nodes and corridors are edges, with edge weights capturing the median actual-to-OSRM delay ratio per corridor. Graph structure is then used to explain and predict delays that trip-level features alone cannot capture.

---

## What's Inside

**Graph Construction**
- Directed graph of 1,590 hubs and 2,506 corridors built from 144k+ trip segments
- Edge weights stratified by route type and time of day
- Node attributes: in/out degree, betweenness centrality, PageRank, clustering coefficient

**Bottleneck Audit**
- Composite bottleneck score combining centrality, volume, and delay magnitude
- 1,853 chronically delayed corridors flagged (median transit time >20% over OSRM)
- Interactive Plotly network graph with delay risk coloring

**Graph-Enhanced ETA Prediction**
| Model | MAE (min) | Within-15% Accuracy |
|---|---|---|
| Baseline XGBoost (trip features) | 10.34 | 36.9% |
| Graph XGBoost (+ Node2Vec + graph metrics) | 9.91 | 39.0% |
| Dual-Branch Keras NN (stacked) | 10.16 | 41.1% |

Graph advantage: **4.2% MAE reduction, +4.2pp within-15% accuracy**

**Node2Vec — Built from Scratch**
The `node2vec` package threw a binary incompatibility error with the environment's NumPy version. Rather than simplifying the approach, Node2Vec was reimplemented entirely in NumPy: biased random walks with configurable p/q parameters and skipgram training with negative sampling. Embeddings are L2-normalized and used as source/destination features for both ETA prediction and route-type classification.

**FTL vs Carting Decision Framework**
- XGBoost classifier using graph position (betweenness, PageRank) alongside trip features
- 85.8% classification accuracy
- Time-cost trade-off surface quantifying when FTL is economically preferable
- Estimated ₹1,966L/month in savings from optimal route-type assignments

**Network Operations Strategy Memo**
- Top 5 bottleneck hubs with monthly revenue impact and recommended interventions
- Top 3 hubs account for ₹21.8 Cr annual impact with sub-1-month upgrade payback
- Written for a non-technical operations leader, not a data scientist

---

## Stack

Python, NetworkX, XGBoost, TensorFlow/Keras, NumPy, Pandas, Plotly, Matplotlib, Scikit-learn

---

## Structure

```
delhivery/
├── delhivery.py               # Full pipeline: data, graph, models, memo
├── delivery_data.csv          # Raw trip segment data (144k rows)
├── hub_metrics.csv            # Graph metrics per hub
├── edge_metrics.csv           # Corridor-level delay stats
├── top5_bottlenecks.csv       # Top bottleneck hubs with impact estimates
├── network_graph.html         # Interactive network visualization
├── route_decision.html        # FTL vs Carting decision boundary
├── strategy_memo.txt          # Operations strategy memo
└── outputs/
    ├── bottleneck_hub_rankings.png
    ├── delay_heatmap.png
    ├── model_benchmark_metrics.png
    ├── predicted_vs_actual.png
    └── revenue_impact.png
```

---

## How to Run

```bash
pip install networkx scikit-learn xgboost matplotlib seaborn plotly tensorflow scipy
python delhivery.py
```

Or open in Google Colab and run top to bottom. Node2Vec training takes approximately 3-5 minutes.

---

## Key Results

- Graph-enhanced model outperforms the OSRM-only baseline on both MAE and business accuracy
- Route classifier achieves 85.8% accuracy, identifying ₹1,966L/month in suboptimal assignments
- Top 3 hub upgrades recoverable in under 1 month at moderate intervention targets
- Strategy memo structured for direct use by a Head of Network Operations
