
# streamlit_app.py  — Run: streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import xgboost as xgb

PATH = "/content/drive/MyDrive/Summer Project"

st.set_page_config(page_title="Delhivery Network Intelligence", layout="wide", page_icon="📦")
st.title("Delhivery - Graph Network Intelligence Dashboard")
st.markdown("Real-time delay risk scores, bottleneck hubs, and route recommendations")

@st.cache_data
def load_data():
    hubs = pd.read_csv(f"{PATH}/hub_metrics.csv")
    edges = pd.read_csv(f"{PATH}/edge_metrics.csv")
    top5 = pd.read_csv(f"{PATH}/top5_bottlenecks.csv")
    return hubs, edges, top5

hub_metrics, edge_df, top5 = load_data()

st.sidebar.header("Filters")
min_volume = st.sidebar.slider("Min Hub Volume", 0, 5000, 100)
show_chronic = st.sidebar.checkbox("Show Chronic Corridors Only", False)

filtered_hubs  = hub_metrics[hub_metrics["total_volume"] >= min_volume]
filtered_edges = edge_df[edge_df["trip_count"] >= 10]
if show_chronic:
    filtered_edges = filtered_edges[filtered_edges["breach_rate"] > 0.2]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Hubs", f"{len(filtered_hubs):,}")
col2.metric("Active Corridors", f"{len(filtered_edges):,}")
col3.metric("Chronic Corridors", f"{(filtered_edges['breach_rate']>0.2).sum():,}")
col4.metric("Avg Network Delay", f"{filtered_edges['median_delay_ratio'].mean():.2f}x OSRM")

st.subheader("Top 5 Bottleneck Hubs")
display_cols = ["hub","breach_trips","breach_rate","total_impact_monthly","bottleneck_score"]
available = [c for c in display_cols if c in top5.columns]
st.dataframe(top5[available].style.background_gradient(cmap="RdYlGn_r", subset=["bottleneck_score"]), use_container_width=True)

st.subheader("🗺 Network Delay Risk Map")
top_n = st.slider("Top N hubs by volume", 50, 300, 150)
top_nodes = filtered_hubs.nlargest(top_n, "total_volume")["hub"].tolist()

G_tmp = nx.DiGraph()
for _, r in filtered_edges.iterrows():
    if r["source_center"] in top_nodes and r["destination_center"] in top_nodes:
        G_tmp.add_edge(r["source_center"], r["destination_center"], weight=r["median_delay_ratio"])

if G_tmp.number_of_nodes() > 0:
    pos = nx.spring_layout(G_tmp, seed=42, k=0.5)
    ex, ey = [], []
    for u,v in G_tmp.edges():
        x0,y0=pos[u]; x1,y1=pos[v]
        ex+=[x0,x1,None]; ey+=[y0,y1,None]
    scores = {r["hub"]: r["bottleneck_score"] for _,r in filtered_hubs.iterrows()}
    fig_net = go.Figure(
        [go.Scatter(x=ex,y=ey,mode="lines",line=dict(width=0.4,color="#555"),hoverinfo="none"),
         go.Scatter(x=[pos[n][0] for n in G_tmp.nodes()],
                    y=[pos[n][1] for n in G_tmp.nodes()],
                    mode="markers", hovertext=list(G_tmp.nodes()),
                    hoverinfo="text",
                    marker=dict(size=10, color=[scores.get(n,0) for n in G_tmp.nodes()],
                                colorscale="RdYlGn_r", showscale=True))],
        layout=go.Layout(showlegend=False,
                         xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                         yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                         height=500, margin=dict(b=0,l=0,r=0,t=0)))
    st.plotly_chart(fig_net, use_container_width=True)

st.subheader("Highest-Risk Corridors")
risk_cols = ["source_center","destination_center","breach_rate","trip_count","median_delay_ratio"]
avail_risk = [c for c in risk_cols if c in filtered_edges.columns]
st.dataframe(
    filtered_edges[avail_risk]
    .nlargest(20, "breach_rate")
    .style.background_gradient(cmap="Reds", subset=["breach_rate"]),
    use_container_width=True
)
