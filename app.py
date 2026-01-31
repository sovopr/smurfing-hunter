import streamlit as st
import pandas as pd
import json
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.graph_objects as go
import random

# 1. Page Config
st.set_page_config(layout="wide", page_title="Smurfing Hunter: AI-Forensics Dashboard")

# CSS Hack (Crucial for Layout)
st.markdown("""
    <style>
        /* Force the Streamlit iframe to be tall */
        iframe { height: 80vh !important; }
        /* Remove whitespace at top */
        .block-container { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("Smurfing Hunter: AI-Forensics Dashboard")

# Load Data
@st.cache_data
def load_data():
    preds = pd.read_csv('final_predictions.csv')
    txns = pd.read_csv('transactions.csv')
    processed_stats = pd.read_csv('processed_graph_data.csv') 
    with open('threshold_metrics.json', 'r') as f:
        metrics = json.load(f)
        
    preds = preds.merge(processed_stats[['Wallet_ID', 'Flow_Ratio']], on='Wallet_ID', how='left')
    return preds, txns, metrics

preds_df, txns_df, metrics_json = load_data()

# Build Global Graph
G_full = nx.from_pandas_edgelist(txns_df, 'Source', 'Target', edge_attr='Amount', create_using=nx.DiGraph())

# AI Forensic Report Generator
def generate_police_report(wallet_id, preds_df, G):
    row = preds_df[preds_df['Wallet_ID'] == wallet_id].iloc[0]
    prob = row['GNN_Prob']
    ratio = row['Flow_Ratio']
    
    # Neighborhood Context
    neighbors = list(G.neighbors(wallet_id)) + list(G.predecessors(wallet_id))
    neighbor_rows = preds_df[preds_df['Wallet_ID'].isin(neighbors)]
    num_illicit = len(neighbor_rows[neighbor_rows['True_Label'] == 1])
    
    risk_level = "CRITICAL" if prob > 0.9 else "HIGH" if prob > 0.7 else "MODERATE"
    
    # Logic for behavior type
    if 0.9 <= ratio <= 1.1:
        behavior = "PASSTHROUGH (Mule)"
    elif ratio < 0.9:
        behavior = "ACCUMULATION (Boss)"
    else:
        behavior = "DISPERSION (Smurf)"
        
    report = f"""
    ### 👮 AI FORENSIC REPORT
    
    **CASE ID**: `REF-{random.randint(10000, 99999)}`
    **SUBJECT**: `{wallet_id}`
    
    ---
    
    **RISK ASSESSMENT**: **{risk_level}**
    **SUSPICION SCORE**: `{prob:.4f}`
    
    **BEHAVIOR ANALYTICS**:
    - **Pattern**: `{behavior}`
    - **Flow Ratio**: `{ratio:.2f}`
    - **Bad Actors**: `{num_illicit}` confirmed illicit connections.
    
    **GNN CONCLUSION**:
    Subject exhibits structural properties consistent with {behavior.lower()} operations. Immediate audit recommended.
    """
    return report

# 2. Sidebar Controls
st.sidebar.header("Global Controls")
threshold = st.sidebar.slider("Suspicion Threshold", 0.0, 1.0, 0.85, 0.05)
st.sidebar.markdown("---")

st.sidebar.title("🕵️ Case Files")

# Top Risk Selection
# Filter based on slider
top_risk_df = preds_df[preds_df['GNN_Prob'] >= threshold].sort_values(by='GNN_Prob', ascending=False)
# If too many, maybe limit? User didn't ask to limit, only filter. But "dropdown" usually implies filtered list.
# Earlier code limited to head(20). User prompt: "Before showing the 'Select Target Subject' dropdown, filter the available options based on the slider."
# "filtered_wallets = top_risk_df[top_risk_df['GNN_Prob'] >= threshold]"
# I should probably keep a reasonable limit or sort them.
# Let's keep the sort by Risk.
top_risk_ids = top_risk_df['Wallet_ID'].tolist()

selected_wallet = st.sidebar.selectbox("Select Target Subject", ["Select Subject..."] + top_risk_ids)

if selected_wallet != "Select Subject...":
    report_text = generate_police_report(selected_wallet, preds_df, G_full)
    st.sidebar.markdown(report_text)
else:
    st.sidebar.info("Select a high-risk subject to open their case file.")

# 3. Main Interface
tab1, tab2, tab3 = st.tabs(["🕸️ Network Inspector", "🌊 Flow Tracer", "📊 Metrics"])

with tab1:
    if selected_wallet == "Select Subject...":
        st.warning("⚠️ Select a subject from the sidebar to initialize the Graph Inspector.")
        # Minimal View
        st.subheader("High Priority Targets")
        st.dataframe(top_risk_df[['Wallet_ID', 'GNN_Prob', 'Flow_Ratio', 'True_Label']].head(10))
    else:
        st.header(f"Investigation: {selected_wallet}")
        
        # Ego Graph Logic
        if selected_wallet in G_full:
            G_ego = nx.ego_graph(G_full, selected_wallet, radius=2)
            
            nodes = []
            edges = []
            ego_nodes_list = list(G_ego.nodes())
            node_attrs = preds_df[preds_df['Wallet_ID'].isin(ego_nodes_list)].set_index('Wallet_ID')
            
            for node_id in ego_nodes_list:
                prob = 0.0
                lbl = 0
                if node_id in node_attrs.index:
                    prob = node_attrs.loc[node_id, 'GNN_Prob']
                    lbl = node_attrs.loc[node_id, 'True_Label']
                
                # Visual Styles
                if node_id == selected_wallet:
                    color = "#FFD700" # GOLD
                    size = 40
                elif prob >= threshold:
                    color = "#FF0000" # RED
                    size = 25
                elif lbl == 1:
                    color = "#0000FF" # BLUE
                    size = 20
                else:
                    color = "grey" # Low suspicion / Clean
                    size = 15
                    
                nodes.append(Node(id=node_id, 
                                  label=node_id[:5], 
                                  size=size, 
                                  color=color, 
                                  title=f"{node_id}\nProb: {prob:.4f}"))
                                  
            for src, dst, data in G_ego.edges(data=True):
                amt = data.get('Amount', 0)
                edges.append(Edge(source=src, 
                                  target=dst, 
                                  color="#AAAAAA", 
                                  label=f"{amt:.1f}"))
            
            # THE ANTI-BUNCHING CONFIG
            config = Config(width=2000, # Force wide canvas
                            height=950, # Force tall canvas
                            directed=True, 
                            physics=True, 
                            hierarchical=False,
                            nodeHighlightBehavior=True,
                            solver='barnesHut',
                            gravitationalConstant=-10000, # MASSIVE REPULSION
                            centralGravity=0.1, # RELAXED GRAVITY (Lets them fly apart)
                            springLength=250, # LONG SPRINGS
                            springConstant=0.04,
                            damping=0.09,
                            avoidOverlap=0.2
                            )
            
            agraph(nodes=nodes, edges=edges, config=config)
        else:
            st.error("Target missing from interaction graph.")

with tab2:
    st.header("Money Flow Tracer (Peeling Analysis)")
    
    if selected_wallet != "Select Subject..." and selected_wallet in G_full:
        # Sankey Logic
        # We want to see IN -> SELECTED -> OUT
        # Get neighbors
        in_edges = list(G_full.in_edges(selected_wallet, data=True))
        out_edges = list(G_full.out_edges(selected_wallet, data=True))
        
        # Build Sankey Mapping
        all_nodes = set()
        all_nodes.add(selected_wallet)
        for u, v, d in in_edges: all_nodes.add(u)
        for u, v, d in out_edges: all_nodes.add(v)
        
        node_list = list(all_nodes)
        node_map = {id: i for i, id in enumerate(node_list)}
        
        sources = []
        targets = []
        values = []
        link_colors = []
        
        # Incoming Flows
        for u, v, d in in_edges:
            sources.append(node_map[u])
            targets.append(node_map[v])
            values.append(d['Amount'])
            link_colors.append("rgba(0, 255, 0, 0.4)") # Green in
            
        # Outgoing Flows
        for u, v, d in out_edges:
            sources.append(node_map[u])
            targets.append(node_map[v])
            values.append(d['Amount'])
            # Highlight 'Peeling' (Small amounts)
            if d['Amount'] < 10:
                link_colors.append("rgba(255, 0, 0, 0.6)") # RED LINK for Smurfing
            else:
                link_colors.append("rgba(0, 0, 255, 0.4)") # Blue for transfer
                
        # Node Labels & Colors
        labels = []
        colors = []
        for id in node_list:
            labels.append(id[:6])
            if id == selected_wallet: colors.append("gold")
            elif id in top_risk_ids: colors.append("red")
            else: colors.append("grey")
            
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = labels,
              color = colors
            ),
            link = dict(
              source = sources,
              target = targets,
              value = values,
              color = link_colors
          ))])
          
        fig.update_layout(title_text=f"Capital Flow for {selected_wallet}", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("🔴 Red Links indicate potential 'Smurfing' (Low value peeling transactions).")
    else:
        st.info("Select a subject to trace money flow.")

with tab3:
    st.header("System Metrics")
    stats_05 = metrics_json.get("threshold_0.5", {})
    recall_05 = stats_05.get("recall", 0) * 100
    st.metric(label="Recall (Detection Rate)", value=f"{recall_05:.1f}%")
    
    categories = ['0.5', '0.7', '0.9']
    precisions = [metrics_json[f"threshold_{t}"]["precision"] for t in categories]
    recalls = [metrics_json[f"threshold_{t}"]["recall"] for t in categories]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=categories, y=precisions, name='Precision', marker_color='#37536d'))
    fig.add_trace(go.Bar(x=categories, y=recalls, name='Recall', marker_color='#1a76ff'))
    
    fig.update_layout(title='Metrics vs Threshold', barmode='group')
    st.plotly_chart(fig, use_container_width=True)
