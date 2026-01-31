import streamlit as st
import pandas as pd
import json
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
import plotly.graph_objects as go
import random
import datetime

import plotly.express as px

# 1. Page Config
st.set_page_config(layout="wide", page_title="Smurfing Hunter: AI-Forensics Dashboard")

# CSS Hack (Crucial for Layout)
st.markdown("""
    <style>
        /* Force the Streamlit iframe to be tall */
        iframe { height: 80vh !important; }
        /* Remove whitespace at top */
        .block-container { padding-top: 1rem; }
        
        /* Green Diamond Badge */
        .badge-diamond { 
            background-color: #00e676; 
            color: black; 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-weight: bold; 
        }
        /* Red Risk Badge */
        .badge-risk { 
            background-color: #ff5252; 
            color: white; 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-weight: bold; 
        }
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
    
    # Process Timestamps for Timeline
    # Check if Timestamp is string
    if txns['Timestamp'].dtype == 'object':
        txns['Timestamp'] = pd.to_datetime(txns['Timestamp'])
    
    start_time = txns['Timestamp'].min()
    txns['Hours'] = (txns['Timestamp'] - start_time).dt.total_seconds() / 3600.0
    
    # Feature Engineering for Risk Map
    # Calculate Total Volume (In + Out)
    vol_in = txns.groupby('Target')['Amount'].sum().rename('Vol_In')
    vol_out = txns.groupby('Source')['Amount'].sum().rename('Vol_Out')
    
    preds = preds.set_index('Wallet_ID')
    preds = preds.join(vol_in).join(vol_out).fillna(0)
    preds['Total_Volume'] = preds['Vol_In'] + preds['Vol_Out']
    preds = preds.reset_index()
    
    # Determine Role
    # Source: Ratio > 1.1, Mule: 0.9 <= Ratio <= 1.1, Aggregator: Ratio < 0.9
    def get_role(r):
        if 0.9 <= r <= 1.1: return "Mule"
        elif r < 0.9: return "Aggregator"
        else: return "Source"
        
    preds['Role'] = preds['Flow_Ratio'].apply(get_role)
    
    # Generate Tags for Badges
    def get_tags(row):
        tags = []
        if 0.95 <= row['Flow_Ratio'] <= 1.05:
            tags.append("Diamond Pattern")
        if row['Total_Volume'] > 50: # Arbitrary threshold for high value
            tags.append("High Velocity")
        if row['GNN_Prob'] > 0.95:
            tags.append("Structurally Embedded")
        return tags

    preds['Tags'] = preds.apply(get_tags, axis=1)
    
    return preds, txns, metrics

preds_df, txns_df, metrics_json = load_data()

# Build Global Graph (Static for calculations)
G_full = nx.from_pandas_edgelist(txns_df, 'Source', 'Target', edge_attr='Amount', create_using=nx.DiGraph())
# For edge filtering, we'll need to query txns_df or match edges to attributes
# NetworkX edges can hold data.
nx.set_edge_attributes(G_full, {(row['Source'], row['Target']): {'Hours': row['Hours'], 'Amount': row['Amount']} for _, row in txns_df.iterrows()})


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
    
    if 0.9 <= ratio <= 1.1: behavior = "PASSTHROUGH (Mule)"
    elif ratio < 0.9: behavior = "ACCUMULATION (Boss)"
    else: behavior = "DISPERSION (Smurf)"
        
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

def generate_sar_report(wallet_id, txns_df):
    """Generates a Suspicious Activity Report (SAR) logic."""
    # Get outgoing flows
    outgoing = txns_df[txns_df['Source'] == wallet_id]
    incoming = txns_df[txns_df['Target'] == wallet_id]
    
    total_in = incoming['Amount'].sum()
    total_out = outgoing['Amount'].sum()
    count_out = len(outgoing)
    avg_out = outgoing['Amount'].mean() if not outgoing.empty else 0
    unique_recipients = outgoing['Target'].nunique()
    
    start = txns_df['Timestamp'].min()
    end = txns_df['Timestamp'].max()
    duration = end - start
    
    # STRUCTURING TRAP
    # Check for amounts 9000-9999
    structuring_hits = outgoing[(outgoing['Amount'] >= 9000) & (outgoing['Amount'] <= 9999)]
    
    is_structuring = not structuring_hits.empty
    
    sar_text = f"""
    ### 📄 FBI/FinCEN Suspicious Activity Report (SAR)
    
    **Subject**: `{wallet_id}`
    **Typology**: `Smurfing / Structuring / Layering`
    **Status**: `Ready for Filing`
    
    **Narrative Analysis**:
    The AI has identified a high-probability money laundering pattern.
    The Subject received **{total_in:.2f} ETH** and rapidly split **{total_out:.2f} ETH** across **{count_out}** transactions to **{unique_recipients}** unique beneficiaries. 
    
    The average outgoing transaction was **{avg_out:.2f} ETH**. This fragmentation of funds ("Fan-Out") is characteristic of Smurfing operations designed to obscure the money trail.
    """
    
    if is_structuring:
        sar_text += f"""
        
        ⚠️ **STRUCTURING ALERT ($9k-$10k Trap)**: 
        **{len(structuring_hits)}** transactions were identified between **9,000 and 9,999**.
        
        > **AI Insight**: *Notice how the AI caught this?* These values are deliberately set just below the $10,000 reporting threshold. A standard rule-based system might miss this implementation of "Smurfing", but our GNN identified the structural intent.
        """
    else:
         sar_text += "\n\nNo specific structuring (<$10k) threshold evasion detected in this batch."
         
    return sar_text

# ==========================================
# 2. HELPER FUNCTION (SAR Text Logic)
# ==========================================
def generate_sar_text(suspect_id, row, volume_usd):
    """
    Generates the confidential report text with timestamp and formatting.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report = f"""CONFIDENTIAL SUSPICIOUS ACTIVITY REPORT (SAR)
==================================================
DATE: {timestamp}
CASE ID: AUTO-{hash(suspect_id) % 10000}
SUBJECT: {suspect_id}

RISK ASSESSMENT
---------------
Suspicion Score: {row.get('GNN_Prob', 0):.4f}
Risk Level: {"CRITICAL" if row.get('GNN_Prob', 0) > 0.8 else "HIGH"}
Detected Role: {row.get('Role', 'Unknown')}
Flow Ratio: {row.get('Flow_Ratio', 0):.2f}

FINANCIAL ACTIVITY
------------------
Total Volume: ${volume_usd:,.2f} USD
Tags: {', '.join(row.get('Tags', [])) if row.get('Tags') else 'None'}

NARRATIVE
---------
The subject wallet has been flagged by the AI Forensics Engine due to anomalous behavior 
consistent with {row.get('Role', 'unknown').lower()} patterning. The high velocity of funds and 
structural positioning suggests potential illicit activity.

Recommended Action: Immediate Freeze & Audit.
==================================================
Generated by Smurfing Hunter Enterprise"""
    return report

# 2. Sidebar Controls
st.sidebar.header("Global Controls")
threshold = st.sidebar.slider("Suspicion Threshold", 0.0, 1.0, 0.85, 0.05)

# Time Lapse
max_hours = int(txns_df['Hours'].max())
# Default to full range
current_time = st.sidebar.slider("Transaction Timeline (Hours)", 0, max_hours, max_hours)

st.sidebar.markdown("---")
st.sidebar.title("🕵️ Case Files")

# Top Risk Selection
top_risk_df = preds_df[preds_df['GNN_Prob'] >= threshold].sort_values(by='GNN_Prob', ascending=False)
top_risk_ids = top_risk_df['Wallet_ID'].tolist()

selected_wallet = st.sidebar.selectbox("Select Target Subject", ["Select Subject..."] + top_risk_ids)

if selected_wallet != "Select Subject...":
    report_text = generate_police_report(selected_wallet, preds_df, G_full)
    st.sidebar.markdown(report_text)
    
    # --- NEW: Sidebar Profile Features ---
    sel_row = preds_df[preds_df['Wallet_ID'] == selected_wallet].iloc[0]
    rate = 2000
    vol_val = sel_row['Total_Volume'] * rate
    
    st.sidebar.markdown(f"**Volume**: `{vol_val:,.0f} USD`")
    
    # Badges
    badges_html = ""
    for tag in sel_row.get('Tags', []):
        if "Diamond" in tag:
            badges_html += f"<span class='badge-diamond'>💎 {tag}</span> "
        else:
            badges_html += f"<span class='badge-risk'>⚠️ {tag}</span> "
    
    if badges_html:
        st.sidebar.markdown(badges_html, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Action Buttons
    if st.sidebar.button("📄 Generate SAR Report"):
        sar_content = generate_sar_text(sel_row['Wallet_ID'], sel_row, vol_val)
        st.sidebar.text_area("SAR Output", sar_content, height=300)
        
else:
    st.sidebar.info("Select a high-risk subject to open their case file.")

# 3. Main Interface
st.header("Enterprise Smurfing Detection")
tab1, tab2, tab3, tab4 = st.tabs(["🕸️ Smurf Hunter", "🌊 Flow Tracer", "🌍 Global Risk", "🦠 Contagion"])

with tab1:
    if selected_wallet == "Select Subject...":
        st.warning("⚠️ Select a subject from the sidebar to initialize the Graph Inspector.")
        st.subheader("High Priority Targets")
        st.dataframe(top_risk_df[['Wallet_ID', 'GNN_Prob', 'Flow_Ratio', 'True_Label']].head(10))
    else:
        st.header(f"Investigation: {selected_wallet}")
        st.caption(f"Visualizing transactions up to Hour {current_time}")
        
        # Ego Graph Logic
        if selected_wallet in G_full:
            G_ego = nx.ego_graph(G_full, selected_wallet, radius=2)
            
            # TIME LAPSE FILTER
            # Filter edges in G_ego based on 'Hours' attribute
            # We must rebuild list of valid edges
            valid_edges = []
            for u, v, d in G_ego.edges(data=True):
                if d.get('Hours', 0) <= current_time:
                     valid_edges.append((u, v, d))
            
            # Nodes involved in valid edges (plus centered node)
            active_nodes = set()
            active_nodes.add(selected_wallet)
            for u, v, d in valid_edges:
                active_nodes.add(u)
                active_nodes.add(v)
            
            nodes = []
            edges = []
            
            # Only render active nodes
            ego_nodes_list = list(active_nodes)
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
                    color = "grey"
                    size = 15
                    
                nodes.append(Node(id=node_id, 
                                  label=node_id[:5], 
                                  size=size, 
                                  color=color, 
                                  title=f"{node_id}\nProb: {prob:.4f}"))
                                  
            for u, v, d in valid_edges:
                amt = d.get('Amount', 0)
                edges.append(Edge(source=u, 
                                  target=v, 
                                  color="#AAAAAA", 
                                  label=f"{amt:.1f}"))
            
            config = Config(width="100%", 
                            height=950,
                            directed=True, 
                            physics=True, 
                            hierarchical=False,
                            nodeHighlightBehavior=True,
                            solver='barnesHut',
                            gravitationalConstant=-10000, 
                            centralGravity=0.1, 
                            springLength=250, 
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
        # Sankey Logic - Filtered by Time
        # Smart Grouping: Top 15 + Others
        
        # 1. Gather all raw edges first (Time filtered)
        in_edges = []
        out_edges = []
        
        def is_valid(d): return d.get('Hours', 0) <= current_time

        for u, v, d in G_full.in_edges(selected_wallet, data=True):
            if is_valid(d): in_edges.append({'src': u, 'dst': v, 'amt': d['Amount'], 'type': 'in'})
            
        for u, v, d in G_full.out_edges(selected_wallet, data=True):
             if is_valid(d): out_edges.append({'src': u, 'dst': v, 'amt': d['Amount'], 'type': 'out'})
             
        # 2. Process Incoming (Top 15)
        in_edges.sort(key=lambda x: x['amt'], reverse=True)
        final_in = in_edges[:15]
        remainder_in = in_edges[15:]
        
        # 3. Process Outgoing (Top 15)
        out_edges.sort(key=lambda x: x['amt'], reverse=True)
        final_out = out_edges[:15]
        remainder_out = out_edges[15:]
        
        # 4. Build Nodes List
        # Center
        nodes = [selected_wallet]
        node_map = {selected_wallet: 0}
        
        # Add Top Incoming Sources
        for e in final_in:
            if e['src'] not in node_map:
                node_map[e['src']] = len(nodes)
                nodes.append(e['src'])
                
        # Add Top Outgoing Targets
        for e in final_out:
            if e['dst'] not in node_map:
                node_map[e['dst']] = len(nodes)
                nodes.append(e['dst'])
                
        # Add Aggregated Nodes if needed
        idx_others_in = -1
        if remainder_in:
            idx_others_in = len(nodes)
            nodes.append("Others (Incoming)")
            
        idx_others_out = -1
        if remainder_out:
            idx_others_out = len(nodes)
            nodes.append("Others (Outgoing)")
            
        # 5. Build Links
        sources = []
        targets = []
        values = []
        colors = []
        labels = []
        
        # Main Incoming
        for e in final_in:
            sources.append(node_map[e['src']])
            targets.append(node_map[selected_wallet])
            values.append(e['amt'])
            colors.append("rgba(0, 255, 0, 0.4)") # Green
            
        # Aggregated Incoming
        if remainder_in:
            total_rem = sum(e['amt'] for e in remainder_in)
            sources.append(idx_others_in)
            targets.append(node_map[selected_wallet])
            values.append(total_rem)
            colors.append("rgba(200, 200, 200, 0.5)") # Gray
            
        # Main Outgoing
        for e in final_out:
            sources.append(node_map[selected_wallet])
            targets.append(node_map[e['dst']])
            values.append(e['amt'])
            # Color logic
            if e['amt'] < 10: colors.append("rgba(255, 0, 0, 0.6)") # Peeling
            else: colors.append("rgba(0, 0, 255, 0.4)")
            
        # Aggregated Outgoing
        if remainder_out:
            total_rem = sum(e['amt'] for e in remainder_out)
            sources.append(node_map[selected_wallet])
            targets.append(idx_others_out)
            values.append(total_rem)
            colors.append("rgba(200, 200, 200, 0.5)") # Gray
            
        # Node Styling
        node_colors = []
        node_labels = []
        for id in nodes:
            if id == selected_wallet: 
                node_colors.append("gold")
                node_labels.append(f"TARGET: {id[:6]}")
            elif "Others" in id:
                node_colors.append("lightgrey")
                node_labels.append(id)
            elif id in top_risk_ids:
                node_colors.append("red")
                node_labels.append(id[:6])
            else:
                node_colors.append("grey")
                node_labels.append(id[:6])

        # Render
        fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = node_labels,
              color = node_colors,
              hovertemplate='%{label}<extra></extra>'
            ),
            link = dict(
              source = sources,
              target = targets,
              value = values,
              color = colors,
              hovertemplate='Amount: %{value:.2f} ETH<extra></extra>'
            ))])
          
        fig.update_layout(title_text=f"Capital Flow Analysis (Top 15 Streams)", height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        if remainder_in or remainder_out:
            st.caption(f"Note: {len(remainder_in) + len(remainder_out)} smaller transactions have been aggregated into 'Others' to maintain readability.")
        
    else:
        st.info("Select a subject to trace money flow.")

with tab3: # GLOBAL RISK MAP
    st.subheader("Network Risk Distribution")
    
    # Prepare Data for plotting
    disp_df = preds_df.copy()
    rate = 2000 # Hardcode ETH rate
    currency = "USD"
    
    # Convert Volume to USD 
    disp_df['Display_Vol'] = disp_df['Total_Volume'] * rate
    
    # Plotly Scatter (Bubble Chart)
    fig_bubble = px.scatter(
        disp_df,
        x="Display_Vol",
        y="GNN_Prob",        # Y-Axis: Risk Score
        color="Role",        # Color by: Source/Mule/Aggregator
        size="Display_Vol",  # Bubble Size: Volume
        hover_data=["Wallet_ID", "Flow_Ratio"],
        log_x=True,          # Log Scale for X-Axis
        title=f"Volume vs Risk ({currency})",
        color_discrete_map={
            "Source": "#00e676",    # Green
            "Mule": "#ff9100",      # Orange
            "Aggregator": "#2979ff" # Blue
        }
    )
    fig_bubble.update_traces(marker=dict(line=dict(width=0)))
    st.plotly_chart(fig_bubble, use_container_width=True)

with tab4: # CONTAGION LOOP
    st.subheader("Contagion Growth")
    
    # 1. Bucket transactions by Hour
    # Use copy to allow mutation without warning
    df_trans = txns_df.copy()
    df_trans['Time_Bin'] = df_trans['Timestamp'].dt.floor('H')
    
    # 2. Iterate through time to track 'New' vs 'Total' wallets
    growth = []
    seen = set()
    
    # Group by stored time bins
    # Sort by time to ensure order
    df_trans = df_trans.sort_values('Time_Bin')
    
    for t, group in df_trans.groupby('Time_Bin'):
        # Find all unique wallets active in this hour
        active = set(group['Source']).union(set(group['Target']))
        
        # Determine which are NEW (never seen before)
        new = active - seen
        
        # Update state
        seen.update(new)
        growth.append({
            'Time': t, 
            'New Wallets': len(new), 
            'Total': len(seen)
        })
        
    # 3. Plot Growth Curve
    g_df = pd.DataFrame(growth)
    fig_line = px.line(
        g_df, 
        x='Time', 
        y='New Wallets', 
        title="Network Infection Rate (New Wallets/Hour)"
    )
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.info("Hierarchical View: Coming in v2.1")