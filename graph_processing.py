import pandas as pd
import networkx as nx
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from datetime import datetime

# Configuration
INPUT_TXNS = 'transactions.csv'
INPUT_LABELS = 'labels.csv'
OUTPUT_NODE_FEATURES = 'node_features.csv'
OUTPUT_EDGE_INDEX = 'edge_index.pt'
OUTPUT_PROCESSED_DATA = 'processed_graph_data.csv'

def load_data():
    print("Loading data...")
    df_txns = pd.read_csv(INPUT_TXNS)
    df_labels = pd.read_csv(INPUT_LABELS)
    
    # Convert timestamps
    df_txns['Timestamp'] = pd.to_datetime(df_txns['Timestamp'])
    
    return df_txns, df_labels

def build_graph(df_txns, all_wallets):
    print("Building graph...")
    G = nx.DiGraph()
    G.add_nodes_from(all_wallets)
    
    # Add edges with attributes
    # NetworkX multigraph might be better for multiple txns, but DiGraph weighs edges by default or we can aggregate.
    # For GNN, we often want the raw edge index. But for feature extraction (centrality), aggregating weights is useful.
    # Let's aggregate for centrality/pagerank, but keep raw structure for Edge Index.
    
    # Aggregate for NetworkX analysis
    weighted_edges = df_txns.groupby(['Source', 'Target'])['Amount'].sum().reset_index()
    for _, row in weighted_edges.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Amount'])
        
    return G

def calculate_flow_features(df_txns, all_wallets):
    print("Calculating Flow features...")
    # Group by Source (Out) and Target (In)
    total_out = df_txns.groupby('Source')['Amount'].sum()
    total_in = df_txns.groupby('Target')['Amount'].sum()
    
    features = pd.DataFrame(index=all_wallets)
    features['Total_Out'] = total_out
    features['Total_In'] = total_in
    features.fillna(0, inplace=True)
    
    # Flow Ratio: Out / In
    # Handle division by zero
    features['Flow_Ratio'] = features.apply(
        lambda row: row['Total_Out'] / row['Total_In'] if row['Total_In'] > 0 else 0.0, axis=1
    )
    
    return features

def calculate_temporal_features(df_txns, all_wallets):
    print("Calculating Temporal features...")
    # Avg time diff between incoming and SUBSEQUENT outgoing
    # This is expensive. Simplified approach: Avg Incoming Time vs Avg Outgoing Time? 
    # Or just average gap?
    # Better: For each node, sort all txns (in and out). 
    # If type changes from In -> Out, measure delta.
    
    # Optimized approach:
    # 1. Collect all timestamps per node, marked as 'in' or 'out'
    node_activity = defaultdict(list)
    
    # It allows iterating once?
    # Let's do a vectorized approach or per-group if possible.
    # Given the scale (25k txns), iteration is okay.
    
    txns_sorted = df_txns.sort_values('Timestamp')
    
    for _, row in txns_sorted.iterrows():
        t = row['Timestamp']
        node_activity[row['Target']].append(('in', t))
        node_activity[row['Source']].append(('out', t))
        
    avg_diffs = {}
    
    for wallet in all_wallets:
        activity = node_activity.get(wallet, [])
        if not activity:
            avg_diffs[wallet] = 0.0
            continue
            
        # Filter sequences of In -> ... -> Out
        # Find all 'in' times and 'out' times
        ins = [t for d, t in activity if d == 'in']
        outs = [t for d, t in activity if d == 'out']
        
        if not ins or not outs:
            avg_diffs[wallet] = 0.0
            continue
            
        # For every Out, find the closest PREVIOUS In? 
        # Or just average latency?
        # Simple heuristic: Mean(Outs) - Mean(Ins) could be negative if order is mixed.
        # Strict definition: "Avg time between incoming and outgoing"
        
        # Let's take the mean of valid (Out - Closest_Previous_In) pairs
        # Or simple average gap if we assume pass-through
        
        delays = []
        # Sort just in case aggregation mixed it up (though we iterated sorted df)
        # activity is already sorted by time
        
        last_in = None
        for direction, t in activity:
            if direction == 'in':
                last_in = t
            elif direction == 'out':
                if last_in:
                    diff = (t - last_in).total_seconds()
                    if diff > 0:
                        delays.append(diff)
                    # Reset last_in? Usually pass-through consumes inputs. 
                    # Let's keep last_in for split payments (one in, multiple out).
                    # But if multiple ins, then one out? 
                    # Keeping last_in is a reasonable heuristic for "latest funding".
        
        if delays:
            avg_diffs[wallet] = np.mean(delays)
        else:
            avg_diffs[wallet] = 0.0
            
    return pd.Series(avg_diffs, name='Avg_Time_Diff')

def main():
    df_txns, df_labels = load_data()
    all_wallets = df_labels['Wallet_ID'].unique()
    
    # 1. Build Graph
    G = build_graph(df_txns, all_wallets)
    
    # 2. Features
    flow_df = calculate_flow_features(df_txns, all_wallets)
    time_series = calculate_temporal_features(df_txns, all_wallets)
    
def calculate_centrality(G):
    print("Calculating Centrality (Approx Betweenness k=100)...")
    # Degree Centrality (Fast)
    deg = nx.degree_centrality(G)
    
    # Betweenness Centrality (Approximate)
    # k=100 pivots for speed
    bet = nx.betweenness_centrality(G, k=100)
    
    return deg, bet

def check_cycles(G):
    print("Checking Cycles (Limit length 6)...")
    nodes_in_cycles = set()
    
    # Use length_bound=6 to avoid exponential explosion
    try:
        # Note: length_bound available in NetworkX 2.8+
        # This returns a generator of cycles
        cycles = nx.simple_cycles(G, length_bound=6)
        count = 0
        for cycle in cycles:
            nodes_in_cycles.update(cycle)
            count += 1
            if count % 1000 == 0:
                print(f"Found {count} cycles so far...")
    except TypeError:
        print("Warning: length_bound argument not supported by this NetworkX version. Fallback to SCC heuristic.")
        # Fallback: Just mark all nodes in non-trivial SCCs (over-approximation but fast)
        sccs = nx.strongly_connected_components(G)
        for scc in sccs:
            if len(scc) > 1:
                nodes_in_cycles.update(scc)
    
    return nodes_in_cycles

def main():
    df_txns, df_labels = load_data()
    all_wallets = df_labels['Wallet_ID'].unique()
    
    # 1. Build Graph
    G = build_graph(df_txns, all_wallets)
    
    # 2. Features
    flow_df = calculate_flow_features(df_txns, all_wallets)
    time_series = calculate_temporal_features(df_txns, all_wallets)
    
    deg_centrality, bet_centrality = calculate_centrality(G)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    print("Calculating TrustRank (PageRank with Illicit seeds)...")
    wallet_label_map = dict(zip(df_labels['Wallet_ID'], df_labels['Label']))
    personalization = {w: wallet_label_map.get(w, 0) for w in all_wallets}
    
    if sum(personalization.values()) == 0:
        print("Warning: No illicit seeds found. Using standard PageRank.")
        trust_rank = nx.pagerank(G)
    else:
        try:
            trust_rank = nx.pagerank(G, personalization=personalization)
        except ZeroDivisionError:
             # Handle disconnected components or other issues
            trust_rank = nx.pagerank(G)

    nodes_in_cycles = check_cycles(G)
                
    # 3. Assemble Features
    print("Assembling Data...")
    final_df = pd.DataFrame(index=all_wallets)
    final_df['In_Degree'] = pd.Series(in_degree)
    final_df['Out_Degree'] = pd.Series(out_degree)
    final_df['Total_In'] = flow_df['Total_In']
    final_df['Flow_Ratio'] = flow_df['Flow_Ratio']
    final_df['Avg_Time_Diff'] = time_series
    final_df['TrustRank'] = pd.Series(trust_rank)
    final_df['Degree_Centrality'] = pd.Series(deg_centrality)
    final_df['Betweenness_Centrality'] = pd.Series(bet_centrality)
    final_df['Is_In_Cycle'] = final_df.index.to_series().apply(lambda x: 1 if x in nodes_in_cycles else 0)
    
    # Normalize
    cols_to_norm = ['In_Degree', 'Out_Degree', 'Total_In', 'Flow_Ratio', 'Avg_Time_Diff', 'TrustRank', 'Degree_Centrality', 'Betweenness_Centrality', 'Is_In_Cycle']
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(final_df[cols_to_norm])
    
    df_normalized = pd.DataFrame(normalized_features, columns=cols_to_norm, index=final_df.index)
    
    # Export Node Features
    df_normalized.to_csv(OUTPUT_NODE_FEATURES)
    print(f"Saved {OUTPUT_NODE_FEATURES}")
    
    # Export Edge Index
    wallet_to_idx = {w: i for i, w in enumerate(all_wallets)}
    
    src_indices = [wallet_to_idx[r['Source']] for _, r in df_txns.iterrows()]
    dst_indices = [wallet_to_idx[r['Target']] for _, r in df_txns.iterrows()]
    
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    torch.save(edge_index, OUTPUT_EDGE_INDEX)
    print(f"Saved {OUTPUT_EDGE_INDEX} with shape {edge_index.shape}")
    
    # Export Processed Graph Data (Human Readable)
    export_df = final_df.copy()
    export_df['Wallet_ID'] = export_df.index
    export_df['True_Label'] = export_df['Wallet_ID'].apply(lambda x: wallet_label_map.get(x, 0))
    
    # Reorder
    cols = ['Wallet_ID', 'TrustRank', 'Flow_Ratio', 'True_Label', 'Is_In_Cycle', 'Avg_Time_Diff', 'Betweenness_Centrality']
    export_df[cols].to_csv(OUTPUT_PROCESSED_DATA, index=False)
    print(f"Saved {OUTPUT_PROCESSED_DATA}")
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
