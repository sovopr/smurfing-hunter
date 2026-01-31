import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import numpy as np

# Configuration
INPUT_NODE_FEATURES = 'node_features.csv'
INPUT_EDGE_INDEX = 'edge_index.pt'
INPUT_LABELS = 'processed_graph_data.csv'
OUTPUT_MODEL = 'gnn_model.pth'
OUTPUT_METRICS = 'threshold_metrics.json'
OUTPUT_LOG = 'training_log.json'
OUTPUT_PREDICTIONS = 'final_predictions.csv'

HIDDEN_CHANNELS = 64
DROPOUT_RATE = 0.3
EPOCHS = 250
CLASS_WEIGHTS = [1.0, 30.0]  # Clean, Illicit

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        # 3-Layer GCN
        self.conv1 = GCNConv(num_features, HIDDEN_CHANNELS)
        self.conv2 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = GCNConv(HIDDEN_CHANNELS, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=DROPOUT_RATE, training=self.training)
        
        x = self.conv3(x, edge_index)
        # Logits usually required for CrossEntropyLoss
        return x

def load_data():
    print("Loading data...")
    # Features (Indices are preserved from CSV)
    df_features = pd.read_csv(INPUT_NODE_FEATURES, index_col=0)
    x = torch.tensor(df_features.values, dtype=torch.float)
    
    # Edge Index
    edge_index = torch.load(INPUT_EDGE_INDEX)
    
    # Labels
    df_labels = pd.read_csv(INPUT_LABELS)
    # Ensure alignment: Wallet_ID in labels should match index of features
    # Assuming standard sorting was preserved or index handling in graph_processing
    # Graph processing exported index=final_df.index, so they should match order
    # Let's verify or map.
    # df_labels has 'Wallet_ID' column. df_features index is Wallet_ID.
    # We can align just to be sure
    df_labels.set_index('Wallet_ID', inplace=True)
    df_labels = df_labels.reindex(df_features.index)
    
    y = torch.tensor(df_labels['True_Label'].values, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Split
    print("Splitting data 80/20...")
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=y.numpy(), random_state=42)
    
    data.train_mask = torch.tensor(np.isin(indices, train_idx), dtype=torch.bool)
    data.test_mask = torch.tensor(np.isin(indices, test_idx), dtype=torch.bool)
    
    return data, df_features.index.tolist()

def main():
    # Hardware
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using device: MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: CUDA")
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
        
    data, wallet_ids = load_data()
    data = data.to(device)
    
    model = GCN(num_features=data.num_features, num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Aggressively penalize missing criminals
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS).to(device))
    
    print(f"Starting training for {EPOCHS} epochs...")
    training_log = []
    
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Loss only on training nodes
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        training_log.append({'epoch': epoch+1, 'loss': loss_val})
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1:03d}, Loss: {loss_val:.4f}')
            
    # Save log
    with open(OUTPUT_LOG, 'w') as f:
        json.dump(training_log, f, indent=4)
        
    # Evaluation
    print("Evaluating...")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1) # [Num_Nodes, 2]
        
        # Explicitly move to CPU and convert to numpy to avoid MPS hangs with sklearn
        probs_cpu = probs.detach().cpu()
        illicit_probs = probs_cpu[:, 1].numpy()
        
        labels_cpu = data.y.detach().cpu()
        true_labels = labels_cpu.numpy()
        
        mask_cpu = data.test_mask.detach().cpu()
        test_mask = mask_cpu.numpy()
        
        # Test set metrics
        test_probs = illicit_probs[test_mask]
        test_labels = true_labels[test_mask]
        
        thresholds = [0.5, 0.7, 0.9]
        metrics = {}
        
        for t in thresholds:
            preds = (test_probs >= t).astype(int)
            p = precision_score(test_labels, preds, zero_division=0)
            r = recall_score(test_labels, preds, zero_division=0)
            f1 = f1_score(test_labels, preds, zero_division=0)
            metrics[f"threshold_{t}"] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1)
            }
            
    # Save Metrics
    with open(OUTPUT_METRICS, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {OUTPUT_METRICS}")
    print(json.dumps(metrics, indent=2))
        
    print("\nSanity Check: Top 10 Wallets by Illicit Probability (Global)")
    
    # Store in DF
    df_res = pd.DataFrame({
        'Wallet_ID': wallet_ids,
        'GNN_Prob': illicit_probs,
        'True_Label': true_labels
    })
    
    # Predict Class based on 0.5 default for the CSV export
    df_res['Predicted_Class'] = (df_res['GNN_Prob'] >= 0.5).astype(int)
    
    top_10 = df_res.sort_values(by='GNN_Prob', ascending=False).head(10)
    print(top_10[['Wallet_ID', 'GNN_Prob', 'True_Label']].to_string(index=False))
    
    # Export Predictions
    print('Saving files...')
    df_res.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Saved predictions to {OUTPUT_PREDICTIONS}")

if __name__ == "__main__":
    main()
