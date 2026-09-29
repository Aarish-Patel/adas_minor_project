#!/usr/bin/env python3
"""
train_model.py — Deep Intent Learning Pipeline.

This script manages the supervised training of the GRU-based Driver Intent 
Classifier. It processes temporal feature windows into a sequence model 
capable of predicting near-future driving behaviors.

Pipeline Stages:
1. Feature Loading: Reads sliding-window kinematics from CSV.
2. Temporal Modeling: Initializes a 2-layer Gated Recurrent Unit (GRU).
3. Supervised Optimization: Minimizes Cross-Entropy loss over 4 intent classes.
4. Validation & Export: Generates performance metrics and serializes 
   the TorchScript/Weights for real-time inference.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ─── Hyperparameters ───
NUM_FEATURES = 7    # [yaw_rate, accel, steer_rate, vel_deriv, lane_dev, effort, min_dist]
WINDOW_SIZE  = 10   # Temporal lookback window (samples)

# Intent Label Mapping
LABEL_MAP = {
    'aggressive':   0,
    'inconsistent': 1,
    'late_braking': 2,
    'defensive':    3,
}
CLASS_NAMES = ['Aggressive', 'Lane Change', 'Sudden Brake', 'Defensive']


class ADASDataset(Dataset):
    """
    Custom Dataset for temporal ADAS features.
    Reshapes flat CSV rows into (Window, Features) tensors.
    """
    def __init__(self, csv_file, window_size=WINDOW_SIZE, num_features=NUM_FEATURES):
        data = pd.read_csv(csv_file)
        self.window_size  = window_size
        self.num_features = num_features

        # Partition features and target
        feature_cols = [c for c in data.columns if c != 'label']
        self.X = data[feature_cols].values.astype(np.float32)
        # Reshape to (Batch, Sequence, Features)
        self.X = self.X.reshape(-1, window_size, num_features)

        y_str  = data['label'].values
        self.y = np.array([LABEL_MAP.get(str(l).strip(), 3) for l in y_str], dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IntentModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) architecture for sequence classification.
    """
    def __init__(self, input_size=NUM_FEATURES, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        
        # Core Temporal Layer
        self.gru  = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # Decision Head
        self.fc  = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        # Take the hidden state of the LAST sequence element for classification
        out = self.fc(out[:, -1, :])
        return out


def plot_diagnostics(train_losses, val_losses, val_accuracies):
    """Generates visual confirmation of model convergence."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r--', label='Validation Loss')
    ax1.set_title('Temporal Convergence (Loss)'); ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_title('Classification Performance (Accuracy)'); ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.savefig('performance_curves.png', dpi=150)
    plt.close()


def main():
    """Main Orchestrator for the training session."""
    csv_path = 'dataset/adas_features.csv'
    if not os.path.exists(csv_path):
        print("[ERROR] Dataset missing. Run 'dataset_exporter.py' first.")
        # Minimal mock generation to allow script to be showcased
        return

    # Data Pipeline
    dataset = ADASDataset(csv_path)
    train_size = int(0.85 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # Hardware Acceleration check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = IntentModel().to(device)

    # Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 40
    train_losses, val_losses, val_accs = [], [], []

    print(f'>>> Starting Intent Learning on {device} ({epochs} epochs)')
    
    for epoch in range(epochs):
        # ─── Training Pass ───
        model.train()
        running_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_b.size(0)
        train_losses.append(running_loss / train_size)

        # ─── Validation Pass ───
        model.eval()
        v_loss, correct = 0.0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                v_loss += criterion(outputs, y_b).item() * X_b.size(0)
                correct += (outputs.argmax(1) == y_b).sum().item()
        
        val_losses.append(v_loss / val_size)
        val_accs.append(correct / val_size)

        if (epoch + 1) % 5 == 0:
            print(f'[*] Epoch {epoch+1:02d} | T-Loss: {train_losses[-1]:.4f} | V-Acc: {val_accs[-1]:.2%}')

    # Finalization
    model_save_path = 'intent_model.pt'
    torch.save(model.state_dict(), model_save_path)
    plot_diagnostics(train_losses, val_losses, val_accs)
    
    print(f'>>> Training Complete. Unified weights saved to {model_save_path}')


if __name__ == '__main__':
    main()
