import pandas as pd
import numpy as np
import torch
import chess
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "chess_positions_c20.csv"
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.001

class ChessDataset(Dataset):
    def __init__(self, X_spatial, X_global, y):
        self.X_spatial = torch.tensor(X_spatial, dtype=torch.float32).view(-1, 12, 8, 8)
        self.X_global = torch.tensor(X_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_spatial[idx], self.X_global[idx], self.y[idx]

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8 + 3, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 3)

    def forward(self, x_spatial, x_global):
        x = torch.relu(self.bn1(self.conv1(x_spatial)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, x_global), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.out(x)

def main():
    df = pd.read_csv(DATA_PATH)
    y = df['label'].values
    
    print("[INFO] Encodage des positions FEN en matrices 3D (cela va prendre ~1 minute)...")
    symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    sym_idx = {s: i for i, s in enumerate(symbols)}

    def fen_to_tensor(fen):
        board = chess.Board(fen.split()[0])
        spatial = np.zeros((12, 8, 8), dtype=np.float32)
        for sq, piece in board.piece_map().items():
            spatial[sym_idx[piece.symbol()], sq // 8, sq % 8] = 1.0
        return spatial

    
    X_spatial = np.stack(df['fen_c20'].apply(fen_to_tensor).values)
    
    
    def material_features_from_fen(fen: str) -> np.ndarray:
        piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}
        board = chess.Board(fen.split()[0])
        mw = mb = 0.0
        for p in board.piece_map().values():
            v = piece_values[p.symbol().lower()]
            if p.color:
                mw += v
            else:
                mb += v
        return np.array([mw, mb, mw - mb], dtype=np.float32)

    X_global = np.stack(df["fen_c20"].astype(str).apply(material_features_from_fen).values)
    
    print(f"[DEBUG] X_spatial shape: {X_spatial.shape} (Attendu: ~258000, 12, 8, 8)")
    print(f"[DEBUG] X_global shape: {X_global.shape} (Attendu: ~258000, 3)")
    
    X_sp_train, X_sp_test, X_gl_train, X_gl_test, y_train, y_test = train_test_split(
        X_spatial, X_global, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_dataset = ChessDataset(X_sp_train, X_gl_train, y_train)
    test_dataset = ChessDataset(X_sp_test, X_gl_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    weights = total_samples / (3.0 * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    print(f"[INFO] Poids appliqués : Noir={weights[0]:.2f}, Nul={weights[1]:.2f}, Blanc={weights[2]:.2f}")
    
    model = ChessCNN().to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("[INFO] Démarrage de l'entraînement...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for sp, gl, labels in train_loader:
            sp, gl, labels = sp.to(device), gl.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sp, gl)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

    #Évaluation test
        
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sp, gl, labels in test_loader:
            sp, gl, labels = sp.to(device), gl.to(device), labels.to(device)
            outputs = model(sp, gl)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n[INFO] Evaluation du test")
    print(classification_report(all_labels, all_preds, digits=4))
    print(f"accuracy : {accuracy_score(all_labels, all_preds):.4f}")
    print("\nMatrice de Confusion:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()
