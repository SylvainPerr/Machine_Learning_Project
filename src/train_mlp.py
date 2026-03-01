import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess
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
        self.X_spatial = torch.tensor(X_spatial, dtype=torch.float32).view(-1, 768)
        self.X_global = torch.tensor(X_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_spatial[idx], self.X_global[idx], self.y[idx]

class ChessMLP(nn.Module):
    def __init__(self):
        super(ChessMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(768 + 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x_spatial, x_global):
        x = torch.cat((x_spatial, x_global), dim=1)
        return self.net(x)

def main():
    df = pd.read_csv(DATA_PATH)
    y = df['label'].values
    symbols = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    sym_idx = {s: i for i, s in enumerate(symbols)}

    def fen_to_flat(fen):
        board = chess.Board(fen.split()[0])
        flat = np.zeros(768, dtype=np.float32)
        for sq, piece in board.piece_map().items():
            flat[sym_idx[piece.symbol()] * 64 + sq] = 1.0
        return flat

    X_spatial = np.stack(df['fen_c20'].apply(fen_to_flat).values)
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
    
    X_sp_train, X_sp_test, X_gl_train, X_gl_test, y_train, y_test = train_test_split(
        X_spatial, X_global, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_dataset = ChessDataset(X_sp_train, X_gl_train, y_train)
    test_dataset = ChessDataset(X_sp_test, X_gl_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessMLP().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        for sp, gl, labels in train_loader:
            sp, gl, labels = sp.to(device), gl.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(sp, gl), labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} terminée")
            
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sp, gl, labels in test_loader:
            sp, gl, labels = sp.to(device), gl.to(device), labels.to(device)
            _, preds = torch.max(model(sp, gl), 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print(classification_report(all_labels, all_preds, digits=4))
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()
