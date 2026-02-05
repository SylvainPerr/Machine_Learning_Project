from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import chess

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -----------------------
# Config
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "chess_positions_c20.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# -----------------------
# FEN -> vector encoder
# -----------------------

PIECE_TO_IDX = {
    "P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5,
    "p": 6, "n": 7, "b": 8, "r": 9, "q": 10, "k": 11,
}


def fen_to_vector(fen: str) -> np.ndarray:
    """
    Encode une FEN en vecteur 8x8x12 -> flatten.
    One-hot des pièces, sans info side-to-move / roques pour baseline.
    """
    board = chess.Board(fen)
    x = np.zeros((8, 8, 12), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        idx = PIECE_TO_IDX[piece.symbol()]
        x[row, col, idx] = 1.0

    return x.reshape(-1)


# -----------------------
# Main
# -----------------------

def main() -> None:
    print("[INFO] Loading data:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    print("[INFO] Encoding boards...")
    X = np.stack(df["fen_c20"].apply(fen_to_vector).values)
    y = df["label"].values

    print("[INFO] X shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                multi_class="multinomial",
                n_jobs=-1,
            )),
        ]
    )

    print("[INFO] Training Logistic Regression...")
    pipe.fit(X_train, y_train)

    print("[INFO] Evaluation on test set")
    y_pred = pipe.predict(X_test)

    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
