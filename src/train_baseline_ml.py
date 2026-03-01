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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# -----------------------
# Config
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "chess_positions_c20.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Choisis ce que tu veux lancer
# options : "logreg", "dt", "rf", "rf_gridsearch", "rf_randomsearch"
RUN_MODELS = [""]  

USE_SMALL_TRAIN = True
TRAIN_SMALL_SIZE = 50000  # ou 0.25 pour une proportion


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

        #Valeur du matériel pour les blancs et les noirs
        value = PIECE_VALUE[piece.symbol()]
        if piece.color:  # True = white
            material_white += value
        else:
            material_black += value

    #Ajout différence de matériel 
    diff_material = material_white - material_black
    flat = x.reshape(-1)
    extra_features = np.array(
        [material_white, material_black, diff_material],
       dtype=np.float32
    )

    return np.concatenate([flat, extra_features])


# -----------------------
# Main
# -----------------------

def main() -> None:
    print("[INFO] Loading data:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    print("[INFO] Encoding boards...")
    X_path = PROJECT_ROOT / "data" / "processed" / "X_c20.npy"
    y_path = PROJECT_ROOT / "data" / "processed" / "y_c20.npy"

    if X_path.exists() and y_path.exists():
        print("[INFO] Loading cached features...")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print("[INFO] Encoding boards (first run can take a while)...")
        X = np.stack(df["fen_c20"].astype(str).apply(fen_to_vector).values)
        y = df["label"].values
        np.save(X_path, X)
        np.save(y_path, y)
        print("[INFO] Saved cache:", X_path.name, y_path.name)

        print("[INFO] X shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Sous-échantillon du train 
    if USE_SMALL_TRAIN and len(y_train) > TRAIN_SMALL_SIZE:
        X_train_small, _, y_train_small, _ = train_test_split(
            X_train, y_train,
            train_size=TRAIN_SMALL_SIZE,
            stratify=y_train,
            random_state=RANDOM_STATE
        )
        X_fit, y_fit = X_train_small, y_train_small
        print(f"[INFO] Using SMALL train for fitting: {X_fit.shape}")
    else:
        X_fit, y_fit = X_train, y_train
        print(f"[INFO] Using FULL train for fitting: {X_fit.shape}")


    models = {}

    #LOG Regression
    if "logreg" in RUN_MODELS:
        models["logreg"] = Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=300, solver="lbfgs"))
        ])
        

    #Decision Tree
    if "dt" in RUN_MODELS:
        models["decision_tree"] = DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }
    
    for name, model in models.items():
        print(f"\n[INFO] Training {name} ...")
        model.fit(X_train, y_train)

        print(f"[INFO] Evaluation {name} on test set")
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred, digits=4))
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))


if __name__ == "__main__":
    main()

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
