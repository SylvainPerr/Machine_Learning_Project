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

        param_grid = {
            "n_estimators": [200],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 2],
        }

        models["rf_gridsearch"] = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring="f1_weighted",
            cv=cv,
            n_jobs=1,            
            verbose=2,
        )



    # RandomizedSearchCV pour Random Forest
    if "rf_randomsearch" in RUN_MODELS:

        cv = StratifiedKFold(
            n_splits=2,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        rf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=2
        )

        param_dist = {
            "n_estimators": randint(100),      
            "max_depth": [10, 20],
            "min_samples_leaf": randint(1, 2),     
        }

        models["rf_randomsearch"] = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=3,                 
            scoring="f1_weighted",
            cv=cv,
            verbose=2,
            random_state=RANDOM_STATE,
            n_jobs=1                  
        )
    
        
        
    # -----------------------
    # Train + Eval
    # -----------------------

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                n_jobs=-1,
                class_weight="balanced",
            )),
        ]
    )

        print(f"[INFO] Evaluation {name} on test set")

        if hasattr(model, "best_params_"):
            print("[INFO] Best params:", model.best_params_)
            print("[INFO] Best CV score (f1_weighted):", model.best_score_)
            best_model = model.best_estimator_
            y_pred = best_model.predict(X_test)
        else:
            y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))


if __name__ == "__main__":
    main()
