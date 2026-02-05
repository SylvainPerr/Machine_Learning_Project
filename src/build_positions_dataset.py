from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import chess
from tqdm import tqdm


# -----------------------
# Config
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "chess_filtered_sample_300k.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "chess_positions_c20.csv"

TARGET_FULLMOVE = 20  # coup 20 (fullmove number)
MIN_FULLMOVES_REQUIRED = 20  # on exige d'atteindre le coup 20

# Colonnes qu'on garde en entrée
KEEP_COLS = [
    "Result",
    "WhiteElo",
    "BlackElo",
    "TimeControl",
    "Termination",
    "AN",
]

# regex pour retirer numéros de coups (ex: "1." "23."), ellipses, et résultats éventuels
MOVE_NUM_RE = re.compile(r"\d+\.(\.\.)?")
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
NAG_RE = re.compile(r"\$\d+")


def parse_base_seconds(tc: str) -> int | None:
    """TimeControl '300+5' -> 300."""
    try:
        base = tc.split("+", 1)[0]
        return int(base)
    except Exception:
        return None


def result_to_label(res: str) -> int | None:
    """
    Map:
      1-0 -> 2 (white win)
      1/2-1/2 -> 1 (draw)
      0-1 -> 0 (black win)
    """
    if res == "1-0":
        return 2
    if res == "1/2-1/2":
        return 1
    if res == "0-1":
        return 0
    return None


def clean_an_moves(an: str) -> list[str]:
    """
    Transforme la colonne AN (type '1. e4 e5 2. Nf3 Nc6 ...')
    en liste de coups SAN ["e4","e5","Nf3","Nc6",...]
    """
    if not isinstance(an, str) or not an.strip():
        return []

    s = an.strip()

    # retire annotations de commentaires éventuels { ... } et ; ...
    s = re.sub(r"\{[^}]*\}", " ", s)
    s = re.sub(r";[^\n]*", " ", s)

    # retire NAG $1, $2...
    s = NAG_RE.sub(" ", s)

    # retire numéros de coups
    s = MOVE_NUM_RE.sub(" ", s)

    # split tokens
    tokens = [t.strip() for t in s.split() if t.strip()]

    # retire résultats éventuels à la fin
    tokens = [t for t in tokens if t not in RESULT_TOKENS]

    # retire ellipses "..." parfois présentes
    tokens = [t for t in tokens if t != "..."]

    return tokens


def fen_at_fullmove(an: str, target_fullmove: int) -> str | None:
    """
    Reconstruit la partie et retourne la FEN juste après la fin du coup `target_fullmove`
    (i.e. après que Noir a joué au coup target_fullmove).
    Si la partie n'atteint pas ce coup, retourne None.
    """
    moves = clean_an_moves(an)
    if not moves:
        return None

    board = chess.Board()
    # nombre de demi-coups requis pour atteindre la fin du coup target_fullmove
    # ex: coup 20 complet => 40 plies
    required_plies = target_fullmove * 2

    ply = 0
    for san in moves:
        try:
            board.push_san(san)
            ply += 1
        except Exception:
            # si un SAN est invalide (bruit), on abandonne cette partie
            return None

        if ply >= required_plies:
            return board.fen()

    return None


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Input not found: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading: {IN_PATH}")
    df = pd.read_csv(IN_PATH, usecols=[c for c in KEEP_COLS if c in pd.read_csv(IN_PATH, nrows=0).columns])

    # Sécurité : supprimer résultats invalides
    df["label"] = df["Result"].map(result_to_label)
    df = df[df["label"].notna()].copy()

    # Feature simple: base_seconds
    df["base_seconds"] = df["TimeControl"].apply(parse_base_seconds)

    # Extraction FEN au coup 20
    fens = []
    kept_rows = []

    print("[INFO] Extracting FEN at fullmove 20 (this can take a while)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        fen = fen_at_fullmove(row["AN"], TARGET_FULLMOVE)
        if fen is None:
            continue
        fens.append(fen)
        kept_rows.append(idx)

    out = df.loc[kept_rows, ["label", "WhiteElo", "BlackElo", "base_seconds", "Termination"]].copy()
    out["fen_c20"] = fens

    # Nettoyage final: enlever base_seconds manquants
    out = out[out["base_seconds"].notna()].copy()

    print(f"[INFO] Final dataset shape: {out.shape}")
    out.to_csv(OUT_PATH, index=False)
    print(f"[DONE] Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
