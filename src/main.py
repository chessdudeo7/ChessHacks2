# src/main.py

import torch
import chess
import random
import os
import sys
from .utils import chess_manager, GameContext

# ----------------------------------------------------
# Add training folder (where model and dataset code live)
# ----------------------------------------------------
TRAINING_DIR = os.path.join(os.path.dirname(__file__), "..", "training")
TRAINING_DIR = os.path.abspath(TRAINING_DIR)

sys.path.insert(0, TRAINING_DIR)

# Import your training code
from chess_dataset import fen_to_tensors
from move_vocab import MoveVocab
from model import ChessNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------------------
# Load the latest trained vocab
# ----------------------------------------------------
VOCAB_PATH = os.path.join(TRAINING_DIR, "move_vocab.pt")
# --- FIX ---
# Set weights_only=False because we are loading a pickled 
# class object (MoveVocab), not just a state dictionary.
vocab = torch.load(VOCAB_PATH, map_location=DEVICE, weights_only=False)
print("Loaded vocab with", len(vocab.idx_to_move), "moves.")

# ----------------------------------------------------
# Load the latest trained model
# ----------------------------------------------------
MODEL_PATH = os.path.join(TRAINING_DIR, "chess_model.pt")
model = ChessNet(output_size=len(vocab.idx_to_move))
# This line is fine as-is because it loads a state_dict (weights)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE) 
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

print("Model loaded successfully! Ready to play chess.\n")

# ----------------------------------------------------
# AI Move Logic
# ----------------------------------------------------
@chess_manager.entrypoint
def play_move(ctx: GameContext):
    legal_moves = list(ctx.board.generate_legal_moves())
    if not legal_moves:
        ctx.logProbabilities({})
        return None

    # Convert FEN to spatial and scalar tensors
    x_spatial, x_scalar = fen_to_tensors(ctx.board.fen())
    x_spatial = x_spatial.unsqueeze(0).to(DEVICE)  # (1,12,8,8)
    x_scalar = x_scalar.unsqueeze(0).to(DEVICE)    # (1,5)

    # Predict move probabilities
    with torch.no_grad():
        logits = model(x_spatial, x_scalar).squeeze(0)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # Map probabilities to legal moves
    move_probs = {}
    for move in legal_moves:
        uci = move.uci()
        if uci in vocab.move_to_idx:
            idx = vocab.move_to_idx[uci]
            move_probs[move] = float(probs[idx])

    # Fallback: uniform distribution if none match
    if not move_probs:
        for move in legal_moves:
            move_probs[move] = 1.0 / len(legal_moves)

    # Log probabilities for debugging / API
    ctx.logProbabilities({m.uci(): p for m, p in move_probs.items()})

    # Choose move based on probabilities
    moves = list(move_probs.keys())
    weights = list(move_probs.values())
    chosen_move = random.choices(moves, weights, k=1)[0]

    return chosen_move

@chess_manager.reset
def reset_func(ctx: GameContext):
    pass