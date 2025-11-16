import os
import torch
import modal

# -----------------------------
# Modal App Configuration
# -----------------------------

stub = modal.App(name="chess-training-job")

# --- FIX ---
# modal 1.2.2 removed `modal.Mount`.
# You must now add mounts directly to the Image definition.
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "python-chess", "tqdm", "numpy")
    .add_local_dir(
        "./training", remote_path="/root/src"  # Mounts 'training' dir to '/root/src'
    )
    .add_local_dir(
        "./training/pgn", remote_path="/root/pgn_data"  # Mounts PGN data
    )
)
# -----------

model_volume = modal.Volume.from_name("chess-model-vol", create_if_missing=True)


# Define paths *inside the container*
CONTAINER_MODEL_PATH = "/model_data/chess_model.pt"
CONTAINER_VOCAB_PATH = "/model_data/move_vocab.pt"
CONTAINER_PGN_PATH = "/root/pgn_data"


# 4. Define the training function to run in the cloud
@stub.function(
    image=image,
    volumes={"/model_data": model_volume},
    gpu="any",
    timeout=86400,
)
def train_model():
    import sys
    sys.path.append("/root/src")
    
    from torch.utils.data import DataLoader, random_split

    from data_loader import load_positions_from_pgn_folder
    from move_vocab import MoveVocab
    from chess_dataset import ChessDataset
    from model import ChessNet

    # -----------------------------
    # Config
    # -----------------------------
    PGN_FOLDER = CONTAINER_PGN_PATH
    MODEL_PATH = CONTAINER_MODEL_PATH
    VOCAB_PATH = CONTAINER_VOCAB_PATH

    BATCH_SIZE = 128
    EPOCHS = 5
    VALID_SPLIT = 0.1
    LR = 1e-3

    # -----------------------------
    # Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Load positions
    # -----------------------------
    print("Loading PGN position data...")
    positions = load_positions_from_pgn_folder(PGN_FOLDER, limit_games=1000)
    if not positions:
        print(
            f"Error: No positions loaded. Did the PGN folder mount correctly at {PGN_FOLDER}?"
        )
        return

    print(f"Loaded {len(positions)} positions.")

    # -----------------------------
    # Build vocab and model
    # -----------------------------
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        print("Found existing model and vocab. Loading them...")

        vocab = torch.load(VOCAB_PATH, map_location=device, weights_only=False)
        output_size = len(vocab.idx_to_move)
        print(f"Vocab loaded with {output_size} moves.")

        model = ChessNet(output_size=output_size).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.")

    else:
        print("No existing model/vocab found. Building from scratch...")

        vocab = MoveVocab()
        vocab.build_from_positions(positions)
        output_size = len(vocab.idx_to_move)
        print(f"New vocab built with {output_size} moves.")

        model = ChessNet(output_size=output_size).to(device)
        print("Initialized new model.")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    # -----------------------------
    # Prepare dataset
    # -----------------------------
    dataset = ChessDataset(positions, vocab)
    val_size = int(len(dataset) * VALID_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Train positions: {len(train_dataset)}, Validation positions: {len(val_dataset)}"
    )

    # -----------------------------
    # Training loop
    # -----------------------------
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X_spatial, X_scalar, y in train_loader:
            X_spatial, X_scalar, y = (
                X_spatial.to(device),
                X_scalar.to(device),
                y.to(device),
            )

            optimizer.zero_grad()
            preds = model(X_spatial, X_scalar)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_spatial.size(0)

        avg_train_loss = train_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_spatial, X_scalar, y in val_loader:
                X_spatial, X_scalar, y = (
                    X_spatial.to(device),
                    X_scalar.to(device),
                    y.to(device),
                )

                preds = model(X_spatial, X_scalar)
                loss = loss_fn(preds, y)
                val_loss += loss.item() * X_spatial.size(0)
                predicted = preds.argmax(dim=1)
                correct += (predicted == y).sum().item()

        avg_val_loss = val_loss / len(val_dataset)
        val_accuracy = correct / len(val_dataset)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    # -----------------------------
    # Save final model
    # -----------------------------
    print("Saving model...")
    try:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"ERROR saving model: {e}")
        raise

    print("Saving vocab...")
    try:
        torch.save(vocab, VOCAB_PATH)
        print(f"Vocab saved to {VOCAB_PATH}")
    except Exception as e:
        print(f"ERROR saving vocab: {e}")
        raise

    print("Training complete!")


# 5. Define the local entrypoint to run the function
@stub.local_entrypoint() # <--- THIS IS THE FIX
def main():
    print("Starting remote training job on Modal...")
    train_model.remote()