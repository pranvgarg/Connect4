import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from connect4 import display_board

# Dataset Class
class Connect4Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, d_model=256):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)  # Add dropout initialization

    def forward(self, x):
        x = x.flatten(start_dim=1)  # Flatten 6x7x2 into 84
        x = self.input_projection(x)  # Shape: (batch_size, d_model)
        x = x.unsqueeze(1)  # Add sequence dimension: (batch_size, seq_len=1, d_model)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)  # Remove sequence dimension: (batch_size, d_model)
        x = self.fc(self.dropout(x))  # Apply dropout before the final layer
        return x


# Load Data from CSV
def load_csv_data(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    X = np.array([np.array(eval(board)).reshape(6, 7, 2) for board in tqdm(df['board'], desc="Loading Boards")])
    Y = df['move'].values
    return X, Y

# Train Model
def train_model(X, Y, model_save_path, epochs=50, batch_size=64, learning_rate=1e-4):
    dataset = Connect4Dataset(X, Y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = TransformerModel(input_dim=84, num_heads=4, num_layers=4, num_classes=7, d_model=256)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X_batch, Y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == Y_batch).sum().item()
            total += Y_batch.size(0)
        train_acc = correct / total * 100
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X_batch, Y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == Y_batch).sum().item()
                total += Y_batch.size(0)
        val_acc = correct / total * 100

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Validation Loss: {val_loss/len(val_loader):.4f}, Train Accuracy: {train_acc:.2f}%, "
              f"Validation Accuracy: {val_acc:.2f}%")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

# Load Model for Inference
def load_model_for_inference(model_path):
    model = TransformerModel(input_dim=84, num_heads=4, num_layers=4, num_classes=7, d_model=256)
    model.load_state_dict(torch.load(model_path))  # Load the trained weights
    model.eval()  # Set to evaluation mode
    return model

# # Predict Function
def predict(model, board):
    # print('hello hello')
    model.eval()
    with torch.no_grad():
        # Convert the board to a tensor
        board_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 6, 7, 2)
        # board_tensor = board_tensor.view(1, -1)  # Flatten to shape (1, 84)
        outputs = model(board_tensor)  # Pass through the model
        _, predicted_move = torch.max(outputs, 1)  # Get the index of the max log-probability
        return predicted_move.squeeze().item()  # Return as scalar

def main():
    boards_path = "data/connect4_boards.npy"  # Path to boards dataset
    moves_path = "data/connect4_moves.npy"  # Path to moves dataset
    model_save_path = "trans_model/transformer_model_pytorch.pth"  # Path to save the trained model

    # Ensure dataset files exist
    if not os.path.exists(boards_path) or not os.path.exists(moves_path):
        print(f"Error: Dataset files not found in 'data/' folder.")
        return

    # Load pre-saved NPY data
    print("Loading dataset from NPY files...")
    X = np.load(boards_path, allow_pickle=True)
    Y = np.load(moves_path, allow_pickle=True)
    # X, Y = X[:3000], Y[:3000]
    print(f"Data Boards loaded: {len(X)} samples.")
    print(f"Data Moves loaded: {len(Y)} samples.")

    # Train the model (only if it doesn't exist)
    if not os.path.exists(model_save_path):
        print("Training the model...")
        train_model(X, Y, model_save_path)
    else:
        print(f"Model already exists at {model_save_path}. Skipping training.")

    # Load the trained model
    print("Loading the trained model for inference...")
    model = load_model_for_inference(model_save_path)

    # Example inference with an actual board state from dataset
    example_board = X[0]  # Take the first board from dataset

    print("Example board state for inference:")
    print(example_board[:, :, 0] - example_board[:, :, 1])  # Convert 6x7x2 into 6x7 view

    predicted_move = predict(model, example_board)
    print(f"Predicted move: {predicted_move}")

if __name__ == "__main__":
    main()
