import os
# Disable all GPU-related TensorFlow optimizations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Prevent TensorFlow from allocating GPU memory
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Prevents TensorFlow from using oneDNN optimizations (which may trigger GPU errors)
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="  # Prevents TensorFlow XLA from initializing CUDA
os.environ["TF_TRT_ENABLE"] = "0"
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from connect4 import display_board, check_for_win, update_board, find_legal

plt.style.use('dark_background')


###############################
# 1) Utility Functions
###############################

def board_to_tensor(board):
    """Convert board to 6x7x2 tensor for model input."""

    if board.shape == (6, 7, 2):  # Already in tensor format, return as is
        return board
    

    tensor = np.zeros((6, 7, 2))
    tensor[:, :, 0] = (board == 1).astype(int)  # Channel for 'plus' (X)
    tensor[:, :, 1] = (board == -1).astype(int) # Channel for 'minus' (O)
    return tensor


def check_gpu():
    """Check if TensorFlow is using the GPU."""
    print("GPUs Available:", tf.config.list_physical_devices('GPU'))
    print("TensorFlow version:", tf.__version__)
    print("GPU Device Name:", tf.test.gpu_device_name())


###############################
# 2) Model Functions
###############################

def residual_block(x, filters, kernel_size=(3,3)):
    """Residual block for ResNet model."""
    shortcut = tf.keras.layers.Conv2D(filters, (1,1), padding='same', 
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x


def build_resnet(input_shape=(6,7,2), num_classes=7):
    """Build a ResNet model for Connect4."""
    inputs = tf.keras.Input(shape=input_shape)
    
    # Initial convolutional layer
    x = tf.keras.layers.Conv2D(64, (3,3), padding='same', 
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4), use_bias=False)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(x)
    
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    # Fully connected layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # Optimizer with learning rate decay
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(X, Y, model_save_path):
    """Train ResNet model and save it."""
    model = build_resnet(input_shape=(6, 7, 2), num_classes=7)

    # Split data into train, validation, and test sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=42)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    ]

    # Train model
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=20, batch_size=128, callbacks=callbacks)

    # Save model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


def load_model_for_inference(model_path):
    """Load trained ResNet model for inference."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    return keras.models.load_model(model_path, compile=False)  # Prevent optimizer loading


def predict(model, board, current_player=None):
    """Predict the best move given a Connect4 board state using the ResNet model.
    
    Args:
        model (keras.Model): Trained ResNet model.
        board (np.ndarray): Connect4 board state (6x7).
        current_player (str, optional): If 'minus', flips board perspective.
    
    Returns:
        int: Best move (column index 0-6).
    """
    if model is None:
        raise ValueError("Model is not loaded. Call load_model_for_inference first.")

    # Flip board perspective for 'minus' player (if applicable)
    if current_player == 'minus':
        board = -board  # Flip board

    # Prepare input tensor
    tensor = board_to_tensor(board)[np.newaxis, ...]

    # Get model predictions
    predictions = model.predict(tensor, verbose=0)[0]

    # Select best legal move
    legal_moves = find_legal(board)
    legal_probs = [(predictions[col], col) for col in legal_moves]

    return max(legal_probs)[1]  # Return column with highest probability


###############################
# 3) Game Functions
###############################


def play_vs_ai(model):
    """Main game loop for human vs AI."""
    board = np.zeros((6, 7))
    winner = 'nobody'

    # Let player choose side
    human_side = input("Choose your side (X/O): ").upper()
    while human_side not in ['X', 'O']:
        human_side = input("Invalid choice. Choose X or O: ").upper()

    human_player = 'plus' if human_side == 'X' else 'minus'
    ai_player = 'minus' if human_side == 'X' else 'plus'

    current_player = 'plus'  # X always starts

    while winner == 'nobody':
        display_board(board)

        if current_player == human_player:
            legal_moves = find_legal(board)
            while True:
                try:
                    move = int(input(f"Your move (0-6) [{', '.join(map(str, legal_moves))}]: "))
                    if move in legal_moves:
                        break
                    print(f"Invalid move! Choose from {legal_moves}.")
                except ValueError:
                    print("Invalid input! Please enter a number between 0-6.")
        else:
            print("AI is thinking...")
            move = predict(board, current_player, model)
            print(f"AI plays column {move}\n")

        board = update_board(board, current_player, move)
        winner = check_for_win(board, move)

        current_player = 'minus' if current_player == 'plus' else 'plus'

        if not find_legal(board) and winner == 'nobody':
            winner = 'tie'

    display_board(board)
    print(f"Winner: {'Tie' if winner == 'tie' else ('Human' if winner[2] == human_side[0] else 'AI')}!")


###############################
# 4) Main Execution
###############################


def main():
    boards_path = "data/connect4_boards.npy"  # Path to boards dataset
    moves_path = "data/connect4_moves.npy"  # Path to moves dataset
    model_save_path = "cnn_model/best_resnet_model_76.h5"  # Path to save the trained model

    # Ensure dataset files exist
    if not os.path.exists(boards_path) or not os.path.exists(moves_path):
        print(f"Error: Dataset files not found in 'data/' folder.")
        return

    # Load pre-saved NPY data
    print("Loading dataset from NPY files...")
    X = np.load(boards_path, allow_pickle=True)
    Y = np.load(moves_path, allow_pickle=True)
    X = X[:300000]
    Y = Y[:300000]
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

    # play_vs_ai(model)

    # Example inference with an actual board state from dataset
    example_board = X[0]  # Take the first board from dataset

    print("Example board state for inference:")
    print(example_board[:, :, 0] - example_board[:, :, 1])  # Convert 6x7x2 into 6x7 view

    predicted_move = predict(model, example_board)
    print(f"Predicted move: {predicted_move}")

if __name__ == "__main__":
    main()
