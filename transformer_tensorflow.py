import os
# Disable all GPU-related TensorFlow optimizations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Prevent TensorFlow from allocating GPU memory
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Prevents TensorFlow from using oneDNN optimizations (which may trigger GPU errors)
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="  # Prevents TensorFlow XLA from initializing CUDA
os.environ["TF_TRT_ENABLE"] = "0"

import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Embedding, LayerNormalization, MultiHeadAttention, Reshape
)
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import custom_object_scope
from sklearn.model_selection import train_test_split

def board_to_tensor(board):
    """
    Converts a Connect4 board into a Tensor for model inference.

    Parameters:
    board (numpy.ndarray): The board to convert. Can be of shape (6, 7) or (6, 7, 2).

    Returns:
    tf.Tensor: A Tensor with shape (1, 6, 7, 2), ready for model inference.
    """
    if board.shape == (6, 7):
        # Convert (6, 7) to (6, 7, 2) format
        encoded_board = np.zeros((6, 7, 2), dtype=np.float32)
        encoded_board[:, :, 0] = (board == 1).astype(np.float32)  # 'plus' player
        encoded_board[:, :, 1] = (board == -1).astype(np.float32)  # 'minus' player
    elif board.shape == (6, 7, 2):
        # Use the existing (6, 7, 2) board
        encoded_board = board.astype(np.float32)
    else:
        raise ValueError("Input board must have shape (6, 7) or (6, 7, 2).")

    # Add batch dimension to make it (1, 6, 7, 2)
    tensor = tf.convert_to_tensor(np.expand_dims(encoded_board, axis=0))
    # tensor = tf.squeeze(tensor, axis=0)
    return tensor

# **1. Positional Encoding**
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, height, width):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.position_embeddings = Embedding(input_dim=height * width, output_dim=embed_dim)

    def call(self, inputs):
        # Create a sequence of position indices from 0 to (height*width - 1)
        position_indices = tf.range(start=0, limit=self.height * self.width, delta=1)
        position_embeddings = self.position_embeddings(position_indices)
        # Reshape to (1, height*width, embed_dim) so it can be added to inputs
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        return inputs + position_embeddings

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='gelu'),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


class CSVLoggerCallback(Callback):
    def __init__(self, file_path):
        super(CSVLoggerCallback, self).__init__()
        self.file_path = file_path

        # Write the header row if the file does not exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Epoch', 'Train Accuracy', 'Validation Accuracy'])

    def on_epoch_end(self, epoch, logs=None):
        # Append the metrics for the current epoch
        with open(self.file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                logs.get('accuracy', None),  # Training accuracy
                logs.get('val_accuracy', None)  # Validation accuracy
            ])



# **3. Transformer Model Definition**
# --- Adjusted Transformer Model Definition ---
def create_transformer_model(
    input_shape, embed_dim=128, num_heads=8, ff_dim=256,
    num_transformer_blocks=3, dropout_rate=0.2
):
    inputs = Input(shape=input_shape)  # Expected shape: (6, 7, 2)
    x = Dense(embed_dim)(inputs)  # Linear projection -> shape (batch, 6, 7, embed_dim)

    # Flatten the 6x7 board to form a sequence of 42 tokens of dimension embed_dim
    # x = tf.reshape(x, (-1, input_shape[0] * input_shape[1], embed_dim))
    x = Reshape((input_shape[0] * input_shape[1], embed_dim))(x) 

    x = PositionalEncoding(embed_dim, input_shape[0], input_shape[1])(x)
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(7, activation='softmax')(x)  # 7 possible moves

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# **4. Data Loading**
def load_npy_data(boards_path, moves_path):
    X = np.load(boards_path, allow_pickle=True)
    Y = np.load(moves_path, allow_pickle=True)
    return X, Y


# **5. Model Training**
def train_transformer_model(X, Y, model_save_path, epochs=100, batch_size=256):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=12)
    
    # Create transformer model
    transformer_model = create_transformer_model(input_shape=(6, 7, 2))

    # Initialize the custom CSV logging callback
    csv_logger_callback = CSVLoggerCallback(file_path='transformer_loss_tracker.csv')

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True),
        csv_logger_callback  # Add the custom callback here
    ]

    # Train the model and save history
    history = transformer_model.fit(
        x_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )

    # Evaluate Model
    results = transformer_model.evaluate(x_test, y_test, batch_size=batch_size)
    print(f"Test Accuracy: {results[1]:.2%}")

    transformer_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


# **6. Load Model for Inference**
# **6. Load Model for Inference**
def load_model_for_inference(model_path):
    """Loads a trained Transformer model for inference."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")
    try:
        # Load with custom objects
        custom_objects = {
            'PositionalEncoding': PositionalEncoding,
            'TransformerBlock': TransformerBlock
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")


# **7. Prediction Function**
def predict(model, board):
    # print(board.shape)
    # print(board)
    board_tensor = board_to_tensor(board)
    # board_tensor = np.expand_dims(board, axis=0)  # Add batch dimension
    print(f"Shape before prediction: {board_tensor.shape}")
    predictions = model.predict(board_tensor, verbose=0)
    x =np.argmax(predictions)
    return x


# **8. Main Execution**
def main():
    boards_path = "data/connect4_boards.npy"  
    moves_path = "data/connect4_moves.npy"  
    model_save_path = "trans_model/transformer_best_67.h5"

    # Load dataset
    if not os.path.exists(boards_path) or not os.path.exists(moves_path):
        print("Error: Dataset files not found.")
        return

    print("Loading dataset from NPY files...")
    X, Y = load_npy_data(boards_path, moves_path)
    # X, Y = X[:8000], Y[:8000]
    print(f"Loaded {len(X)} samples.")

    if not os.path.exists(model_save_path):
        print("Training the Transformer model...")
        train_transformer_model(X, Y, model_save_path)
    else:
        print(f"Model already exists at {model_save_path}. Skipping training.")

    print("Loading the trained model for inference...")
    model = load_model_for_inference(model_save_path)

    example_board = X[0]
    print("Example board state for inference:")
    print(example_board[:, :, 0] - example_board[:, :, 1])  # Show board difference for clarity
    predicted_move = predict(model, example_board)
    print(f"Predicted move: {predicted_move}")


if __name__ == "__main__":
    main()