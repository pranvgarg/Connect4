# Playing Connect 4 with AI: A Deep Dive into Developing the Bot

## Welcome to Our Connect 4 AI Game!

Connect 4 AI is a computer-powered opponent that leverages advanced algorithms to challenge human players. Far more than a simple game of dropping pieces into a grid, it is a battle of tactics, foresight, and strategy. Every move influences your opponent’s subsequent choices, requiring not only immediate tactical responses but also long-term planning.

Our system systematically analyzes board states to identify winning patterns, block threats, and plan multi-turn strategies. By combining simulation techniques with deep learning models, we have built an AI that not only plays Connect 4 at a high level but also provides a practical example of applying advanced machine learning to strategic decision-making.

This project details our approach—from generating training data using Monte Carlo Tree Search (MCTS) to training both Convolutional Neural Network (CNN) and Transformer models, and finally deploying the solution using Docker on AWS Lightsail.


## Board Update & Move Functions

### `update_board(board_temp, color, column)`
- **Purpose:**  
  Creates a new board state by placing a piece of the given color in the specified column.
- **Inputs:**  
  - `board_temp`: Current 6×7 board state (NumPy array).
  - `color`: Either `'plus'` (represents 1) or `'minus'` (represents -1).
  - `column`: Column index (0–6) where the piece is dropped.
- **Output:**  
  Returns the updated board state.

---

### `make_move(board, color, column)`
- **Purpose:**  
  Places a piece of the given color in the specified column **in-place** and returns the row where it was placed.
- **Inputs:**  
  - `board`: The current board state.
  - `color`: `'plus'` or `'minus'`.
  - `column`: Column index for the move.
- **Output:**  
  The row index where the piece lands, or -1 if the column is full.

---

### `undo_move(board, row, column)`
- **Purpose:**  
  Resets the board cell at the given row and column back to 0 (empty).
- **Inputs:**  
  - `board`: The current board state.
  - `row`: The row index of the move.
  - `column`: The column index of the move.

---

## Win Checking Functions

### `check_for_win_slow(board)`
- **Purpose:**  
  Checks the entire board for a winning sequence (vertical, horizontal, or diagonal) using a brute-force approach.
- **Inputs:**  
  - `board`: The current 6×7 board state.
- **Output:**  
  Returns a string indicating the winning pattern (e.g., `'v-plus'`, `'h-minus'`, etc.) or `'nobody'` if no win is detected.

---

### `check_for_win(board, col)`
- **Purpose:**  
  Checks if the last move (placed in column `col`) resulted in a win.
- **Inputs:**  
  - `board`: The current board state.
  - `col`: Column index of the last move.
- **Output:**  
  Returns a string representing the winning pattern (e.g., `'v-plus'`) or `'nobody'` if there is no winner.

---

### `find_legal(board)`
- **Purpose:**  
  Determines which columns still have available space (i.e., legal moves).
- **Inputs:**  
  - `board`: Current board state (either 6×7 or 6×7×2; if 6×7×2, it is converted).
- **Output:**  
  Returns a list of column indices (0–6) where a move can be made.

---

## AI Helper Functions for MCTS

### `look_for_win(board, color)`
- **Purpose:**  
  Checks all legal moves to see if any move results in an immediate win for the given color.
- **Inputs:**  
  - `board`: Current board state.
  - `color`: `'plus'` or `'minus'`.
- **Output:**  
  Returns the column index of the winning move, or -1 if none is found.

---

### `find_all_nonlosers(board, color)`
- **Purpose:**  
  Identifies all legal moves that do not immediately allow the opponent to win.
- **Inputs:**  
  - `board`: Current board state.
  - `color`: The player’s color.
- **Output:**  
  Returns a list of column indices that are considered safe moves.

---

### `back_prop(winner, path, color0, md)`
- **Purpose:**  
  Updates the MCTS dictionary (`md`) for each board state in the given path based on the simulation outcome.
- **Inputs:**  
  - `winner`: The winning pattern from the simulation.
  - `path`: List of board states (as tuples) traversed during simulation.
  - `color0`: The player for whom MCTS is running.
  - `md`: Dictionary mapping board states to statistics `[visits, score]`.

---

### `rollout(board, next_player, debug=False)`
- **Purpose:**  
  Performs a randomized simulation (rollout) from the current board until a terminal state (win/tie) is reached.
- **Inputs:**  
  - `board`: Current board state.
  - `next_player`: The color to move next.
  - `debug`: Optional flag to print debugging output.
- **Output:**  
  Returns the outcome of the simulation (`'nobody'`, `'tie'`, or a win code).

---

### `mcts(board_temp, color0, nsteps)`
- **Purpose:**  
  Runs Monte Carlo Tree Search for a specified number of iterations (`nsteps`) to determine the best move.
- **Inputs:**  
  - `board_temp`: Current board state.
  - `color0`: The AI’s color.
  - `nsteps`: Number of iterations for the search (higher value improves quality but increases computation).
- **Output:**  
  Returns the column index of the best move found by MCTS.

---

## Display Function

### `display_board(board)`
- **Purpose:**  
  Prints an ASCII representation of the board to the console using:
  - `'X'` for player `+1`
  - `'O'` for player `-1`
  - Blanks for empty cells.
- **Inputs:**  
  - `board`: Current board state.
- **Output:**  
  Displays the board along with column indices.

---

## Main Game Loop

In the main block (when running `python connect4.py`), the following steps occur:
1. **Initialization:**  
   - An empty board is created.
   - The player chooses whether to go first.
2. **Game Loop:**  
   - The board is displayed.
   - If it's the human's turn, input is requested until a legal move is made.
   - If it's the AI's turn, MCTS is used to determine the move.
   - The move is applied using `update_board()`, and win status is checked.
   - Turns alternate until a winner or tie is declared.
3. **Final Display:**  
   - The final board state is printed and the result (win/tie) is announced.

---

## 1. Generating Training Data with MCTS

To train our Connect 4 AI effectively, we needed a vast dataset of board states and optimal moves. Manually labeling millions of positions was impractical, so we leveraged **Monte Carlo Tree Search (MCTS)** to generate high-quality training data through self-play.

### How MCTS Works:
- **Simulating Possible Games:**  
  MCTS simulates numerous games from various board positions. In each simulation (or rollout), the algorithm randomly plays out moves to explore potential outcomes.
  
- **Tracking Move Effectiveness:**  
  Every move is recorded along with the eventual win/loss result from the simulation. This statistical evaluation helps the AI determine which moves lead to better outcomes.
  
- **Balancing Exploration and Exploitation:**  
  The algorithm employs the Upper Confidence Bound (UCB1) formula to strike a balance between:
  - **Exploration:** Trying new moves to discover their potential.
  - **Exploitation:** Refining moves that are already known to be effective.

- **Refining Move Selection:**  
  As more simulations are run, MCTS learns to select statistically better moves. It also includes pre-checks for immediate wins or blocks, ensuring that obvious tactical moves are prioritized.

### Data Generation Process:
- **Self-Play:**  
  MCTS is used for self-play, where the AI plays roughly 40,000 games against itself, each time recording the best move for every board state.
  
- **Recording Moves:**  
  Each board state is captured as a 6×7 tensor (or optionally as a 6×7×2 tensor with separate channels for each player), and the optimal move is stored.
  
- **Data Augmentation:**  
  The board states are mirrored horizontally to double the dataset size, preserving the strategic context while enhancing data diversity.
  
- **Parallel Processing:**  
  Multiple simulations run concurrently via multiprocessing, drastically reducing the time needed to generate the large dataset.
  
- **Efficient Storage:**  
  Given the large volume (over 1.8 million snapshots), we use NumPy memmaps to manage and update the dataset without exceeding memory limits.

This extensive MCTS-driven self-play process provided the robust training data necessary to develop our AI models.

---

## 2. Bot 1: CNN-based Model

Our first AI agent is built using Convolutional Neural Networks (CNNs), which are particularly well-suited for grid-based games like Connect 4. CNNs excel at capturing local spatial patterns—critical for detecting winning formations, blocking moves, and setting up tactical traps.

### Model Architecture

- **Input Representation:**  
  The game board is represented as a 6×7×2 tensor. Each channel corresponds to one player's pieces:
  - **Channel 0:** Indicates a 'plus' (represented by +1)
  - **Channel 1:** Indicates a 'minus' (represented by -1)

- **Initial Convolutional Layer:**  
  - Uses 64 filters of size 3×3 with "same" padding to preserve the board dimensions.
  - Applies L2 regularization (1e-4), followed by Batch Normalization and ReLU activation to extract low-level spatial features.

- **Residual Blocks:**  
  - Inspired by ResNet, multiple residual blocks are used to allow the network to learn deeper features without the vanishing gradient problem.
  - Each block consists of two convolutional layers (with batch normalization and ReLU) and includes a skip connection that adds the block’s input to its output.

- **Max Pooling Layers:**  
  - Applied after selected residual blocks to downsample the spatial dimensions.
  - This reduction in size helps the model focus on larger, more abstract patterns and reduces computational complexity.

- **Fully Connected Layers:**  
  - The output from the convolutional and pooling layers is flattened into a one-dimensional vector.
  - A dense layer with 512 units (with ReLU activation) processes these features, followed by Batch Normalization and Dropout (50%) for regularization.

- **Output Layer:**  
  - A final dense layer with 7 units and Softmax activation produces a probability distribution over the 7 possible moves (columns), indicating the likelihood of each move leading to a win.

### Training and Performance

- **Data:**  
  The CNN is trained on a vast dataset of board states generated via MCTS self-play. Each board is converted into a 6×7×2 tensor.
  
- **Optimization:**  
  - The model is optimized using Adam (or another suitable optimizer) with a scheduled learning rate.
  - Regularization through dropout and batch normalization ensures the network generalizes well.

- **Results:**  
  - The CNN-based model achieves a validation accuracy of approximately **76%**.
  - It is particularly effective at recognizing localized spatial patterns—vital for blocking opponent moves and executing immediate winning strategies.

### Advantages

- **Spatial Pattern Recognition:**  
  CNNs naturally excel at detecting local patterns, which is crucial for recognizing winning configurations in Connect 4.
  
- **Efficiency and Robustness:**  
  The architecture is computationally efficient and benefits from residual connections and regularization, resulting in a model that trains quickly and performs reliably across various board states.

### Example Architecture Code Snippet

```python
# Input shape: 6x7x2
inputs = tf.keras.Input(shape=input_shape)

# Initial convolution with 3x3 filters
x = tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                           use_bias=False)(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.ReLU()(x)

# Residual Blocks
x = residual_block(x, 64)
x = residual_block(x, 64)

# Max Pooling to downsample spatial dimensions
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = residual_block(x, 128)
x = residual_block(x, 128)

x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = residual_block(x, 256)
x = residual_block(x, 256)

# Fully connected layers
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.5)(x)

# Output layer with softmax for 7 possible moves
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

```
<img width="693" alt="image" src="https://github.com/user-attachments/assets/8ce3cd00-aaad-4c42-b8b1-12fd1847c5b7" />
![resnet_model h5](https://github.com/user-attachments/assets/8999785a-a880-4d7e-8262-2a4a39e6448d)

---

## 3. Bot 2: Transformer-based Model

While Convolutional Neural Networks (CNNs) excel at spatial pattern recognition, Transformers offer a unique approach by focusing on sequence-based decision-making. In Connect 4, understanding board states holistically—beyond just local patterns—is crucial for strategic play. Our Transformer-based model leverages self-attention mechanisms to analyze the game board and predict optimal moves.

---

### **Model Architecture**

Unlike CNNs, which focus on nearby spatial dependencies, the Transformer model considers **global relationships** across the board. The architecture consists of several key components:

#### **1. Input Representation**
- The board is represented as a **6×7×2 tensor**, where:
  - **Channel 0** represents the `plus` player's pieces.
  - **Channel 1** represents the `minus` player's pieces.
- This format ensures that the model understands both player perspectives.

#### **2. Linear Projection**
- The board tensor is passed through a **dense (fully connected) layer** to **map board values into a high-dimensional feature space**.
- This step helps the model extract meaningful numerical representations of board states.

#### **3. Positional Encoding**
- Since Transformers were originally designed for sequences (e.g., text data), they lack an inherent sense of position.
- A **Positional Encoding Layer** is added to embed spatial information into the board representation.
- This enables the Transformer to recognize board locations and their significance.

#### **4. Transformer Blocks**
The core of our model is built using **multiple Transformer blocks**, each consisting of:
- **Multi-Head Self-Attention:**  
  - This mechanism enables the model to focus on different board areas simultaneously.
  - It helps detect relationships between pieces that may be far apart on the board.
  
- **Feedforward Network (FFN):**  
  - A two-layer dense network that applies **non-linear transformations** to refine feature representations.
  - The first layer uses **GELU activation**, known for smoother gradient updates.
  
- **Layer Normalization & Dropout:**  
  - Used for stabilizing training and reducing overfitting.

#### **5. Output Layer**
- The final board representation is **flattened** and passed through dense layers:
  - **256 dense units → ReLU activation**
  - **128 dense units → ReLU activation**
  - **Dropout layers for regularization**
- The output layer is a **softmax layer with 7 units**, corresponding to **the probability distribution over valid columns** (0–6).

---

### **Transformer Model Code Snippet**
```python
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, embed_dim, height, width):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.height = height
        self.width = width
        self.position_embeddings = Embedding(input_dim=height * width, output_dim=embed_dim)

    def call(self, inputs):
        position_indices = tf.range(start=0, limit=self.height * self.width, delta=1)
        position_embeddings = self.position_embeddings(position_indices)
        position_embeddings = tf.expand_dims(position_embeddings, axis=0)
        return inputs + position_embeddings


def create_transformer_model(input_shape, embed_dim=128, num_heads=8, ff_dim=256, num_transformer_blocks=3, dropout_rate=0.2):
    inputs = Input(shape=input_shape)  # Expected shape: (6, 7, 2)
    x = Dense(embed_dim)(inputs)  # Linear projection

    # Flatten the 6x7 board into a sequence of 42 tokens
    x = Reshape((input_shape[0] * input_shape[1], embed_dim))(x)

    # Add positional encoding
    x = PositionalEncoding(embed_dim, input_shape[0], input_shape[1])(x)

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    # Output layer for move prediction
    outputs = Dense(7, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
```
<img width="678" alt="image" src="https://github.com/user-attachments/assets/85af50c6-9b38-4fdd-9180-c703ff3b4fc6" />
![transformer_model h5](https://github.com/user-attachments/assets/079b268a-b637-4062-874d-a0a54ce21cc9)

---

## 4. Model Performance Comparison

In our exploration of deep learning approaches for Connect 4, we implemented two distinct models:  
- **CNN-based Model (ResNet-inspired)**
- **Transformer-based Model (Self-Attention Mechanism)**  

Each model was trained on the **same dataset generated by Monte Carlo Tree Search (MCTS)**, but their performances varied significantly due to differences in architectural strengths and limitations.

---

### **📊 Performance Metrics**

| Model                  | Accuracy | Training Speed | Computational Cost | Strengths | Weaknesses |
|------------------------|----------|---------------|----------------------|------------|------------|
| **CNN-based Model**    | **76%**  | **Fast**      | **Lower**            | Excellent pattern recognition, effective at tactical moves | Struggles with long-term planning |
| **Transformer Model**  | **67%**  | **Slow**      | **Higher**            | Holistic board awareness, flexible for long-term strategies | Requires more data, struggles with short-term tactics |

🔹 **CNNs** excelled at recognizing localized spatial patterns essential for **short-term tactical play**.  
🔹 **Transformers**, while better at **long-term strategic planning**, struggled with Connect 4’s **spatial dependencies**.

---

# **5. Docker Implementation of the Connect 4 Game**
---

## **Overview**
This section details the **Docker-based deployment** of the Connect 4 AI game on **AWS Lightsail**. The project integrates **Django, TensorFlow, and Anvil Uplink** within a **containerized environment** for efficient deployment and management.  

By using **Docker and Docker Compose**, we ensure:
- **Portability**: The application runs in an isolated container across different environments.
- **Scalability**: Future improvements can leverage **Kubernetes** for container orchestration.
- **Automation**: CI/CD pipelines can be incorporated for continuous deployment.

---

## **Anvil Uplink - Docker Process**
The deployment process includes:
1. **Setting up Docker** on the AWS Lightsail instance.
2. **Transferring project files** (codebase, models, and dependencies).
3. **Building & launching the Docker container** to host the backend AI service.

The **Dockerfile** defines the containerized environment, ensuring **all dependencies** (Python, TensorFlow, Django, Anvil Uplink) are installed and managed.

The **Docker Compose** file simplifies the process by defining multiple services, including automatic restarts.
<img width="707" alt="image" src="https://github.com/user-attachments/assets/1f04917b-7487-4975-8b60-f771deb401b8" />

---

## **Key Steps for Running the Game in Docker**
Below are the essential commands to build, launch, and debug the container:

```sh
# Build the Docker image
sudo docker compose build

# Run the container in detached mode
sudo docker compose up -d

# View logs for debugging
sudo docker compose logs

# Verify running containers
sudo docker compose ps

# Stop and remove containers
sudo docker compose down
```

Debugging Best Practices
	•	Ensure correct file permissions for project files before copying them into the container.
	•	Adjust TensorFlow GPU settings in docker-compose.yml for optimized performance.
	•	Regularly clean up unused images and containers:
```sh
docker system prune -a
```


Challenges Encountered

We encountered multiple challenges during deployment:
	•	Understanding Image vs. Container Workflows: Differentiating between image creation and container execution required fine-tuning.
	•	Handling Large TensorFlow Dependencies: The first deployment failed due to excessive memory usage when loading deep learning models.
	•	File Transfer Issues: Using FileZilla to transfer project files to AWS ensured complete dependency management.

Future Improvements

To enhance performance, we propose:
	1.	CI/CD Pipeline Automation: Automating deployments via GitHub Actions or Jenkins.
	2.	Using Kubernetes: For scalable container orchestration and efficient GPU allocation.
	3.	Optimizing AI Response Time: Currently, the backend processes the board state before returning the next move. Future improvements will focus on reducing response delay.


