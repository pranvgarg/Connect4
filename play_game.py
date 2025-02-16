import os

# Disable all GPU-related TensorFlow optimizations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress warnings
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"  # Prevent TensorFlow from allocating GPU memory
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Prevents TensorFlow from using oneDNN optimizations (which may trigger GPU errors)
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir="  # Prevents TensorFlow XLA from initializing CUDA
os.environ["TF_TRT_ENABLE"] = "0"

import numpy as np  # type: ignore
import logging
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import anvil.server
import anvil.mpl_util
from transformer_tensorflow import load_model_for_inference as load_transformer, predict as predict_transformer
from resnet_cnn import load_model_for_inference as load_cnn_model, predict as predict_cnn
from connect4 import update_board, check_for_win, find_legal 

##########################
# Connect to Anvil Uplink
##########################
# Replace with your actual Uplink key
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

UPLINK_KEY = "server_XPYO4PRWZOOAGHZMIU4E6MX4-43KJBQX6ZG4GCSHM"

def connect_to_anvil():
    retry_delay = 5  # Start with 5 seconds

    while True:
        try:
            logging.info("Attempting to connect to Anvil...")
            anvil.server.connect(UPLINK_KEY)
            logging.info("Connected to Anvil server.")
            anvil.server.wait_forever()  # Keep server running

        except anvil.server.UplinkDisconnectedError:
            logging.warning("Uplink disconnected. Reconnecting in %d seconds...", retry_delay)
        
        except Exception as e:
            logging.error("Unexpected error: %s", e)

        # Exponential backoff (up to 60 seconds)
        time.sleep(retry_delay)
        retry_delay = min(retry_delay * 2, 60)

def find_winning_move(board, player):
    """
    Finds an immediate winning move for the given player.
    If a winning move exists, returns the column number.
    Otherwise, returns None.
    """
    legal_moves = find_legal(board)  # Get all legal columns
    player_value = 1 if player == 'plus' else -1  # Convert player to board values

    print(f"Checking for winning move for {player}. Legal moves: {legal_moves}")

    for col in legal_moves:
        temp_board = np.copy(board)  # Copy board to avoid modifying the original
        for row in range(5, -1, -1):  # Find the first empty row in this column
            if temp_board[row, col] == 0:
                temp_board[row, col] = player_value  # Simulate placing a piece
                break  # Stop once a move is made

        if check_for_win(temp_board, col) != 'nobody':  # If this move wins, return it
            print(f"Winning move found at column {col}")
            return col

    print("No winning move found.")
    return None  # No winning move found

def find_blocking_move(board, opponent):
    """
    Checks if the opponent has a winning move and returns the column to block them.
    If there's no threat, returns None.
    """
    return find_winning_move(board, opponent)  # Just use the same logic as winning move

# Store pending human move
pending_human_move = None

##########################
# Display Board Function
##########################
@anvil.server.callable
def display_board(board):
    """Generates a Matplotlib figure representing the Connect Four board with rounded edges."""
    
    # Create figure (keeping the same size)
    fig, ax = plt.subplots(figsize=(8, 8))  # Keeps the board dimensions unchanged
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')  # White background for rounded effect

    # Draw rounded rectangle as board background
    board_rect = patches.FancyBboxPatch(
        (-0.5, -0.5), 7, 6,  # Board dimensions (width=7, height=6)
        boxstyle="round,pad=0.4",  # Rounded corners
        facecolor='#2753C4',  # Dark blue board background
        edgecolor='black',  # Black outline
        linewidth=3
    )
    ax.add_patch(board_rect)

    # Draw the board grid with styled pieces
    for row in range(6):
        for col in range(7):
            piece_color = (
                'white' if board[row, col] == 0 
                else 'red' if board[row, col] == 1  
                else 'yellow'
            )

            # Add a black outline to pieces for depth
            circle = plt.Circle(
                (col, 5 - row), 0.4,  # Keeping same size as before
                fill=True, 
                facecolor=piece_color, 
                edgecolor='black',  # Black outline for contrast
                linewidth=2
            )
            ax.add_patch(circle)

    # Set plot limits and aspect ratio
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')

    # Customize x-axis (1 to 7)
    ax.set_xticks(range(7))
    ax.set_xticklabels([str(i+1) for i in range(7)], fontsize=14, fontweight='bold')

    # Customize y-axis (A to F) from top to bottom
    ax.set_yticks(range(6))
    ax.set_yticklabels(["A", "B", "C", "D", "E", "F"][::-1], fontsize=14, fontweight='bold')

    # Hide ticks
    ax.tick_params(axis='both', length=0)  # Removes tick marks

    # Set the title with spacing
    plt.title("Connect Four", fontsize=18, fontweight='bold', pad=20)

    # Convert Matplotlib plot to Anvil image
    boardimg = anvil.mpl_util.plot_image()

    # Close the figure to prevent memory leaks
    plt.close(fig)

    return boardimg  # Return the figure image for Anvil UI rendering


##########################
# Human Move Handler
##########################
@anvil.server.callable
def get_human_move(legal_moves):
    """
    Requests a move from the human player via Anvil UI.
    Returns nothing but waits for the frontend to call `submit_human_move`.
    """
    global pending_human_move
    pending_human_move = None  # Reset before waiting
    print(f"Waiting for human move. Legal moves: {legal_moves}")

    while pending_human_move is None:  # Keep checking for input
        anvil.server.wait_forever()  # Block until frontend provides input

    return pending_human_move  # Return the selected move


@anvil.server.callable
def submit_human_move(move):
    """
    Called by the frontend to send the human move to the backend.
    """
    global pending_human_move
    pending_human_move = move
    print(f"Received human move: {move}")



##########################
# Transformer AI Game Logic
##########################
@anvil.server.callable
def play_game_cnn(cnn_model_path):
    """Initialize game and return initial board state and image"""
    global cnn_model, current_board
    # Load CNN model
    cnn_model = load_cnn_model(cnn_model_path)
    # Initialize empty board
    current_board = np.zeros((6, 7))
    # Create board image
    board_image = display_board(current_board)
    return {
        'board': current_board,
        'image': board_image
    }

@anvil.server.callable
def play_game_transformer(trans_model_path):
    """Initialize game and return initial board state and image"""
    global trans_model, current_board
    # Load Transformer model
    trans_model = load_transformer(trans_model_path)
    print("TESTING")
    # Initialize empty board
    current_board = np.zeros((6, 7))
    # Create board image
    board_image = display_board(current_board)
    return {
        'board': current_board,
        'image': board_image
    }



# @anvil.server.callable
# def get_cnn_response(board):
#     board = np.array(board)
#     cnn_move = int(predict_cnn(cnn_model, board))  # Ensure Python int
#     new_board = update_board(board, 'plus', cnn_move)
#     board_image = display_board(new_board)
#     winner = check_for_win(new_board, cnn_move)

#     return {
#         'board': new_board.tolist(),  # Convert NumPy array to list
#         'image': board_image,
#         'winner': winner if isinstance(winner, str) else str(winner),
#         'move': cnn_move
#     }

# @anvil.server.callable
# def get_trans_response(board):
#     board = np.array(board)
#     trans_move = int(predict_transformer(trans_model, board))  # Ensure Python int
#     new_board = update_board(board, 'plus', trans_move)
#     board_image = display_board(new_board)
#     winner = check_for_win(new_board, trans_move)

#     return {
#         'board': new_board.tolist(),
#         'image': board_image,
#         'winner': winner if isinstance(winner, str) else str(winner),
#         'move': trans_move
#     }

@anvil.server.callable
def get_cnn_response(board):
    board = np.array(board)
    bot_player = 'plus'
    human_player = 'minus'

    # 1️⃣ Check for an immediate winning move
    winning_move = find_winning_move(board, bot_player)
    if winning_move is not None:
        print(f"AI playing winning move at column {winning_move}")
        cnn_move = winning_move  # Play the winning move immediately
    else:
        # 2️⃣ Check if the human has a winning move and block it
        blocking_move = find_blocking_move(board, human_player)
        if blocking_move is not None:
            print(f"AI blocking opponent at column {blocking_move}")
            cnn_move = blocking_move  # Block human’s winning move
        else:
            # 3️⃣ If no win/loss risk, use CNN model prediction
            cnn_move = int(predict_cnn(cnn_model, board))
            print(f"AI using CNN model prediction: column {cnn_move}")

    # Check if the move is legal
    legal_moves = find_legal(board)
    if cnn_move not in legal_moves:
        print(f"AI predicted an illegal move ({cnn_move}). Choosing a random legal move.")
        cnn_move = np.random.choice(legal_moves)  # Ensure a valid move is chosen

    # Update the board
    new_board = update_board(board, bot_player, cnn_move)
    board_image = display_board(new_board)
    winner = check_for_win(new_board, cnn_move)

    return {
        'board': new_board.tolist(),
        'image': board_image,
        'winner': winner if isinstance(winner, str) else str(winner),
        'move': cnn_move
    }

@anvil.server.callable
def get_trans_response(board):
    board = np.array(board)
    bot_player = 'plus'
    human_player = 'minus'

    # 1️⃣ Check for an immediate winning move
    winning_move = find_winning_move(board, bot_player)
    if winning_move is not None:
        print(f"AI playing winning move at column {winning_move}")
        trans_move = winning_move
    else:
        # 2️⃣ Check if the human has a winning move and block it
        blocking_move = find_blocking_move(board, human_player)
        if blocking_move is not None:
            print(f"AI blocking opponent at column {blocking_move}")
            trans_move = blocking_move
        else:
            # 3️⃣ If no win/loss risk, use Transformer model prediction
            trans_move = int(predict_transformer(trans_model, board))
            print(f"AI using Transformer model prediction: column {trans_move}")

    # Ensure move is legal
    legal_moves = find_legal(board)
    if trans_move not in legal_moves:
        print(f"AI predicted an illegal move ({trans_move}). Choosing a random legal move.")
        trans_move = np.random.choice(legal_moves)  # Pick a valid move

    # Update board
    new_board = update_board(board, bot_player, trans_move)
    board_image = display_board(new_board)
    winner = check_for_win(new_board, trans_move)

    return {
        'board': new_board.tolist(),
        'image': board_image,
        'winner': winner if isinstance(winner, str) else str(winner),
        'move': trans_move
    }


##########################
# Human Game Logic
##########################
@anvil.server.callable
def process_player_move(board, column):
    board = np.array(board)
    legal_moves = find_legal(board)
    if column not in legal_moves:
        return {"error": "Invalid move! Please try again."}
    new_board = update_board(board, 'minus', column)
    board_image = display_board(new_board)
    winner = check_for_win(new_board, column)

    return {
        'board': new_board.tolist(),
        'image': board_image,
        'winner': winner if isinstance(winner, str) else str(winner)
    }


##########################
# Utility Functions
##########################
def announce_winner(winner):
    """
    Displays the winner of the game.
    """
    if winner == 'tie':
        print("The game is a tie!")
    else:
        print(f"The Winner is: {'AI' if winner[2] == 'p' else 'Human'}!")




##########################
# Keep the Server Running
##########################
if __name__ == "__main__":
    logging.info("Anvil Uplink is starting...")
    connect_to_anvil()
