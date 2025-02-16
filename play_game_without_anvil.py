import numpy as np
import matplotlib.pyplot as plt
import time
from transformer_tensorflow import load_model_for_inference as load_transformer, predict as predict_transformer
from resnet_cnn import load_model_for_inference as load_cnn_model, predict as predict_cnn
from connect4 import update_board, check_for_win, find_legal


##########################
# Fix: Improved Board Display
##########################

def display_board(board):
    plt.clf()  # Clear current plot
    plt.close('all')
    # Set up figure and axis
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#2753C4')  # Darker blue for contrast

    # Draw the board grid
    for row in range(6):
        for col in range(7):
            circle = plt.Circle((col, 5 - row), 0.45,
                                fill=True,
                                facecolor='white' if board[row, col] == 0
                                else '#FF3030' if board[row, col] == 1  # Brighter red
                                else '#FFD700',  # Golden yellow
                                edgecolor='black',
                                linewidth=2)
            ax.add_patch(circle)

    # Set limits and aspect ratio
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 5.5)
    ax.set_aspect('equal')

    # Format labels
    ax.set_xticks(range(7))
    ax.set_xticklabels([str(i) for i in range(7)], fontsize=12, color='white', fontweight='bold')
    ax.set_yticks([])
    
    plt.title("Connect Four", fontsize=18, fontweight='bold', pad=15)
    ax.spines.values().__iter__().__next__().set_visible(False)

    plt.draw()
    plt.pause(0.3)

def animate_drop(board, row, col, player):
    """
    Animates a piece dropping in Connect Four.
    board: Current board state (numpy array)
    row: Final row where the piece lands
    col: Column where the piece is placed
    player: 1 (Red) or 2 (Yellow)
    """
    plt.close('all')
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_facecolor('#2753C4')  # Dark blue background

    # Simulate the dropping animation by placing the piece at each step
    for r in range(row + 1):
        ax.clear()
        
        # Draw the board with previous pieces
        for i in range(6):
            for j in range(7):
                color = 'white' if board[i, j] == 0 else '#FF3030' if board[i, j] == 1 else '#FFD700'
                ax.add_patch(plt.Circle((j, 5 - i), 0.45, fill=True, facecolor=color, edgecolor='black', linewidth=2))
        
        # Animate the new piece dropping
        color = '#FF3030' if player == 1 else '#FFD700'
        ax.add_patch(plt.Circle((col, 5 - r), 0.45, fill=True, facecolor=color, edgecolor='black', linewidth=2))

        # Set plot limits and appearance
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 5.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(7))
        ax.set_xticklabels([str(i) for i in range(7)], fontsize=12, color='white', fontweight='bold')
        ax.set_yticks([])

        plt.title("Connect Four", fontsize=18, fontweight='bold', pad=15)
        ax.spines.values().__iter__().__next__().set_visible(False)

        plt.draw()
        plt.pause(0.1)  # Pause to create the drop effect
    
    plt.pause(0.3)


##########################
# Fix: Game Function (Transformer)
##########################

def play_game_transformer(transformer_model_path):
    """
    Play a game where:
    - The Transformer makes the first move.
    - The human makes the second move.
    - The game alternates between the Transformer and the human.
    """
    model = load_transformer(transformer_model_path)
    print("Transformer model loaded.")

    board = np.zeros((6, 7))  # 6x7 board
    player = 'plus'  # Transformer starts
    winner = 'nobody'

    while winner == 'nobody':
        display_board(board)
        
        legal_moves = find_legal(board)

        # **Fix: Check if the board is full**
        if not legal_moves:
            print("The board is full! It's a tie! 🎭")
            winner = 'tie'
            break

        if player == 'plus':  # Transformer’s turn
            print("\n[Transformer's Turn]")
            
            while True:
                move = predict_transformer(model, board)  # Predict move
                
                if move in legal_moves:  # Ensure it's a legal move
                    print(f"Transformer chooses column: {move}")
                    break
                else:
                    print(f"⚠ Transformer made an illegal move: {move}. Retrying...")

        else:  # Human's turn
            print("\n[Human's Turn]")
            print(f"Legal moves: {legal_moves}")

            while True:
                try:
                    move = int(input(f"Pick a move (0-6): "))
                    if move in legal_moves:
                        break
                    else:
                        print(f"❌ Invalid move. Choose from {legal_moves}.")
                except ValueError:
                    print("⚠ Invalid input. Enter a number between 0 and 6.")

        # Get the row where the piece lands
        row = np.max(np.where(board[:, move] == 0))  # Find the lowest empty row in the column

        # Animate the dropping piece
        animate_drop(board, row, move, 1 if player == 'plus' else 2)

        board = update_board(board, player, move)
        winner = check_for_win(board, move)

        player = 'minus' if player == 'plus' else 'plus'  # Switch turn

    display_board(board)
    print(f"🎉 The Winner is: {'Transformer' if winner[2] == 'p' else 'Human'}!" if winner != 'tie' else "It's a tie!")

    return board, player, winner


##########################
# Fix: Game Function (CNN)
##########################

def play_game_cnn(cnn_model_path):
    """ 
    Play a game where the CNN model plays against a human.
    """
    model = load_cnn_model(cnn_model_path)
    print("CNN model loaded.")

    board = np.zeros((6, 7))
    player = 'plus'
    winner = 'nobody'

    while winner == 'nobody':
        display_board(board)
        
        legal_moves = find_legal(board)

        # **Fix: Check for full board**
        if not legal_moves:
            print("The board is full! It's a tie! 🎭")
            winner = 'tie'
            break

        if player == 'plus':  # CNN’s turn
            print("\n[CNN's Turn]")
            
            while True:
                move = predict_cnn(model, board)
                
                if move in legal_moves:
                    print(f"CNN chooses column: {move}")
                    break
                else:
                    print(f"⚠ CNN made an illegal move: {move}. Retrying...")

        else:  # Human's turn
            print("\n[Human's Turn]")
            print(f"Legal moves: {legal_moves}")

            while True:
                try:
                    move = int(input("Pick a move (0-6): "))
                    if move in legal_moves:
                        break
                    else:
                        print(f"❌ Invalid move. Choose from {legal_moves}.")
                except ValueError:
                    print("⚠ Invalid input. Enter a number between 0 and 6.")

         # Get the row where the piece lands
        row = np.max(np.where(board[:, move] == 0))  # Find the lowest empty row in the column

        # Animate the dropping piece
        animate_drop(board, row, move, 1 if player == 'plus' else 2)
        
        board = update_board(board, player, move)
        winner = check_for_win(board, move)

        player = 'minus' if player == 'plus' else 'plus'  # Switch turn

    display_board(board)
    print(f"🎉 The Winner is: {'CNN' if winner[2] == 'p' else 'Human'}!" if winner != 'tie' else "It's a tie!")

    return board, player, winner


##########################
# Fix: Improved Main Menu
##########################

if __name__ == "__main__":
    cnn_model_path = "cnn_model/best_resnet_model_76.h5"
    transformer_model_path = "trans_model/transformer_best_67.h5"

    while True:
        choice = input("Enter 'CNN' to play against CNN, or 'Transformer' to play against Transformer: ").strip().lower()

        if choice == "cnn":
            print("You chose: CNN\n")
            board, player, winner = play_game_cnn(cnn_model_path)
            break
        elif choice == "transformer":
            print("You chose: Transformer\n")
            board, player, winner = play_game_transformer(transformer_model_path)
            break
        else:
            print("❌ Invalid choice. Please enter 'CNN' or 'Transformer'.")