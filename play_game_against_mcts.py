import numpy as np  # type: ignore
from transformer_tensorflow import load_model_for_inference as load_transformer, predict as predict_transformer
from resnet_cnn import load_model_for_inference as load_cnn_model, predict as predict_cnn
from connect4 import update_board, check_for_win, find_legal, display_board, mcts

def play_game_transformer(transformer_model_path, mcts_steps=3000):
    # Load Transformer model
    model = load_transformer(transformer_model_path)
    print("Transformer model loaded.")

    # Initialize game board
    board = np.zeros((6, 7))  # 6 rows x 7 columns
    player = 'plus'
    winner = 'nobody'

    while winner == 'nobody':
        print("\nCurrent Board:")
        display_board(board)

        if player == 'plus':  # MCTS's turn
            print("\n[MCTS's Turn]")
            move = mcts(board, player, mcts_steps)
            print(f"MCTS chose column: {move}")
        else:  # Human's turn with Transformer suggestion
            print("\n[Human's Turn]")
            # Encode board for Transformer
            encoded_board = np.zeros((6, 7, 2))
            encoded_board[:, :, 0] = (board == 1)  # Plus player
            encoded_board[:, :, 1] = (board == -1)  # Minus player
            # print(encoded_board)

            suggested_move = predict_transformer(model, encoded_board)
            print(f"Transformer suggests column: {suggested_move}")
            legal_moves = find_legal(board)
            print(f"Legal moves: {legal_moves}")

            if suggested_move not in legal_moves:
                print(f"⚠️ Suggested move {suggested_move} is not legal.")
                print("Please choose a valid move manually.")

            while True:
                try:
                    move = int(input(f"Pick a move (0-6), suggested: {suggested_move}: "))
                    if move in legal_moves:
                        break
                    else:
                        print(f"Illegal move. Choose from {legal_moves}.")
                except ValueError:
                    print("Invalid input. Enter a number between 0 and 6.")

        # Update the board
        board = update_board(board, player, move)

        # Check for a winner
        winner = check_for_win(board, move)

        # Switch player
        player = 'minus' if player == 'plus' else 'plus'

    # Final Board Display
    display_board(board)
    print(f"\nThe Winner is: {'Transformer' if winner[2] == 'p' else 'Human'}!")
    return board, player, winner



##############################
# Play Against CNN AI
##############################
def play_game_cnn(cnn_model_path, mcts_steps=3000):
    """
    Play against CNN AI:
    - CNN makes the first move.
    - Human makes the second move.
    - The game alternates turns.
    - CNN suggests moves, but human is free to choose.
    """
    # Load CNN model
    model = load_cnn_model(cnn_model_path)
    print("CNN model loaded.")

    # Initialize game board
    board = np.zeros((6, 7))  # 6 rows x 7 columns
    player = 'plus'  # CNN starts
    winner = 'nobody'

    while winner == 'nobody':
        print("\nCurrent Board:")
        display_board(board)
        # legal_moves = find_legal(board)

        if player == 'plus':  # MCTS's turn
            print("\n[MCTS's Turn]")
            move = mcts(board, player, mcts_steps)
            print(f"MCTS chose column: {move}")

        else:  # Human's turn
            print("\n[Human's Turn]")
            print(f"Legal moves: {legal_moves}")

            suggested_move = predict_cnn(model, board)
            print(f"CNN suggests column: {suggested_move}")
            legal_moves = find_legal(board)
            print(f"Legal moves: {legal_moves}")

            if suggested_move not in legal_moves:
                print(f"⚠️ Suggested move {suggested_move} is not legal.")
                print("Please choose a valid move manually.")

            while True:
                try:
                    move = int(input(f"Pick a move (0-6), suggested: {suggested_move}: "))
                    if move in legal_moves:
                        break
                    else:
                        print(f"Illegal move. Choose from {legal_moves}.")
                except ValueError:
                    print("Invalid input. Enter a number between 0 and 6.")

        # Update the board with the chosen move
        board = update_board(board, player, move)

        # Check if the game is won
        winner = check_for_win(board, move)

        # Switch turn
        player = 'minus' if player == 'plus' else 'plus'

    # Final Board Display
    display_board(board)
    print(f"\nThe Winner is: {'CNN' if winner[2] == 'p' else 'Human'}!")
    return board, player, winner

##############################
# Main: Choose CNN or Transformer AI
##############################
if __name__ == "__main__":
    # Set default model paths (adjust as needed)
    cnn_model_path = "cnn_model/best_resnet_model.h5"
    transformer_model_path = "trans_model/best_transformer_model.h5"

    # Keep prompting the user until they enter a valid choice
    while True:
        choice = input("Enter '1' to play against CNN, or '2' to play against Transformer: ").strip()

        if choice == "1":
            print("You chose: CNN\n")
            board, player, winner = play_game_cnn(cnn_model_path)
            break  # Exit loop after valid selection
        elif choice == "2":
            print("You chose: Transformer\n")
            board, player, winner = play_game_transformer(transformer_model_path)
            break  # Exit loop after valid selection
        else:
            print("❌ Invalid choice. Please enter '1' for CNN or '2' for Transformer.")