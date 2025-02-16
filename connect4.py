import numpy as np # type: ignore
import time
import random

########################################
# Define all your functions here
########################################


# def update_board(board_temp,color,column):
#     # this is a function that takes the current board status, a color, and a column and outputs the new board status
#     # columns 0 - 6 are for putting a checker on the board: if column is full just return the current board...this should be forbidden by the player
    
#     # the color input should be either 'plus' or 'minus'
    
#     board = board_temp.copy()
#     ncol = board.shape[1]
#     nrow = board.shape[0]
    
#     # this seems silly, but actually faster to run than using sum because of overhead! 
#     colsum = abs(board[0,column])+abs(board[1,column])+abs(board[2,column])+\
#         abs(board[3,column])+abs(board[4,column])+abs(board[5,column])
#     row = int(5-colsum)
#     if row > -0.5:
#         if color == 'plus':
#             board[row,column] = 1
#         else:
#             board[row,column] = -1
#     return board

def update_board(board_temp, color, column):
    """
    Updates the board by placing a piece of the given color in the specified column.
    
    :param board_temp: Current board state (6x7 NumPy array)
    :param color: 'plus' (1) or 'minus' (-1)
    :param column: Column index (0-6) where the piece should be dropped
    :return: Updated board state
    """
    board = board_temp.copy()
    nrow = board.shape[0]

    # Calculate the number of occupied cells in the column
    colsum = np.abs(board[:, column]).sum()  # ✅ Now always a single integer

    # Find the row where the piece will land
    row = int(nrow - 1 - colsum)  # ✅ Ensures row is a single integer

    if row >= 0:  # If column is not full
        board[row, column] = 1 if color == 'plus' else -1
    return board
    
    # in this code the board is a 6x7 numpy array.  Each entry is +1, -1 or 0.  You WILL be able to do a better
    # job training your neural network if you rearrange this to be a 6x7x2 numpy array.  If the i'th row and j'th
    # column is +1, this can be represented by board[i,j,0]=1.  If it is -1, this can be represented by
    # board[i,j,1]=1. It's up to you how you represent your board.

def make_move(board, color, column):
    """
    Place a piece of the given color in 'column' on 'board' in-place.
    Returns the row where the piece was placed (or -1 if column is full).
    color is 'plus' or 'minus'.
    """
    color_val = 1 if color == 'plus' else -1
    for row in reversed(range(board.shape[0])):  # 5..0
        if board[row, column] == 0:
            board[row, column] = color_val
            return row
    return -1

def undo_move(board, row, column):
    board[row, column] = 0

def check_for_win_slow(board):
    # this function checks to see if anyone has won on the given board
    nrow = board.shape[0]
    ncol = board.shape[1]
    winner = 'nobody'
    for col in range(ncol):
        for row in reversed(range(nrow)):
            if abs(board[row,col]) < 0.1: # if this cell is empty, all the cells above it are too!
                break
            # check for vertical winners
            if row <= (nrow-4): # can't have a column go from rows 4-7...
                tempsum = board[row,col]+board[row+1,col]+board[row+2,col]+board[row+3,col] # this is WAY faster than np.sum!!!
                if tempsum==4:
                    winner = 'v-plus'
                    return winner
                elif tempsum==-4:
                    winner = 'v-minus'
                    return winner
            # check for horizontal winners
            if col <= (ncol-4):
                tempsum = board[row,col]+board[row,col+1]+board[row,col+2]+board[row,col+3]
                if tempsum==4:
                    winner = 'h-plus'
                    return winner
                elif tempsum==-4:
                    winner = 'h-minus'
                    return winner
            # check for top left to bottom right diagonal winners
            if (row <= (nrow-4)) and (col <= (ncol-4)):
                tempsum = board[row,col]+board[row+1,col+1]+board[row+2,col+2]+board[row+3,col+3]
                if tempsum==4:
                    winner = 'd-plus'
                    return winner
                elif tempsum==-4:
                    winner = 'd-minus'
                    return winner
            # check for top right to bottom left diagonal winners
            if (row <= (nrow-4)) and (col >= 3):
                tempsum = board[row,col]+board[row+1,col-1]+board[row+2,col-2]+board[row+3,col-3]
                if tempsum==4:
                    winner = 'd-plus'
                    return winner
                elif tempsum==-4:
                    winner = 'd-minus'
                    return winner
    return winner

# def check_for_win(board,col):
#     # this code is faster than the above code, but it requires knowing where the last checker was dropped
#     # it may seem extreme, but in MCTS this function is called more than anything and actually makes up
#     # a large portion of total time spent finding a good move.  So every microsecond is worth saving!
#     nrow = 6
#     ncol = 7
#     # take advantage of knowing what column was last played in...need to check way fewer possibilities
#     colsum = abs(board[0,col])+abs(board[1,col])+abs(board[2,col])+abs(board[3,col])+abs(board[4,col])+abs(board[5,col])
#     row = int(6-colsum)
#     if row+3<6:
#         vert = board[row,col] + board[row+1,col] + board[row+2,col] + board[row+3,col]
#         if vert == 4:
#             return 'v-plus'
#         elif vert == -4:
#             return 'v-minus'
#     if col+3<7:
#         hor = board[row,col] + board[row,col+1] + board[row,col+2] + board[row,col+3]
#         if hor == 4:
#             return 'h-plus'
#         elif hor == -4:
#             return 'h-minus'
#     if col-1>=0 and col+2<7:
#         hor = board[row,col-1] + board[row,col] + board[row,col+1] + board[row,col+2]
#         if hor == 4:
#             return 'h-plus'
#         elif hor == -4:
#             return 'h-minus'
#     if col-2>=0 and col+1<7:
#         hor = board[row,col-2] + board[row,col-1] + board[row,col] + board[row,col+1]
#         if hor == 4:
#             return 'h-plus'
#         elif hor == -4:
#             return 'h-minus'
#     if col-3>=0:
#         hor = board[row,col-3] + board[row,col-2] + board[row,col-1] + board[row,col]
#         if hor == 4:
#             return 'h-plus'
#         elif hor == -4:
#             return 'h-minus'
#     if row < 3 and col < 4:
#         DR = board[row,col] + board[row+1,col+1] + board[row+2,col+2] + board[row+3,col+3]
#         if DR == 4:
#             return 'd-plus'
#         elif DR == -4:
#             return 'd-minus'
#     if row-1>=0 and col-1>=0 and row+2<6 and col+2<7:
#         DR = board[row-1,col-1] + board[row,col] + board[row+1,col+1] + board[row+2,col+2]
#         if DR == 4:
#             return 'd-plus'
#         elif DR == -4:
#             return 'd-minus'
#     if row-2>=0 and col-2>=0 and row+1<6 and col+1<7:
#         DR = board[row-2,col-2] + board[row-1,col-1] + board[row,col] + board[row+1,col+1]
#         if DR == 4:
#             return 'd-plus'
#         elif DR == -4:
#             return 'd-minus'
#     if row-3>=0 and col-3>=0:
#         DR = board[row-3,col-3] + board[row-2,col-2] + board[row-1,col-1] + board[row,col]
#         if DR == 4:
#             return 'd-plus'
#         elif DR == -4:
#             return 'd-minus'
#     if row+3<6 and col-3>=0:
#         DL = board[row,col] + board[row+1,col-1] + board[row+2,col-2] + board[row+3,col-3]
#         if DL == 4:
#             return 'd-plus'
#         elif DL == -4:
#             return 'd-minus'
#     if row-1 >= 0 and col+1 < 7 and row+2<6 and col-2>=0:
#         DL = board[row-1,col+1] + board[row,col] + board[row+1,col-1] + board[row+2,col-2]
#         if DL == 4:
#             return 'd-plus'
#         elif DL == -4:
#             return 'd-minus'
#     if row-2 >=0 and col+2<7 and row+1<6 and col-1>=0:
#         DL = board[row-2,col+2] + board[row-1,col+1] + board[row,col] + board[row+1,col-1]
#         if DL == 4:
#             return 'd-plus'
#         elif DL == -4:
#             return 'd-minus'
#     if row-3>=0 and col+3<7:
#         DL = board[row-3,col+3] + board[row-2,col+2] + board[row-1,col+1] + board[row,col]
#         if DL == 4:
#             return 'd-plus'
#         elif DL == -4:
#             return 'd-minus'
#     return 'nobody'

def check_for_win(board, col):
    """
    Checks if the last move resulted in a win.
    
    :param board: Current game board (6x7 NumPy array)
    :param col: The column index where the last piece was placed
    :return: Winning pattern ('v-plus', 'h-minus', 'd-plus', etc.) or 'nobody' if no winner
    """
    nrow, ncol = board.shape  # Get board dimensions

    # Find the row where the last piece was dropped
    row = -1
    for r in range(nrow):  # Start from top and find first occupied row
        if board[r, col] != 0:
            row = r
            break

    if row == -1:
        return 'nobody'  # No piece found in this column (should never happen in a valid move)

    player = board[row, col]  # Get the player piece (1 or -1)

    ### **1️⃣ Vertical Check (|)**
    if row + 3 < nrow:
        if all(board[row + i, col] == player for i in range(4)):
            return f'v-{"plus" if player == 1 else "minus"}'

    ### **2️⃣ Horizontal Check (-)**
    count = 1  # Start with the current piece
    for i in range(1, 4):  # Check right
        if col + i < ncol and board[row, col + i] == player:
            count += 1
        else:
            break
    for i in range(1, 4):  # Check left
        if col - i >= 0 and board[row, col - i] == player:
            count += 1
        else:
            break
    if count >= 4:
        return f'h-{"plus" if player == 1 else "minus"}'

    ### **3️⃣ Diagonal Right-Down (\) Check**
    count = 1
    for i in range(1, 4):  # Check down-right
        if row + i < nrow and col + i < ncol and board[row + i, col + i] == player:
            count += 1
        else:
            break
    for i in range(1, 4):  # Check up-left
        if row - i >= 0 and col - i >= 0 and board[row - i, col - i] == player:
            count += 1
        else:
            break
    if count >= 4:
        return f'd-{"plus" if player == 1 else "minus"}'

    ### **4️⃣ Diagonal Left-Down (/) Check**
    count = 1
    for i in range(1, 4):  # Check down-left
        if row + i < nrow and col - i >= 0 and board[row + i, col - i] == player:
            count += 1
        else:
            break
    for i in range(1, 4):  # Check up-right
        if row - i >= 0 and col + i < ncol and board[row - i, col + i] == player:
            count += 1
        else:
            break
    if count >= 4:
        return f'd-{"plus" if player == 1 else "minus"}'

    return 'nobody'  # No winner found

def find_legal(board):
    if board.shape == (6, 7, 2):
        board = board[:, :, 0] - board[:, :, 1]  # Convert (6,7,2) to (6,7)
        
    legal = [i for i in range(7) if abs(board[0,i]) < 0.1]
    return legal

def look_for_win(board_,color):
    board_ = board_.copy()
    legal = find_legal(board_)
    winner = -1
    for m in legal:
        bt = update_board(board_.copy(),color,m)
        wi = check_for_win(bt,m)
        if wi[2:] == color:
            winner = m
            break
    return winner

def find_all_nonlosers(board,color):
    if color == 'plus':
        opp = 'minus'
    else:
        opp = 'plus'
    legal = find_legal(board)
    poss_boards = [update_board(board,color,l) for l in legal]
    poss_legal = [find_legal(b) for b in poss_boards]
    allowed = []
    for i in range(len(legal)):
        wins = [j for j in poss_legal[i] if check_for_win(update_board(poss_boards[i],opp,j),j) != 'nobody']
        if len(wins) == 0:
            allowed.append(legal[i])
    return allowed

def back_prop(winner,path,color0,md):
    for i in range(len(path)):
        board_temp = path[i]
        
        md[board_temp][0]+=1
        if winner[2]==color0[0]:
            if i % 2 == 1:
                md[board_temp][1] += 1
            else:
                md[board_temp][1] -= 1
        elif winner[2]=='e': # tie
            # md[board_temp][1] += 0
            pass
        else:
            if i % 2 == 1:
                md[board_temp][1] -= 1
            else:
                md[board_temp][1] += 1

def rollout(board, next_player, debug=False):
    winner = 'nobody'
    player = next_player
    move_count = 0

    while winner == 'nobody':
        legal = find_legal(board)
        if not legal:
            winner = 'tie'
            if debug:
                print("[Rollout] No legal moves left, tie.")
            return winner

        move = random.choice(legal)
        board = update_board(board, player, move)
        move_count += 1

        winner = check_for_win(board, move)
        if debug:
            print(f"[Rollout] Player {player} plays col {move}, move #{move_count}. Winner? {winner}")

        if player == 'plus':
            player = 'minus'
        else:
            player = 'plus'
    return winner

def mcts(board_temp,color0,nsteps):
    # nsteps is a parameter that determines the skill (and slowness) of the player
    # bigger values of nsteps means the player is better, but also slower to figure out a move.

    # print(f"\n[MCTS] Starting search for color: {color0} with {nsteps} iterations.")
    board = board_temp.copy()
    ##############################################
    # Optional: pre-check for immediate win
    winColumn = look_for_win(board,color0) # check to find a winning column
    if winColumn > -0.5:
        # print(f"[MCTS] Found immediate winning column: {winColumn}")
        return winColumn # if there is one - play that!
    legal0 = find_all_nonlosers(board,color0) # find all moves that won't immediately lead to your opponent winning
    if len(legal0) == 0: # if you can't block your opponent - just find the 'best' losing move
        legal0 = find_legal(board)
        # print("[MCTS] No non-losing moves found; using any legal move.")
    ##############################################
    # the code above, in between the hash rows, is not part of traditional MCTS
    # but it makes it better and faster - so I included it!
    # MCTS occasionally makes stupid mistakes
    # like not dropping the checker on a winning column, or not blocking an obvious opponent win
    # this avoids a little bit of that stupidity!
    # we could also add this logic to the rest of the MCTS and rollout functions - I just haven't done that yet...
    # feel free to experiment!

    mcts_dict = {tuple(board.ravel()): [0, 0]}

    for step in range(nsteps):
        # # Optionally print every X steps to avoid spamming
        # if step % 20 == 0:
        #     print(f"[MCTS] Iteration {step}/{nsteps}")

        color = color0
        winner = 'nobody'
        board_mcts = board.copy()
        path = [tuple(board_mcts.ravel())]

        while winner == 'nobody':
            legal = find_legal(board_mcts)
            if len(legal) == 0:
                winner = 'tie'
                back_prop(winner, path, color0, mcts_dict)
                break

            board_list = []
            for col in legal:
                board_list.append(tuple(update_board(board_mcts, color, col).ravel()))

            for bl in board_list:
                if bl not in mcts_dict.keys():
                    mcts_dict[bl] = [0, 0]

            ucb1 = np.zeros(len(legal))
            for i in range(len(legal)):
                num_denom = mcts_dict[board_list[i]]
                if num_denom[0] == 0:
                    ucb1[i] = 10*nsteps
                else:
                    ucb1[i] = (
                        num_denom[1]/num_denom[0] 
                        + 2*np.sqrt(
                            np.log(mcts_dict[path[-1]][0]) / mcts_dict[board_list[i]][0]
                        )
                    )

            chosen = np.argmax(ucb1)
            board_mcts = update_board(board_mcts, color, legal[chosen])
            path.append(tuple(board_mcts.ravel()))
            winner = check_for_win(board_mcts, legal[chosen])

            # If we found a winner...
            if winner[2] == color[0]:
                back_prop(winner, path, color0, mcts_dict)
                # Optional: debug print
                # print(f"[MCTS] Found winning move for {color} at col {legal[chosen]}.")
                break

            # Switch player
            color = 'minus' if color == 'plus' else 'plus'

            # If this child was never visited, rollout from here
            if mcts_dict[tuple(board_mcts.ravel())][0] == 0:
                winner = rollout(board_mcts, color)
                back_prop(winner, path, color0, mcts_dict)
                break

    # After nsteps, pick the best column
    maxval = -np.inf
    best_col = -1
    for col in legal0:
        board_hash = tuple(update_board(board, color0, col).ravel())
        num_denom = mcts_dict[board_hash]
        if num_denom[0] == 0:
            compare = -np.inf
        else:
            compare = num_denom[1] / num_denom[0]
        if compare > maxval:
            maxval = compare
            best_col = col

    # print(f"[MCTS] Best column for {color0} after {nsteps} iter: {best_col} (score={maxval:.3f})\n")
    return best_col

def display_board(board):
    # this function displays the board as ascii using X for +1 and O for -1
    # For the project, this should be a better picture of the board...
    # clear_output()
    horizontal_line = '-'*(7*5+8)
    blank_line = '|'+' '*5
    blank_line *= 7
    blank_line += '|'
    print('   0     1     2     3     4     5     6')
    print(horizontal_line)
    for row in range(6):
        print(blank_line)
        this_line = '|'
        for col in range(7):
            if board[row,col] == 0:
                this_line += ' '*5 + '|'
            elif board[row,col] == 1:
                this_line += '  X  |'
            else:
                this_line += '  O  |'
        print(this_line)
        print(blank_line)
        print(horizontal_line)
    print('   0     1     2     3     4     5     6')


########################################
# Now the main block that runs when you do "python connect4.py"
########################################

if __name__ == "__main__":
    # Set MCTS skill level (higher means better AI but slower)
    mcts_steps = 100  

    # Initialize an empty board (6x7)
    board = np.zeros((6, 7))

    # Decide if the human is playing 'plus' (first) or 'minus' (second)
    human_color = input("Do you want to go first? (yes/no): ").strip().lower()
    if human_color in ["yes", "y"]:
        human_color = 'plus'
        ai_color = 'minus'
    else:
        human_color = 'minus'
        ai_color = 'plus'

    # Start the game loop
    current_color = 'plus'  # 'plus' always goes first
    winner = 'nobody'

    while winner == 'nobody':
        display_board(board)  # Show the current board state
        
        if current_color == human_color:
            # Human Turn
            valid_move = False
            while not valid_move:
                try:
                    col = int(input(f"Your turn ({human_color}). Choose a column (0-6): "))
                    if col in find_legal(board):
                        valid_move = True
                    else:
                        print("Column is full or invalid. Try again.")
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 6.")
        else:
            # AI Turn
            print(f"AI ({ai_color}) is thinking...")
            col = mcts(board, ai_color, mcts_steps)
            time.sleep(1)  # Small delay for realism

        # Apply the move to the board
        board = update_board(board, current_color, col)
        winner = check_for_win(board, col)

        # Switch turns
        current_color = 'minus' if current_color == 'plus' else 'plus'

    # Final board display and result
    display_board(board)
    if winner[2:] == human_color:
        print("Congratulations! You won! 🎉")
    elif winner[2:] == ai_color:
        print("AI wins! Better luck next time. 🤖")
    else:
        print("It's a tie! 🤝")


