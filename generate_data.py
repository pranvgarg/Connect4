import numpy as np
import random
from collections import defaultdict
from multiprocess import Pool
import time
import pandas as pd

def _simulate_single_game(args):


    def find_legal(board):
        legal = [i for i in range(7) if abs(board[0,i]) < 0.1]
        return legal
    
    def update_board(board_temp,color,column):
        # this is a function that takes the current board status, a color, and a column and outputs the new board status
        # columns 0 - 6 are for putting a checker on the board: if column is full just return the current board...this should be forbidden by the player
        
        # the color input should be either 'plus' or 'minus'
        
        board = board_temp.copy()
        ncol = board.shape[1]
        nrow = board.shape[0]
        
        # this seems silly, but actually faster to run than using sum because of overhead! 
        colsum = abs(board[0,column])+abs(board[1,column])+abs(board[2,column])+abs(board[3,column])+abs(board[4,column])+abs(board[5,column])
        row = int(5-colsum)
        if row > -0.5:
            if color == 'plus':
                board[row,column] = 1
            else:
                board[row,column] = -1
        return board

    def mcts(board_temp,color0,nsteps):
        # nsteps is a parameter that determines the skill (and slowness) of the player
        # bigger values of nsteps means the player is better, but also slower to figure out a move.
        board = board_temp.copy()
        ##############################################
        winColumn = look_for_win(board,color0) # check to find a winning column
        if winColumn > -0.5:
            return winColumn # if there is one - play that!
        legal0 = find_all_nonlosers(board,color0) # find all moves that won't immediately lead to your opponent winning
        if len(legal0) == 0: # if you can't block your opponent - just find the 'best' losing move
            legal0 = find_legal(board)
        ##############################################
        # the code above, in between the hash rows, is not part of traditional MCTS
        # but it makes it better and faster - so I included it!
        # MCTS occasionally makes stupid mistakes
        # like not dropping the checker on a winning column, or not blocking an obvious opponent win
        # this avoids a little bit of that stupidity!
        # we could also add this logic to the rest of the MCTS and rollout functions - I just haven't done that yet...
        # feel free to experiment!
        mcts_dict = {tuple(board.ravel()):[0,0]}
        for ijk in range(nsteps):
            color = color0
            winner = 'nobody'
            board_mcts = board.copy()
            path = [tuple(board_mcts.ravel())]
            while winner == 'nobody':
                legal = find_legal(board_mcts)
                if len(legal) == 0:
                    winner = 'tie'
                    back_prop(winner,path,color0,mcts_dict)
                    break
                board_list = []
                for col in legal:
                    board_list.append(tuple(update_board(board_mcts,color,col).ravel()))
                for bl in board_list:
                    if bl not in mcts_dict.keys():
                        mcts_dict[bl] = [0,0]
                ucb1 = np.zeros(len(legal))
                for i in range(len(legal)):
                    num_denom = mcts_dict[board_list[i]]
                    if num_denom[0] == 0:
                        ucb1[i] = 10*nsteps
                    else:
                        ucb1[i] = num_denom[1]/num_denom[0] + 2*np.sqrt(np.log(mcts_dict[path[-1]][0])/mcts_dict[board_list[i]][0])
                chosen = np.argmax(ucb1)
                
                board_mcts = update_board(board_mcts,color,legal[chosen])
                path.append(tuple(board_mcts.ravel()))
                winner = check_for_win(board_mcts,legal[chosen])
                if winner[2]==color[0]:
                    back_prop(winner,path,color0,mcts_dict)
                    break
                if color == 'plus':
                    color = 'minus'
                else:
                    color = 'plus' 
                if mcts_dict[tuple(board_mcts.ravel())][0] == 0:
                    winner = rollout(board_mcts,color)
                    back_prop(winner,path,color0,mcts_dict)
                    break
                
        maxval = -np.inf
        best_col = -1
        for col in legal0:
            board_temp = tuple(update_board(board,color0,col).ravel())
            num_denom = mcts_dict[board_temp]
            if num_denom[0] == 0:
                compare = -np.inf
            else:
                compare = num_denom[1] / num_denom[0]
            if compare > maxval:
                maxval = compare
                best_col = col
        return (best_col)
    
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
                
    def check_for_win(board,col):
        # this code is faster than the above code, but it requires knowing where the last checker was dropped
        # it may seem extreme, but in MCTS this function is called more than anything and actually makes up
        # a large portion of total time spent finding a good move.  So every microsecond is worth saving!
        nrow = 6
        ncol = 7
        # take advantage of knowing what column was last played in...need to check way fewer possibilities
        colsum = abs(board[0,col])+abs(board[1,col])+abs(board[2,col])+abs(board[3,col])+abs(board[4,col])+abs(board[5,col])
        row = int(6-colsum)
        if row+3<6:
            vert = board[row,col] + board[row+1,col] + board[row+2,col] + board[row+3,col]
            if vert == 4:
                return 'v-plus'
            elif vert == -4:
                return 'v-minus'
        if col+3<7:
            hor = board[row,col] + board[row,col+1] + board[row,col+2] + board[row,col+3]
            if hor == 4:
                return 'h-plus'
            elif hor == -4:
                return 'h-minus'
        if col-1>=0 and col+2<7:
            hor = board[row,col-1] + board[row,col] + board[row,col+1] + board[row,col+2]
            if hor == 4:
                return 'h-plus'
            elif hor == -4:
                return 'h-minus'
        if col-2>=0 and col+1<7:
            hor = board[row,col-2] + board[row,col-1] + board[row,col] + board[row,col+1]
            if hor == 4:
                return 'h-plus'
            elif hor == -4:
                return 'h-minus'
        if col-3>=0:
            hor = board[row,col-3] + board[row,col-2] + board[row,col-1] + board[row,col]
            if hor == 4:
                return 'h-plus'
            elif hor == -4:
                return 'h-minus'
        if row < 3 and col < 4:
            DR = board[row,col] + board[row+1,col+1] + board[row+2,col+2] + board[row+3,col+3]
            if DR == 4:
                return 'd-plus'
            elif DR == -4:
                return 'd-minus'
        if row-1>=0 and col-1>=0 and row+2<6 and col+2<7:
            DR = board[row-1,col-1] + board[row,col] + board[row+1,col+1] + board[row+2,col+2]
            if DR == 4:
                return 'd-plus'
            elif DR == -4:
                return 'd-minus'
        if row-2>=0 and col-2>=0 and row+1<6 and col+1<7:
            DR = board[row-2,col-2] + board[row-1,col-1] + board[row,col] + board[row+1,col+1]
            if DR == 4:
                return 'd-plus'
            elif DR == -4:
                return 'd-minus'
        if row-3>=0 and col-3>=0:
            DR = board[row-3,col-3] + board[row-2,col-2] + board[row-1,col-1] + board[row,col]
            if DR == 4:
                return 'd-plus'
            elif DR == -4:
                return 'd-minus'
        if row+3<6 and col-3>=0:
            DL = board[row,col] + board[row+1,col-1] + board[row+2,col-2] + board[row+3,col-3]
            if DL == 4:
                return 'd-plus'
            elif DL == -4:
                return 'd-minus'
        if row-1 >= 0 and col+1 < 7 and row+2<6 and col-2>=0:
            DL = board[row-1,col+1] + board[row,col] + board[row+1,col-1] + board[row+2,col-2]
            if DL == 4:
                return 'd-plus'
            elif DL == -4:
                return 'd-minus'
        if row-2 >=0 and col+2<7 and row+1<6 and col-1>=0:
            DL = board[row-2,col+2] + board[row-1,col+1] + board[row,col] + board[row+1,col-1]
            if DL == 4:
                return 'd-plus'
            elif DL == -4:
                return 'd-minus'
        if row-3>=0 and col+3<7:
            DL = board[row-3,col+3] + board[row-2,col+2] + board[row-1,col+1] + board[row,col]
            if DL == 4:
                return 'd-plus'
            elif DL == -4:
                return 'd-minus'
        return 'nobody'
    
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
    
    def rollout(board,next_player):
        winner = 'nobody'
        player = next_player
        while winner == 'nobody':
            legal = find_legal(board)
            if len(legal) == 0:
                winner = 'tie'
                return winner
            move = random.choice(legal)
            board = update_board(board,player,move)
            winner = check_for_win(board,move)
            
            if player == 'plus':
                player = 'minus'
            else:
                player = 'plus'
        return winner
    
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

    def board_to_tensor(board):
        """
        Convert a 6x7 board with +1, -1, and 0 values to a 6x7x2 tensor.

        Args:
            board (np.ndarray): 6x7 board.

        Returns:
            np.ndarray: 6x7x2 tensor representation of the board.
        """
        tensor = np.zeros((6, 7, 2))
        tensor[:, :, 0] = (board == 1).astype(int)  # Channel for 'plus'
        tensor[:, :, 1] = (board == -1).astype(int)  # Channel for 'minus'
        return tensor


    game_idx, n_games, min_steps_mcts, max_steps_mcts, max_random_moves = args
    X_local = []
    Y_local = []

    board = np.zeros((6, 7))
    current_player = random.choice(['plus', 'minus'])

    n_random_moves = random.randint(0, max_random_moves)
    for _ in range(n_random_moves):
        legal_moves = find_legal(board)
        if not legal_moves:
            break
        random_move = random.choice(legal_moves)
        board = update_board(board, current_player, random_move)
        current_player = 'minus' if current_player == 'plus' else 'plus'

    winner = 'nobody'
    while winner == 'nobody':
        legal_moves = find_legal(board)
        if not legal_moves:
            break

        n_steps = random.randint(min_steps_mcts, max_steps_mcts)
        best_move = mcts(board, current_player, n_steps)

        X_local.append(board_to_tensor(board))
        Y_local.append(best_move)
        
        board = update_board(board, current_player, best_move)
        winner = check_for_win(board, best_move)
        current_player = 'minus' if current_player == 'plus' else 'plus'

    print(f"Game {game_idx + 1}/{n_games} completed. Winner: {winner}")
    return X_local, Y_local

def generate_dataset(n_games, min_steps_mcts=750, max_steps_mcts=5000, max_random_moves=15):
    args_list = [
        (idx, n_games, min_steps_mcts, max_steps_mcts, max_random_moves)
        for idx in range(n_games)
    ]

    with Pool(processes=10) as pool:
        results = pool.map(_simulate_single_game, args_list)

    X_all, Y_all = [], []
    for X_game, Y_game in results:
        X_all.extend(X_game)
        Y_all.extend(Y_game)

    dataset = defaultdict(list)
    for board, move in zip(X_all, Y_all):
        dataset[tuple(board.ravel())].append(move)

    X_dedup, Y_dedup = [], []
    for board, moves in dataset.items():
        X_dedup.append(np.array(board).reshape(6, 7, 2))
        Y_dedup.append(max(set(moves), key=moves.count))

    return np.array(X_dedup), np.array(Y_dedup)

def mirror_board_and_move(board, move):
    mirrored_board = np.flip(board, axis=1)
    mirrored_move = 6 - move
    return mirrored_board, mirrored_move


def save_dataset(n_games, file_boards, file_moves):
    X, Y = generate_dataset(n_games)

    # Generate mirrored instances
    mirrored_boards, mirrored_moves = zip(*[mirror_board_and_move(board, move) for board, move in zip(X, Y)])

    # Combine original and mirrored datasets
    all_boards = np.concatenate([X, np.array(mirrored_boards)], axis=0)
    all_moves = np.concatenate([Y, np.array(mirrored_moves)], axis=0)

    # Remove duplicates, keeping the most frequent move per board
    unique_boards, indices = np.unique([board.tobytes() for board in all_boards], return_index=True)
    final_boards = all_boards[indices]
    final_moves = all_moves[indices]

    print(f"Final dataset size: {len(final_boards)}")

    # Save to .npy files
    np.save(file_boards, final_boards)
    np.save(file_moves, final_moves)
    print(f"Saved dataset to {file_boards} and {file_moves}")


if __name__ == "__main__":
    total_games_to_generate = 3
    file_boards="test/connect4_boards.npy"
    file_moves="test/connect4_moves.npy"
    save_dataset(total_games_to_generate, file_boards, file_moves)
