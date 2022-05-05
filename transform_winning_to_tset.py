import numpy as np


def convert_3_rounds_fs1_obs_to_bin_obs(obs):
    stack_len = 11
    rounds = 3
    
    coin_numbers = [0, 5, 6, 9, 14, 15, 18, 23, 24]
    board_card_numbers = [1,2,3,7,10,11,12,16,19,20,21,25,38,39,40]
    can_bet_numbers = [4,13,22]
    order_numbers = [8,17,26]
    std_number = 41
    last_stack_nr = 37
    bin_rep = []
    onehot_of_stack = np.zeros(stack_len, dtype="float32")
    for i, num in enumerate(obs):
        if i in coin_numbers:
            bin_rep.extend(list("{0:05b}".format(int(num))))
        elif i in board_card_numbers:
            bin_rep.extend(list("{0:04b}".format(int(num))))
        elif i in can_bet_numbers:
            bin_rep.append(int(num))
        elif i in order_numbers:
            bin_rep.extend(list("{0:02b}".format(int(num))))
        elif i == std_number:
            if num > (np.std([1,11]) - np.std([1])) / 2:
                bin_rep.append(1)
            else:
                bin_rep.append(0)
            
        else:
            if i == last_stack_nr:
                onehot_of_stack[-int(num)] = 1
                bin_rep.extend(list(onehot_of_stack))
            else:
                onehot_of_stack[-int(num)] = 1
            
    bin_rep = np.hstack(bin_rep)
    return bin_rep.astype(np.float32)

if __name__ == "__main__":
    tsetlinified_games = []
    for i, game in enumerate(games):
        tset_game = convert_3_rounds_fs1_obs_to_bin_obs(game[0])   
        tsetlinified_games.append((tset_game, game[1]))

    tsetlinified_games = np.array(tsetlinified_games, dtype=object)