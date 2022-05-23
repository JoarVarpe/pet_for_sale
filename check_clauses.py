from matplotlib.pyplot import text
from tmu.tsetlin_machine import TMCoalescedClassifier
import numpy as np 
import pickle
import torch
import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from apriori_python import apriori

n_clauses_per_class = 8000
treshold = int(n_clauses_per_class * 0.8)
s_param = 11.40000000000001

game_amount = 100000
# tsetlin_filename, self_can_bet_index= "/home/jaoi/master22/pet_for_sale/winning_games_db/{}_tsetlined_games.pkl".format(game_amount), 17
tsetlin_filename, self_can_bet_index = "/home/jaoi/master22/pet_for_sale/winning_games_db/4PPO_{}_tsetlined_games.pkl".format(game_amount), 21

def get_name_of_feature_from_4_rounds(feature_nr):
    # for own player with 4 bits representing the cards as max number is 11
    reverse_4_bits = [3, 2, 1, 0]
    reverse_5_bits = [4, 3, 2, 1, 0]
    reverse_2_bits = [1, 0]
    if feature_nr in [0,1,2,3,4]:
        return "own coins bit {}".format(2**(reverse_5_bits[feature_nr]))
    elif feature_nr in [5,6,7,8]:
        return "own nr 1 card bit {}".format(2**reverse_4_bits[feature_nr - 5])
    elif feature_nr in [9,10,11,12]:
        return "own nr 2 card bit".format(2**reverse_4_bits[feature_nr - 9])
    elif feature_nr in [13,14,15,16]:
        return "own nr 3 card bit".format(2**reverse_4_bits[feature_nr - 13])
    elif feature_nr in [17,18,19,20]:
        return "own nr 4 card bit".format(2**reverse_4_bits[feature_nr - 17])
    elif feature_nr == 21:
        return "self can bid bit"
    elif feature_nr in [22,23,24,25,26]:
        return "current own bid {}".format(2**reverse_5_bits[feature_nr - 22])
    elif feature_nr in [27,28,29,30,31]:
        return "resulting own coins if fold {}".format(2**reverse_5_bits[feature_nr - 27])
    elif feature_nr in [32,33,34,35]:
        return "resulting card on fold {}".format(2**reverse_4_bits[feature_nr - 32])
    elif feature_nr in [36,37]:
        return "self next turn order {}".format(2**reverse_2_bits[feature_nr - 36])
    # for next player in line
    elif feature_nr in [38,39,40,41,42]:
        return "own coins bit {}".format(2**(reverse_5_bits[feature_nr - 38]))
    elif feature_nr in [43,44,45,46]:
        return "own nr 1 card bit {}".format(2**reverse_4_bits[feature_nr - 43])
    elif feature_nr in [47,48,49,50]:
        return "own nr 2 card bit".format(2**reverse_4_bits[feature_nr - 47])
    elif feature_nr in [51,52,53,54]:
        return "own nr 3 card bit".format(2**reverse_4_bits[feature_nr - 51])
    elif feature_nr in [55,56,57,58]:
        return "own nr 3 card bit".format(2**reverse_4_bits[feature_nr - 55])
    elif feature_nr == 59:
        return "next can bid bit"
    elif feature_nr in [60, 61, 62, 63, 64]:
        return "current next player bid {} bit".format(2**reverse_5_bits[feature_nr - 60])
    elif feature_nr in [65, 66, 67, 68, 69]:
        return "resulting next player coins if fold {}".format(2**reverse_5_bits[feature_nr - 65])
    elif feature_nr in [70, 71, 72, 73]:
        return "resulting card on fold {}".format(2**reverse_4_bits[feature_nr - 70])
    elif feature_nr in [74, 75]:
        return "next player turn order {}".format(2**reverse_2_bits[feature_nr - 74])
    # for last player in line
    elif feature_nr in [76,77,78,79,80]:
        return "own coins bit {}".format(2**(reverse_5_bits[feature_nr - 76]))
    elif feature_nr in [81,82,83,84]:
        return "own nr 1 card bit {}".format(2**reverse_4_bits[feature_nr - 81])
    elif feature_nr in [85,86,87,88]:
        return "own nr 2 card bit".format(2**reverse_4_bits[feature_nr - 85])
    elif feature_nr in [89,90,91,92]:
        return "own nr 3 card bit".format(2**reverse_4_bits[feature_nr - 89])
    elif feature_nr in [93,94,95,96]:
        return "own nr 3 card bit".format(2**reverse_4_bits[feature_nr - 93])
    elif feature_nr == 97:
        return "last player can bid"
    elif feature_nr in [98, 99, 100, 101, 102]:
        return "current next player bid {}".format(2**reverse_5_bits[feature_nr - 98])
    elif feature_nr in [103, 104, 105, 106, 107]:
        return "resulting next player coins if fold {}".format(2**reverse_5_bits[feature_nr - 103])
    elif feature_nr in [108, 109, 110, 111]:
        return "resulting card on fold {}".format(2**reverse_4_bits[feature_nr - 108])
    elif feature_nr in [112, 113]:
        return "last player turn order {}".format(2**reverse_2_bits[feature_nr - 112])
    # remainder stats
    elif feature_nr in [114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128]:
        return "{}. largest in stack".format(feature_nr - 114)
    elif feature_nr in [129, 130, 131, 132]:
        return "last player cards 3. bit {}".format(2**reverse_4_bits[feature_nr - 129])
    elif feature_nr in [133, 134, 135, 136]:
        return "last player cards 3. bit {}".format(2**reverse_4_bits[feature_nr - 133])
    elif feature_nr in [137, 138, 139, 140]:
        return "last player cards 3. bit {}".format(2**reverse_4_bits[feature_nr - 137])
    elif feature_nr == 141:
        return "board_std"
    return "not implemented"

def show_coalesced_clause_4r(tm, clause):
    number_of_features = 142
    print("Clause #%d: " % (clause), end=' ')
    l = []
    for k in range(number_of_features*2):
        if tm.get_ta_action(clause, k) == 1:
            if k < number_of_features:
                l.append(" "+get_name_of_feature_from_4_rounds(k))
            else:
                l.append("¬"+get_name_of_feature_from_4_rounds(k-number_of_features))
    return " ∧ ".join(l)




if __name__ == "__main__":
    with open(tsetlin_filename, "rb") as fp:
            games = pickle.load(fp)

    where_can = []
    for i, game in enumerate(games):
        if game[0][self_can_bet_index] == 1:
            where_can.append(i)
    indexes = np.array(where_can)

    games_can_bid = games[indexes]

    def remove_duplicate_pairs(games):
        new_games = []
        for i,game in enumerate(games):
            int_arr = game[0].astype(int)
            complete_arr = np.append(int_arr, game[1])
            new_games.append(complete_arr)

        new_games = np.array(new_games)
        
        x = np.random.rand(new_games.shape[1])
        y = new_games.dot(x)
        unique, index = np.unique(y, return_index=True)
        unique_xy = new_games[index]
        
        unique_pairs = []
        for game in unique_xy:
            last, rest = game[-1], game[:-1]
            unique_pairs.append(np.array((rest.astype(np.float32), last), dtype=object))
        return np.array(unique_pairs)
    unique_pairs = remove_duplicate_pairs(games_can_bid)

    ysu = Counter()
    for game in unique_pairs:
        ysu[game[1]] += 1
        
    need_of_oversample = [tup for tup in ysu.most_common() if tup[1] < 2000]

    def oversample(info_tup, desired_amount, source_games):
        bob = []
        for game in source_games:
            if game[1] == info_tup[0]:
                bob.append(game)
        indexes = np.random.randint(0, info_tup[1], desired_amount - info_tup[1])
        temp = []
        for ind in indexes:
            temp.append(bob[ind])
        return np.array(temp)

    unique_oversample = np.copy(unique_pairs)
    temp = []
    for overnd in need_of_oversample:
        temp.extend(oversample(overnd, 2000, unique_pairs))
    unique_oversample = np.concatenate((unique_oversample, np.array(temp)), axis=0)

    uo = Counter()
    for game in unique_oversample:
        uo[game[1]] += 1

    x =np.array([game[0] for game in unique_oversample[:, :1]])
    y =np.array([game[0] for game in unique_oversample[:, 1:]])
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.15)


    cuda_e1_tm = TMCoalescedClassifier(n_clauses_per_class, treshold, s_param,  platform='CUDA', boost_true_positive_feedback=0)
    cuda_e2_tm = TMCoalescedClassifier(n_clauses_per_class, treshold, s_param,  platform='CUDA', boost_true_positive_feedback=0)
    cuda_e3_tm = TMCoalescedClassifier(n_clauses_per_class, treshold, s_param,  platform='CUDA', boost_true_positive_feedback=0)

    epochs = 2
    for epoch in range(epochs):
        cuda_e1_tm.fit(x_train, y_train)
        cuda_e2_tm.fit(x_train, y_train)
        cuda_e3_tm.fit(x_train, y_train)
        
        

        print("Epoch {} Accuracy tm1:".format(epoch), 100*(cuda_e1_tm.predict(x_test) == y_test).mean())
        print("Epoch {} Accuracy tm2:".format(epoch), 100*(cuda_e2_tm.predict(x_test) == y_test).mean())    
        print("Epoch {} Accuracy tm3:".format(epoch), 100*(cuda_e3_tm.predict(x_test) == y_test).mean())  
    
    c_out1 = cuda_e1_tm.transform(x_test[0].reshape(1, -1))
    clauses_out1 = np.where(c_out1 == 1)[1].astype(int)
    c_out2 = cuda_e2_tm.transform(x_test[0].reshape(1, -1))
    clauses_out2 = np.where(c_out2 == 1)[1].astype(int)
    c_out3 = cuda_e3_tm.transform(x_test[0].reshape(1, -1))
    clauses_out3 = np.where(c_out3 == 1)[1].astype(int)
    text_output = [show_coalesced_clause_4r(cuda_e1_tm, clause) for clause in clauses_out1]
    freqItemSet, rules = apriori(text_output, minSup=0.5, minConf=0.5)
    print(rules)
    with open("rulesheet", "wb") as fp:
        pickle.dump(rules, fp)
    
