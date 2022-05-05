from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from sklearn.ensemble import RandomForestClassifier
import numpy as np 
import tqdm
import pickle
# !export OMP_NUM_THREADS=10

def tsetlin_hyperparameter_search(n_clauses_per_class, x_t, y_t, x_v, y_v):
    """
    Search for the best hyperparameters for the tsetlin machine
    """
    best = 0.0
    treshold = int(0.8 * n_clauses_per_class)
    all_accuracy = []
    for s_param in tqdm.tqdm(np.arange(1.1, 100.0, 0.1)):
        # print("making tm")
        tm = MultiClassTsetlinMachine(n_clauses_per_class, treshold, s_param, weighted_clauses=True, boost_true_positive_feedback=0)
        # print("fitting tm")
        tm.fit(x_t, y_t, epochs=200)
        # print("getting accuracy")
        acc = 100*(tm.predict(x_v) == y_v).mean()
        if acc > best:
            print("NEW BEST ACC {} WITH s_param {}".format(acc, s_param))
            best = acc
            all_accuracy.append((acc, s_param))
            with open("hyperparams/tsetlin_hyperparam_search{}.txt".format(n_clauses_per_class), "a") as fp:
                fp.write("NEW BEST ACC {} WITH s_param {}\n".format(acc, s_param))
    with open("hyperparams/tsetlin_hyperparam_search{}.pkl".format(n_clauses_per_class), "wb") as fp:
        pickle.dump(all_accuracy, fp)


if __name__ == "__main__":
    # n_clauses_per_class = [50, 100, 150, 200]
    n_clauses_per_class = [200]
    game_amount = 100000
    with open("/home/jaoi/master22/pet_for_sale/winning_games_db/{}_tsetlined_games.pkl".format(game_amount), "rb") as fp:
        games = pickle.load(fp)
    train_i = int(games.shape[0] * 0.9)
    x_train =np.array([game[0] for game in games[:train_i, :1]])
    x_test =np.array([game[0] for game in games[train_i:, :1]])
    y_train =np.array([game[0] for game in games[:train_i, 1:]])
    y_test =np.array([game[0] for game in games[train_i:, 1:]])
    print("starting search")
    for clauses in n_clauses_per_class:
        tsetlin_hyperparameter_search(clauses, x_train, y_train, x_test, y_test)