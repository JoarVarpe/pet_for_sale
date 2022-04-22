from argparse import Action
import functools
from turtle import pos
from gym import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from pettingzoo.classic import tictactoe_v3
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random


def env(rounds=3, static_hands=False, static_draw=False):
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env(rounds=rounds, static_hands=static_hands,
                  static_draw=static_draw)
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(rounds=3, static_hands=False, static_draw=False):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = for_sale_r2(rounds=rounds, static_hands=static_hands,
                      static_draw=static_draw)
    env = parallel_to_aec(env)
    return env


class for_sale_r2(ParallelEnv):
    metadata = {'render_modes': ['human', "ansi"], "name": "fs_v2"}

    def __init__(self, static_hands=False, static_draw=False, rounds=3):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''

        self.total_rounds = rounds
        self.static_hands = static_hands
        self.static_draw = static_draw

        properties = np.arange(1, 31).reshape(-1, 1)
        self.encode_p = OneHotEncoder(sparse=False)
        self.encode_p.fit_transform(properties)

        self.possible_agents = ["player_" + str(r) for r in range(3)]
        self.agents = self.possible_agents[:]
        self.max_score = 108

        self.action_spaces = {i: spaces.Discrete(30) for i in self.agents}
        self.observation_spaces = {i: spaces.Dict({
            'observation': spaces.Box(low=0, high=1, shape=(474,), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(30,), dtype=np.float32)
        }) for i in self.agents}

    # this cache ensures that same space object is returned for the same agent
    # allows action space seeding to work as expected

    @ functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        # check this out lol
        return self.observation_spaces[agent]

    @ functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def get_agent_ids(self):
        return set(self.possible_agents)
    # Helper functions to help with observation
    # encodes properties in hand to array, in second auction there are 30 possible cards at max

    def _onehot_encode_hand(self, hand):
        if len(hand) < 1:
            return np.zeros(30, dtype=np.float32)
        return sum(self.encode_p.transform(np.array(hand).reshape(-1, 1)))

    def _onehot_encode_board_or_stack(self, board):
        l1 = np.zeros(15, dtype=np.float32)
        l2 = np.zeros(15, dtype=np.float32)

        for number in board:
            if number == 0:
                if l1[number] == 1:
                    l2[number] = 1
                else:
                    l1[number] = 1
            else:
                if l1[number - 1] == 1:
                    l2[number - 1] = 1
                else:
                    l1[number - 1] = 1
        output = [l1, l2]
        return np.array(output)

    def _cumulative_encoding(self, value, max_val):
        # ones = [1] * value.item()
        ones = [1] * value
        zeros = [0] * (max_val - value)
        return np.array(ones + zeros, dtype=np.float32)

    # Functions to help with hands, boards and draw
    # 31 to get range to work properly
    def _get_hands(self, rounds=3, max_card=31):
        num_players = len(self.possible_agents)
        hands = [[] for i in range(num_players)]
        max_card += 1
        if max_card > 31:
            max_card = 31
        cards = list(range(1, max_card))
        random.shuffle(cards)
        for i, card in enumerate(cards[:num_players*rounds]):
            hands[i % num_players].append(card)
        return hands

    def _get_boards_and_stack(self, rounds):
        # insert 6 random cards and the cards decided for board in stack
        num_players = len(self.possible_agents)
        list_of_currencies = [i for i in range(16) if i != 1] * 2
        if rounds == 8:
            stack = random.sample(list_of_currencies, rounds * num_players)
        else:
            stack = random.sample(list_of_currencies, rounds * num_players + 6)
        boards = iter([sorted(stack[r*3:r*3 + 3], reverse=True)
                       for r in range(rounds)])
        return boards, stack

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        if mode == "human":
            print(self.current_board)
            print(self.stack)
            for agent, hand in self.agents_and_hands.items():
                print(hand, self.scores[agent])

    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def reset(self):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        '''
        if self.static_hands:
            hands = [[1, 2, 6], [3, 7, 8], [4, 5, 9]]
        else:
            # make it so that the maximum property card value is  of cards can vary
            hands = self._get_hands(self.total_rounds, max_card=len(
                self.possible_agents)*self.total_rounds+6)

        if self.static_draw:
            self.boards = iter([sorted([0, 3, 5], reverse=True), sorted(
                [4, 6, 15], reverse=True), sorted([14, 3, 7], reverse=True)])
            self.stack = [0, 2, 2, 4, 5, 6, 7, 8, 3, 5, 7, 15, 15, 14, 3]

        else:
            self.boards, self.stack = self._get_boards_and_stack(
                self.total_rounds)

        self.agents_and_hands = {agent: hand for agent,
                                 hand in zip(self.possible_agents, hands)}

        self.agents = self.possible_agents[:]
        self.num_rounds = 0
        observations = {}

        board = next(self.boards)
        self.current_board = board
        self.scores = {agent: 0 for agent in self.possible_agents}

        for agent in self.agents:
            # current hands from agents perspective
            observation = []
            other_hands = []
            other_scores = []
            for a, h in self.agents_and_hands.items():
                if a != agent:
                    other_hands.append(self._onehot_encode_hand(h))
                    other_scores.append(self._cumulative_encoding(
                        self.scores[a], self.max_score))
                else:
                    other_hands.insert(0, self._onehot_encode_hand(h))
                    other_scores.insert(0, self._cumulative_encoding(
                        self.scores[a], self.max_score))

                    # what cards the agents has is encoded as onehot
                    action_mask = np.zeros(30, dtype=np.float32)
                    for card in h:
                        action_mask[card - 1] = 1

            observation.extend(np.array(other_hands))
            observation.extend(self._onehot_encode_board_or_stack(board))
            observation.extend(self._onehot_encode_board_or_stack(self.stack))
            observation.extend(np.array(other_scores, dtype=object))
            observation = np.hstack(observation)

            observations[agent] = {"observation": observation.astype(
                np.float32), "action_mask": action_mask}

        return observations

    def step(self, actions):
        '''
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        '''

        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        # actions are a dict of {"player_x": action_x}, remove chosen cards from hands and stacks
        for (agent, action), board_card in zip(sorted(actions.items(), key=lambda x: x[1], reverse=True), self.current_board):
            try:
                self.agents_and_hands[agent].remove(action + 1)
                # print("removing {} from {}".format(board_card, self.stack))
                self.stack.remove(board_card)
                # print("{} received {} currency by playing {}".format(agent, board_card, action))
                self.scores[agent] += board_card
            except ValueError:
                # print("Action: {}, agent: {}, cards: {}".format(action+1, agent, self.agents_and_hands[agent]))
                self.agents_and_hands[agent].pop()
                self.stack.remove(board_card)
                self.scores[agent] += board_card

        self.num_rounds += 1

        # decide if game is over
        env_done = self.num_rounds >= self.total_rounds
        empties = 0
        for k, v in self.agents_and_hands.items():
            if len(v) == 0:
                empties += 1
        if empties == 3:
            env_done = True

        try:
            board = next(self.boards)
        except StopIteration:
            board = []
            env_done = True
        self.current_board = board
        dones = {agent: env_done for agent in self.agents}

        # rewards for all agents are placed in the rewards dictionary to be returned
        # print("env_done: ", env_done)
        if env_done:
            biggest = [a for a, s in self.scores.items() if s == max(
                self.scores.items(), key=lambda x: x[1])[1]]
            rewards = {agent: 1 if agent in
                       biggest else -1 for agent in self.agents}
            # print(biggest, self.scores, max(self.scores.items(), key=lambda x: x[1]))
            # if len(biggest) > 1:
            #     rewards = {agent: 0 if agent in biggest else -
            #                1 for agent in self.agents}
            # else:
            #     rewards = {agent: 1 if agent ==
            #                biggest[0] else -1 for agent in self.agents}
        else:
            rewards = {agent: 0 for agent in self.agents}

        # getting observations after actions have been taken
        observations = {}

        stack = self.stack
        for agent in self.agents:
            # current hands from agents perspective (agent is the first hand, and the first score)
            observation = []
            other_hands = []
            other_scores = []
            for a, h in self.agents_and_hands.items():
                if a != agent:
                    other_hands.append(self._onehot_encode_hand(h))
                    other_scores.append(self._cumulative_encoding(
                        self.scores[a], self.max_score))
                else:
                    other_hands.insert(0, self._onehot_encode_hand(h))
                    other_scores.insert(0, self._cumulative_encoding(
                        self.scores[a], self.max_score))

                    # what cards the agents has is encoded as onehot
                    action_mask = np.zeros(30, dtype=np.float32)
                    for card in h:
                        action_mask[card - 1] = 1

            observation.extend(np.array(other_hands))
            observation.extend(self._onehot_encode_board_or_stack(board))
            observation.extend(self._onehot_encode_board_or_stack(stack))
            observation.extend(np.array(other_scores, dtype=object))
            observation = np.hstack(observation)

            observations[agent] = {"observation": observation.astype(
                np.float32), "action_mask": action_mask}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos


if __name__ == "__main__":
    bob = raw_env()
    bob.reset()
    # steps = iter([2, 3, 5, 1, 7, 8, 4, 6, 9])
    # for agent in bob.agent_iter():
    #     observation, reward, done, info = bob.last()
    #     print(agent, "gets: ", reward)
    #     if not done:
    #         bob.step(next(steps))
    #     else:
    #         bob.step(None)
    bob.render()
    bob.step(2)
    bob.step(3)
    bob.render()
    bob.step(5)
    bob.render()

    bob.step(1)
    bob.step(7)
    bob.step(8)

    bob.step(4)
    bob.step(6)
    bob.step(9)

    # print(bob.last())
