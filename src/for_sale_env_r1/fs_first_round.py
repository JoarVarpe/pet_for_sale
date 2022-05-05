from argparse import Action
import functools
from turtle import pos
from gym import spaces
from nbformat import current_nbformat_minor
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import agent_selector
from pettingzoo.classic import uno_v4
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import random
from collections import deque
import math


def env(rounds=3):
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = raw_env(rounds=rounds)
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(rounds=3):
    '''
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    '''
    env = for_sale_r2(rounds=rounds)
    return env


class for_sale_r2(AECEnv):
    metadata = {'render_modes': ['human', "ansi"], "name": "fs_r1"}

    def __init__(self, static_draw=False, rounds=3):
        '''
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        self.starting_coins = 18
        self.total_rounds = rounds
        self.static_draw = static_draw

        properties = np.arange(1, 31).reshape(-1, 1)
        self.encode_p = OneHotEncoder(sparse=False)
        self.encode_p.fit_transform(properties)

        self.possible_agents = ["player_" + str(r) for r in range(3)]
        self.agent_encoding = {agent: i for i,
                               agent in enumerate(self.possible_agents)}
        self.agents = deque(self.possible_agents[:])
        self.observations = {agent: [] for agent in self.agents}

        # one action for each possible monetary bid, in addition to passing which is 0
        self.action_spaces = {i: spaces.Discrete(19) for i in self.agents}
        self.observation_spaces = {i: spaces.Dict({
            'observation': spaces.Box(low=0, high=30, shape=(42,), dtype=np.float32),
            'action_mask': spaces.Box(low=0, high=1, shape=(19,), dtype=np.float32)
        }) for i in self.agents}
        self.agent_cards = {agent: [] for agent in self.possible_agents}
        self.agent_coins = {
            agent: self.starting_coins for agent in self.possible_agents}

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
        l1 = np.zeros(30, "float32")

        for number in board:
            if number == 0:
                l1[number] = 1
            else:
                l1[number - 1] = 1
        return l1

    def _onehot_encode_coins(self, coins, max_coins=18):
        ones = [1] * (coins + 1)
        zeros = [0] * (max_coins - (coins + 1))
        return np.array(ones + zeros, "float32")

    def _onehot_encode_bid(self, bid, max_bid=19):
        output = np.zeros(max_bid)
        output[bid] = 1
        return output

    def _onhot_encode_agent(self, agent):
        return np.array([1 if a == agent else 0 for a in self.possible_agents])

    # Functions to help with hands, boards and draw
    # 31 to get range to work properly

    def _get_boards_and_stack(self, rounds, max_card=30):
        # insert 6 random cards and the cards decided for board in stack
        num_players = len(self.possible_agents)
        amount_to_add = int(num_players * rounds * 0.25)
        if max_card > 30:
            max_card = 30
        list_of_currencies = [i for i in range(1, max_card + 1)]
        if rounds == 8:
            stack = random.sample(list_of_currencies, rounds * num_players)
        else:
            stack = random.sample(list_of_currencies, rounds * num_players + amount_to_add)
        boards = iter([sorted(stack[r*3:r*3 + 3], reverse=True)
                       for r in range(rounds)])
        return boards, sorted(stack, reverse=True)

    def _pay_for_bid(self, agent, bid):
        # if the agent won the auction, they pay full price
        if self.current_bid["agent"] == agent and self.current_bid["value"] == bid:
            self.agent_coins[agent] -= bid
        else:
            # if the agent lost the auction, they pay half price
            self.agent_coins[agent] -= math.ceil(bid / 2)

    def render(self, mode="human"):
        '''
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        '''
        if mode == "human":
            the_a = self.agent_selection
            print()
            print("highest bid: ", self.current_bid)
            print("current bids by each player".format(self.agent_bids))
            print("made by player: ", self._agent_selector._current_agent)
            print("current agents in game: ", self.agents)
            print("The agent to move: ", the_a)
            print("board: ", self.current_board)
            print("stack: ", self.stack)
            print("coins: {}".format(self.agent_coins))
            print("cards: {}".format(self.agent_cards))
            print()



    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass

    def observe(self, agent):
        '''
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        '''

        return self.observations[agent]

    def reset(self):
        '''
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.

        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.

        Returns the observations for each agent
        '''
        self.agent_bids = {agent: 0 for agent in self.possible_agents}
        amount_to_add = int(len(self.possible_agents) * self.total_rounds * 0.25)
        self.boards, self.stack = self._get_boards_and_stack(
            self.total_rounds, self.total_rounds * len(self.possible_agents) + amount_to_add)

        self.agent_cards = {agent: [] for agent in self.possible_agents}
        self.agent_coins = {
            agent: self.starting_coins for agent in self.possible_agents}

        self.agents = deque(self.possible_agents[:])
        self.current_bid = {"agent": self.agents[-1], "value": 0}
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: [] for agent in self.agents}
        self.num_rounds = 0

        board = next(self.boards)
        self.current_board = board
        # self.scores = {agent: 0 for agent in self.possible_agents}

        for i, agent in enumerate(self.agents):
            observation = []
            # current hands from agents perspective
            own_coins = self.agent_coins[agent]
            own_hand = np.zeros(self.total_rounds)
            can_bid = self.agent_coins[agent] > self.current_bid["value"]
            current_own_bid = self.agent_bids[agent]
            coins_on_fold = own_coins - math.ceil(current_own_bid / 2)

            next_coins = self.agent_coins[self.agents[(i + 1) % 3]]
            next_hand = np.zeros(self.total_rounds)
            next_can_bid = self.agent_coins[self.agents[(
                i + 1) % 3]] > self.current_bid["value"]
            next_own_bid = self.agent_bids[self.agents[(i + 1) % 3]]
            next_coins_on_fold = next_coins - math.ceil(next_own_bid / 2)

            final_coins = self.agent_coins[self.agents[(i + 2) % 3]]
            final_hand = np.zeros(self.total_rounds)
            final_can_bid = self.agent_coins[self.agents[(
                i + 2) % 3]] > self.current_bid["value"]
            final_own_bid = self.agent_bids[self.agents[(i + 2) % 3]]
            final_coins_on_fold = final_coins - final_own_bid

            # every stat is encoded into onehot
            action_mask = np.zeros(self.starting_coins + 1, "float32")
            # can always bet pass, other possible actions are bets
            for bet in range(self.agent_coins[agent]+1):
                action_mask[bet] = 1

            observation.append(own_coins)
            observation.extend(sorted(own_hand, reverse=True))
            observation.append(can_bid)
            observation.append(current_own_bid)
            observation.append(coins_on_fold)
            observation.append(min(board))
            observation.append(3)

            observation.append(next_coins)
            observation.extend(sorted(next_hand, reverse=True))
            observation.append(next_can_bid)
            observation.append(next_own_bid)
            observation.append(next_coins_on_fold)
            observation.append(board[1])
            observation.append(2)

            observation.append(final_coins)
            observation.extend(sorted(final_hand, reverse=True))
            observation.append(final_can_bid)
            observation.append(final_own_bid)
            observation.append(final_coins_on_fold)
            observation.append(max(board))
            observation.append(1)

            observation.extend(self.stack)
            observation.extend(board)
            observation.append(np.std(board))

            observation = np.hstack(observation)

            self.observations[agent] = {"observation": observation.astype(
                np.float32), "action_mask": action_mask}

        self._agent_selector = agent_selector(list(self.agents))
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        '''
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        '''

        agent = self.agent_selection


        if self.dones[agent]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)





        passed = False
        # if player passes, then remove them from the list of agents and give the agent the lowest card
        if action == 0:
            card_to_add = self.current_board.pop()
            self.agent_cards[agent].append(card_to_add)
            self.stack[self.stack.index(card_to_add)] = 0
            self.stack.sort(reverse=True)
            self._pay_for_bid(agent, self.agent_bids[agent])
            self.agents.popleft()
            self._agent_selector = agent_selector(list(self.agents))
            passed = True
            # if remaining agent is the last one, then they get the last card and will start next auction
            if len(self.agents) == 1:
                card_to_add = self.current_board.pop()
                self.agent_cards[self.agents[0]].append(card_to_add)
                self.stack[self.stack.index(card_to_add)] = 0
                self.stack.sort(reverse=True)
                self._pay_for_bid(self.agents[0], self.current_bid["value"])
                self.agents = deque(self.possible_agents[:])
                while self.agents[0] != self.current_bid["agent"]:
                    self.agents.rotate(-1)
                self._agent_selector = agent_selector(list(self.agents))

        elif action > self.current_bid["value"] and self.agent_coins[agent] >= action:
            self.current_bid["value"] = action
            self.current_bid["agent"] = agent
            self.agent_bids[agent] = action
        else:
            raise ValueError("Invalid action {} by {}. Status is bids are {}, coins are {}, highest bid {}".format(action, agent, self.agent_bids, self.agent_coins, self.current_bid))
       
        # game is over if there are no more boards only after all cards
        # have been taken from the board
        board = self.current_board
        env_done = False
        if len(board) == 0:
            try:
                board = next(self.boards)
                self.agents = deque(self.possible_agents[:])
                while self.agents[0] != self.current_bid["agent"]:
                    self.agents.rotate(-1)
                self.current_bid = {"agent": self.agents[-1], "value": 0}
                self.agent_bids = {agent: 0 for agent in self.possible_agents}
            except StopIteration:
                board = []
                env_done = True
                self.dones = {agent: env_done for agent in self.possible_agents}
                self.agents = deque(self.possible_agents[:])
                while self.agents[0] != self.current_bid["agent"]:
                    self.agents.rotate(-1)
                self._agent_selector = agent_selector(list(self.agents))
                

            self.current_board = board

        # rewards for all agents are placed in the rewards dictionary to be returned
        # print("env_done: ", env_done)
        if env_done:
            agents_and_scores = {agent: self.agent_coins[agent] + sum(
                self.agent_cards[agent]) for agent in self.possible_agents}
            for agent, score in sorted(agents_and_scores.items(), key=lambda x: x[1], reverse=True):
                if score == max(agents_and_scores.items(), key=lambda x: x[1])[1]:
                    self._cumulative_rewards[agent] = 1
                elif score == min(agents_and_scores.items(), key=lambda x: x[1])[1]:
                    self._cumulative_rewards[agent] = -1
                else:
                    self._cumulative_rewards[agent] = 0
        else:
                
            # getting observations after actions have been taken

            odd_one_out_list = [
                x for x in self.possible_agents if x not in self.agents]
            if len(odd_one_out_list):
                try:
                    for i, a in enumerate(self.agents):
                        observation = []
                        own_coins = self.agent_coins[a]
                        own_hand = self.agent_cards[a] + \
                            (self.total_rounds - len(self.agent_cards[a])) * [0]
                        can_bid = self.agent_coins[a] > self.current_bid["value"]
                        current_own_bid = self.agent_bids[a]
                        coins_on_fold = own_coins - math.ceil(current_own_bid / 2)

                        next_coins = self.agent_coins[self.agents[(i + 1) % 2]]
                        next_hand = self.agent_cards[self.agents[(i + 1) % 2]] + (
                            self.total_rounds - len(self.agent_cards[self.agents[(i + 1) % 2]])) * [0]
                        next_can_bid = self.agent_coins[self.agents[(
                            i + 1) % 2]] > self.current_bid["value"]
                        next_own_bid = self.agent_bids[self.agents[(i + 1) % 2]]
                        next_coins_on_fold = next_coins - math.ceil(next_own_bid / 2)

                        out_a = odd_one_out_list[0]
                        odd_coins = self.agent_coins[out_a]
                        odd_hand = self.agent_cards[out_a] + \
                            (self.total_rounds - len(self.agent_cards[out_a])) * [0]
                        odd_bid = False
                        current_odd_bid = self.agent_bids[out_a]
                        odd_coins_on_fold = odd_coins

                        action_mask = np.zeros(self.starting_coins + 1, "float32")
                        # can always bet pass, other possible actions are bets
                        action_mask[0] = 1
                        for bet in range(0, self.agent_coins[a]):
                            if bet+1 > self.current_bid["value"]:
                                action_mask[bet+1] = 1

                        observation.append(own_coins)
                        observation.extend(own_hand)
                        observation.append(can_bid)
                        observation.append(current_own_bid)
                        observation.append(coins_on_fold)
                        observation.append(min(board))
                        observation.append(2)

                        observation.append(next_coins)
                        observation.extend(next_hand)
                        observation.append(next_can_bid)
                        observation.append(next_own_bid)
                        observation.append(next_coins_on_fold)
                        observation.append(max(board))
                        observation.append(1)

                        observation.append(odd_coins)
                        observation.extend(odd_hand)
                        observation.append(odd_bid)
                        observation.append(current_odd_bid)
                        observation.append(odd_coins_on_fold)
                        observation.append(0)
                        observation.append(3)

                        observation.extend(self.stack)
                        observation.extend(board + [0] * (3 - len(board)))
                        observation.append(np.std(board))

                        observation = np.hstack(observation)

                        self.observations[a] = {"observation": observation.astype(
                            np.float32), "action_mask": action_mask}
                except IndexError:
                    self.render()
                    raise IndexError("shit pommes frites {} {} {} {}".format(self.agents, board, self.agent_cards, env_done))

            else:
                for i, a in enumerate(self.agents):
                    observation = []
                # current hands from agents perspective
                    own_coins = self.agent_coins[a]
                    own_hand = self.agent_cards[a] + \
                        (self.total_rounds - len(self.agent_cards[a])) * [0]
                    can_bid = self.agent_coins[a] > self.current_bid["value"]
                    current_own_bid = self.agent_bids[a]
                    coins_on_fold = own_coins - math.ceil(current_own_bid / 2)

                    next_coins = self.agent_coins[self.agents[(i + 1) % 3]]
                    next_hand = self.agent_cards[self.agents[(i + 1) % 3]] + (
                        self.total_rounds - len(self.agent_cards[self.agents[(i + 1) % 3]])) * [0]
                    next_can_bid = self.agent_coins[self.agents[(
                        i + 1) % 3]] > self.current_bid["value"]
                    next_own_bid = self.agent_bids[self.agents[(i + 1) % 3]]
                    next_coins_on_fold = next_coins - math.ceil(next_own_bid / 2)

                    final_coins = self.agent_coins[self.agents[(i + 2) % 3]]
                    final_hand = self.agent_cards[self.agents[(i + 2) % 3]] + (
                        self.total_rounds - len(self.agent_cards[self.agents[(i + 2) % 3]])) * [0]
                    final_can_bid = self.agent_coins[self.agents[(
                        i + 2) % 3]] > self.current_bid["value"]
                    final_own_bid = self.agent_bids[self.agents[(i + 2) % 3]]
                    final_coins_on_fold = final_coins - final_own_bid

                    # every stat is encoded into onehot
                    action_mask = np.zeros(self.starting_coins + 1, "float32")

                    # can always bet pass, other possible actions are bets
                    action_mask[0] = 1
                    for bet in range(0, self.agent_coins[a]):
                        if bet+1 > self.current_bid["value"]:
                            action_mask[bet+1] = 1

                    observation.append(own_coins)
                    observation.extend(sorted(own_hand, reverse=True))
                    observation.append(can_bid)
                    observation.append(current_own_bid)
                    observation.append(coins_on_fold)
                    observation.append(min(board))
                    observation.append(3)

                    observation.append(next_coins)
                    observation.extend(sorted(next_hand, reverse=True))
                    observation.append(next_can_bid)
                    observation.append(next_own_bid)
                    observation.append(next_coins_on_fold)
                    observation.append(board[1])
                    observation.append(2)

                    observation.append(final_coins)
                    observation.extend(sorted(final_hand, reverse=True))
                    observation.append(final_can_bid)
                    observation.append(final_own_bid)
                    observation.append(final_coins_on_fold)
                    observation.append(max(board))
                    observation.append(1)

                    observation.extend(self.stack)
                    observation.extend(board)
                    observation.append(np.std(board))

                    observation = np.hstack(observation)

                    self.observations[a] = {"observation": observation.astype(
                        np.float32), "action_mask": action_mask}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.possible_agents}
        # selects the next agent.
        if passed:
            self.agent_selection = self._agent_selector.next()

        else:
            self.agent_selection = self._agent_selector.next()
            self.agents.rotate(-1)



if __name__ == "__main__":
    env = env()
    env.reset()

    for i, agent in enumerate(env.agent_iter()):
        print("*"*20, str(i), "*"*20)
        obs, rew, done, _ = env.last()
        print(obs["observation"].shape)
        if done:
            print(rew)
            print(env._cumulative_rewards)
            env.step(None)
            env.render()
        else:
            print("NOT DONE REW", rew)
            if agent == "player_1":
                env.step(1)
            else:
                env.step(0)
            env.render()
    # other_hands = []
    # other_coins = []
    # for a, h in self.agent_cards.items():
    #     if a != agent:
    #         other_hands.append(self._onehot_encode_hand(h))
    #         other_coins.append(self._onehot_encode_coins(
    #             self.agent_coins[a], self.starting_coins))
    #     else:
    #         other_hands.insert(0, self._onehot_encode_hand(h))
    #         other_coins.insert(0, self._onehot_encode_coins(
    #             self.agent_coins[a], self.starting_coins))

    #         # what cards the agents has is encoded as onehot
    #         action_mask = np.zeros(self.starting_coins + 1, "float32")
    #         # can always bet pass, other possible actions are bets
    #         for bet in range(len(self.agent_coins)+1):
    #             action_mask[bet] = 1

    # observation.extend(np.array(other_coins))
    # observation.extend(self._onehot_encode_board_or_stack(board))
    # observation.extend(np.array(other_hands))
    # observation.extend(self._onehot_encode_bid(self.current_bid["value"], self.starting_coins+1))
    # observation.extend(self._onhot_encode_agent(self.current_bid["agent"]))
    # observation = np.hstack(observation)
