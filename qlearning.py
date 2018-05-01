import random

ALPHA = 50
GAMMA = 0.7
EPSILON = 0.05


# https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/3_Sarsa_maze/RL_brain.py
class RL(object):
    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        self.actions = [-1, 0, 1]  # a list
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table ={}
        self.seen = {}

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def learn_q(self, state, action, reward, value):
        if (state, action) not in self.seen:
            self.seen[(state, action)] = 0
        self.seen[(state, action)] += 1
        old_value = self.q_table.get((state, action), None)
        if old_value is None:
            self.q_table[(state, action)] = reward
        else:
            # C / (C + N(s, a))
            self.q_table[(state, action)] = old_value + float(self.alpha) / float(
                self.alpha + self.seen[(state, action)]) * (value - old_value)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q = [self.get_q(state, a) for a in self.actions]
            maxQ = max(q)
            if q.count(maxQ) > 1:
                best = [i for i in range(3) if q[i] == maxQ]
                action = self.actions[random.choice(best)]
                return action
            else:
                return self.actions[q.index(maxQ)]

    def choose_action_random(self):
            return random.choice([-1.0, 0.0, 1.0])

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        super(QLearningTable, self).__init__(alpha, gamma, epsilon)

    def learn(self, state1, action1, reward, state2):
        max_q_new = max([self.get_q(state2, a) for a in self.actions])
        self.learn_q(state1, action1, reward, reward + self.gamma * max_q_new)


# on-policy
class SarsaTable(RL):

    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        super(SarsaTable, self).__init__(alpha, gamma, epsilon)

    def learn(self, state1, action1, reward, state2, action2):
        q_next = self.get_q(state2, action2)
        self.learn_q(state1, action1, reward, reward + self.gamma * q_next)

