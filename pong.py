import math
from ball import *
from paddle import *
from qlearning_sarsa import *
from NN import *

ROUND = 50000
LIMIT = True
LARGER_THAN_NINE = False


class Pong(object):
    def __init__(self, agent):
        self.ball = Ball()
        self.paddle = Paddle(agent)
        self.agent = agent
        self.round_finished = False
        self.all_finished = False
        self.lose_times = 0

        self.state = (self.ball.x, self.ball.y, self.ball.velocity_x, self.ball.velocity_y, self.paddle.y)
        self.x = [0.0]
        self.y = [0.0]
        if agent == 'S':
            self.rl = RL()
        elif agent == 'NN':
            self.nn = NN(model_dir='.')
            self.all_finished = True
        self.success = 0
        self.score = 0
        self.bounce_off_paddle = False
        self.scores = []
        self.round = 0

        self.lastState = None
        self.lastAction = None

    def finish(self):
        if len(self.scores) == 1000:
            self.scores = self.scores[1:]
        self.scores.append(self.score)
        self.score = 0
        self.lose_times += 1
        self.round_finished = True
        self.round += 1
        total = 0


        if self.round % 1000 == 0:
            total = float(sum(self.scores)) / 1000
            print(self.round, total)
            self.x.append(self.round)
            self.y.append(total)

        if LIMIT:
            if self.round == ROUND:
                self.all_finished = True

        if LARGER_THAN_NINE:
            if total > 9.0:
                self.all_finished = True

    def check(self):
        if self.ball.x > self.paddle.x:
            if self.paddle.y < self.ball.y < self.paddle.y + self.paddle.height:
                self.ball.bounce_off_paddle()
                self.success += 1
                self.score += 1
                self.bounce_off_paddle = True
            else:
                self.finish()

    def update_state(self):
        if self.round_finished:
            return 12, 12, 12, 12, 12
        else:
            if self.ball.velocity_x > 0:
                x_velocity = 1
            else:
                x_velocity = -1

            if self.ball.velocity_y >= 0.02:
                y_velocity = 1
            elif self.ball.velocity_y <= 0.02:
                y_velocity = -1
            else:
                y_velocity = 0
            discrete_ball_x = min(11, int(math.floor(12 * self.ball.x)))
            discrete_ball_y = min(11, int(math.floor(12 * self.ball.y)))
            discrete_paddle = min(11, int(math.floor(12 * self.paddle.y / (1 - self.paddle.height))))
            return discrete_ball_x, discrete_ball_y, x_velocity, y_velocity, discrete_paddle

    def update(self):
        self.check()
        if self.agent == 'NN':
            action = self.nn.test(np.array([self.ball.x,
                                            self.ball.y,
                                            self.ball.velocity_x,
                                            self.ball.velocity_y,
                                            self.paddle.y]))
            self.paddle.update(action)
            self.ball.update()

            if self.round_finished:
                self.ball = Ball()
                self.paddle = Paddle('NN')
                self.round_finished = False

            if self.bounce_off_paddle:
                self.bounce_off_paddle = False

            return

        state = self.update_state()
        action = self.rl.choose_action(state)
        reward = 0.0

        if self.round_finished:
            reward = -1000.0
            if self.lastState is not None:
                # self.rl.learn_qlearning(self.lastState, self.lastAction, reward, state)
                self.rl.learn_sarsa(self.lastState, self.lastAction, reward, state, action)
            self.lastState = state
            self.lastAction = action

            # restart game
            self.ball = Ball()
            self.paddle = Paddle('S')
            # self.paddle = Paddle('S')
            self.round_finished = False
            return

        if self.bounce_off_paddle:
            self.bounce_off_paddle = False
            reward = 1000.0

        if self.lastState is not None:
            # self.rl.learn_qlearning(self.lastState, self.lastAction, reward, state)
            self.rl.learn_sarsa(self.lastState, self.lastAction, reward, state, action)

        state = self.update_state()
        action = self.rl.choose_action(state)

        self.lastState = state
        self.lastAction = action
        self.paddle.update(action)
        self.ball.update()
