from ball import *
from paddle import *
from qlearning import *
from sarsa import *

class Pong(object):
    def __init__(self):
        self.ball = Ball()
        self.paddle = Paddle()
        self.lose = False
        self.state = (self.ball.x, self.ball.y, self.ball.velocity_x, self.ball.velocity_y, self.paddle.y)
        self.x = [0]
        self.y = [0]

    def lose(self):
        if self.ball.x > self.paddle.x:
            self.lose = True
