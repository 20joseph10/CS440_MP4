import random


class Paddle(object):
    def __init__(self, agent):
        self.height = 0.2
        self.x = 1
        self.y = 0.5 - self.height/2
        self.agent = agent
        self.actions = [0.0, -0.04, 0.04]

    def update(self, ball_y):
        if self.agent == "Q" or self.agent == "S":
            self.y += ball_y * 0.04
        elif self.agent == "R":
            self.y += random.choice(self.actions)
        elif self.agent == "A":
            if ball_y < self.y + self.height/2:
                self.y -= 0.02
            elif ball_y > self.y + self.height/2:
                self.y += 0.02

        self.y = max(0.0, self.y)
        self.y = min(1 - self.height, self.y)


