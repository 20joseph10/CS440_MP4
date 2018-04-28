import random


class Ball(object):
    def __init__(self):
        self.x = 0.5  # [0,1]
        self.y = 0.5  # [0,1]
        self.velocity_x = 0.03
        self.velocity_y = 0.01

    def bounce(self):
        if self.y < 0:
            self.y = -self.y
            self.velocity_y = -self.velocity_y
        elif self.y > 1:
            self.y = 2 - self.y
            self.velocity_y = -self.velocity_y
        elif self.x < 0:
            self.x = -self.x
            self.velocity_x = -self.velocity_x

    def bounce_off_paddle(self):
            self.x = 2 * 1 - self.x
            self.velocity_x = -self.velocity_x + random.uniform(-0.015, 0.015)
            self.velocity_y = self.velocity_y + random.uniform(-0.03, 0.03)

            if self.velocity_x > 0:
                self.velocity_x = min(1.0, max(0.03, self.velocity_x))
            elif self.velocity_x < 0:
                self.velocity_x = max(-1.0, min(-0.03, self.velocity_x))

            if self.velocity_y > 0:
                self.velocity_y = min(1.0, self.velocity_y)
            elif self.velocity_y < 0:
                self.velocity_y = max(-1.0, self.velocity_y)

    def update(self):
            self.x += self.v_x
            self.y += self.v_y
            self.bounce()

