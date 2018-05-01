import pygame
from pygame.locals import *
from pong import *
import sys
import atexit
import signal
import matplotlib.pyplot as plt

game = Pong()
train = True
# train = False


def sigint_handler(signum, frame):
    draw_plot()


signal.signal(signal.SIGINT, sigint_handler)
# Source for drawing
# http://trevorappleton.blogspot.com/2014/04/writing-pong-using-python-and-pygame.html

# number of frames per second
FPS = 200

# window size
WINDOWWIDTH = 500
WINDOWHEIGHT = 500
# line information
LINETHICKNESS = 10
PADDLESIZE = 100
# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)


def drawArena():
    DISPLAYSURF.fill(WHITE)


def drawBall(ball, game):
    ball.x = game.ball.x * 500 - LINETHICKNESS / 2
    ball.y = game.ball.y * 500 - LINETHICKNESS / 2
    pygame.draw.rect(DISPLAYSURF, RED, ball)


def drawPaddle(paddle, game):
    paddle.y = game.paddle.y * 500 - LINETHICKNESS / 2
    pygame.draw.rect(DISPLAYSURF, BLACK, paddle)


def drawWall(wall):
    pygame.draw.rect(DISPLAYSURF, BLACK, wall)

def draw_plot():

    line_x = [0, ROUND]
    line_y = [9, 9]
    f, ax = plt.subplots()
    ax.plot(game.x, game.y, 'b', label='Average Bounce')
    ax.plot(line_x, line_y, 'r', label='9')
    plt.ylabel('average rebounce')
    plt.xlabel('num of games')
    plt.title(
        'Alpha = ' + str(C) + '/(' + str(C) + '+N(s,a)), Gamma = ' + str(GAMMA) + ', Epsilon = ' + str(EPSILON))
    plt.legend(loc='lower right')
    plt.ylim(0, 14)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    if train == True:
        while True:
            game.update()
            if game.all_finished:
                draw_plot()

    else:
        # initial pygame and surface
        pygame.init()
        global DISPLAYSURF

        # set up screen
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        pygame.display.set_caption("CS440_MP4")

        ball = pygame.Rect(game.state[0] * 600 - LINETHICKNESS / 2, game.state[1] * 600 - LINETHICKNESS / 2, LINETHICKNESS, LINETHICKNESS)
        paddle = pygame.Rect(600 - LINETHICKNESS, game.state[4] * 600 - LINETHICKNESS, LINETHICKNESS,
                             game.paddle.height * 600)
        wall = pygame.Rect(0, 0, LINETHICKNESS, WINDOWHEIGHT)

        # draw game
        drawArena()
        drawBall(ball, game)
        drawPaddle(paddle, game)
        drawWall(wall)

        while True:  # main game loop
            for event in pygame.event.get():
                # exit game
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            # update game
            game.update()
            drawArena()
            drawBall(ball, game)
            drawWall(wall)
            drawPaddle(paddle, game)

            # update the screen
            pygame.display.update()
            FPSCLOCK.tick(FPS)
