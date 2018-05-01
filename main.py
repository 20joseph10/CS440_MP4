import pygame
from pygame.locals import *
from pong import *
import sys
import atexit
import signal
import matplotlib.pyplot as plt

MODE = 'train'
# MODE = 'play'

# initialize game
game = Pong()


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
    ball.x = scaleBall(game.ball.x)
    ball.y = scaleBall(game.ball.y)
    pygame.draw.rect(DISPLAYSURF, RED, ball)


def drawPaddle(paddle, game):
    paddle.y = scaleBall(game.paddle.y)
    pygame.draw.rect(DISPLAYSURF, BLACK, paddle)


def drawWall(wall):
    pygame.draw.rect(DISPLAYSURF, BLACK, wall)


def scaleBall(unit):
    return 500 * unit - LINETHICKNESS / 2


def scalePaddle(unit):
    return 500 * unit - LINETHICKNESS


def scaleWall(unit):
    return 500 * unit

def draw_plot():

    with open('x.txt', 'w+') as xFile:
        x = [str(i) for i in game.x]
        xFile.write(' '.join(x))
    with open('y.txt', 'w+') as yFile:
        y = [str(i) for i in game.y]
        yFile.write(' '.join(y))

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
    if MODE == 'train':
        while True:
            game.update()
            if game.all_finished:
                draw_plot()

    elif MODE == 'play':
        # initial pygame and surface
        pygame.init()
        global DISPLAYSURF

        # set up screen
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        pygame.display.set_caption("CS440_MP4")

        # ball
        ball_x = scaleBall(game.state[0])
        ball_y = scaleBall(game.state[1])
        ball = pygame.Rect(ball_x, ball_y, LINETHICKNESS, LINETHICKNESS)
        # paddle
        paddle_x = scalePaddle(1.0)
        paddle_y = scalePaddle(game.state[4])
        paddle_height = scaleWall(game.paddle.height)
        paddle = pygame.Rect(paddle_x, paddle_y, LINETHICKNESS, paddle_height)
        # wall
        wall_x = scaleWall(0.0)
        wall_y = scaleWall(0.0)
        wall = pygame.Rect(wall_x, wall_y, LINETHICKNESS, WINDOWHEIGHT)

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
