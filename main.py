import pygame
from pygame.locals import *
from pong import *
import sys
import signal
import matplotlib.pyplot as plt

# http://trevorappleton.blogspot.com/2014/04/writing-pong-using-python-and-pygame.html

game = Pong('S')
# game = Pong('NN')
train = True
# train = False


def sigint_handler(signum, frame):
    draw_plot()


signal.signal(signal.SIGINT, sigint_handler)

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
#font
pygame.font.init()
FONT = pygame.font.SysFont("monospace", 16)


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

    line_x = [0, game.ROUND]
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
        # train
        while not game.all_finished:
            game.update()
        draw_plot()

        #
        # initial pygame and surface
        pygame.init()
        global DISPLAYSURF

        # set up screen
        FPSCLOCK = pygame.time.Clock()
        DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
        pygame.display.set_caption("CS440_MP4_hwu63")

        ball = pygame.Rect(game.state[0] * 600 - LINETHICKNESS / 2, game.state[1] * 600 - LINETHICKNESS / 2, LINETHICKNESS, LINETHICKNESS)
        paddle = pygame.Rect(500 - LINETHICKNESS, game.state[4] * 500 - LINETHICKNESS, LINETHICKNESS,
                             game.paddle.height * 500)
        wall = pygame.Rect(0, 0, LINETHICKNESS, WINDOWHEIGHT)

        # draw game
        DISPLAYSURF.fill(WHITE)
        drawBall(ball, game)
        drawPaddle(paddle, game)
        drawWall(wall)

        game.ROUND = 200
        game.round = 0
        game.all_finished = False
        game.x = []
        game.y = []

        while not game.all_finished:  # main game loop
            for event in pygame.event.get():
                # exit game
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            # update game
            game.update()
            DISPLAYSURF.fill(WHITE)
            drawBall(ball, game)
            drawWall(wall)
            drawPaddle(paddle, game)

            # update the screen
            roundtext = FONT.render("Round {0}".format(game.round), 1, (0, 0, 0))
            DISPLAYSURF.blit(roundtext, (10, 10))
            scoretext = FONT.render("Score {0}".format(game.score), 1, (0, 0, 0))
            DISPLAYSURF.blit(scoretext, (10, 20))
            pygame.display.update()
            FPSCLOCK.tick(FPS)

        draw_plot()
