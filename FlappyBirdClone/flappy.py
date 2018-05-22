from itertools import cycle
import random
import sys
import os

import pygame
from pygame.locals import *
from ple.games.base import PyGameWrapper


FPS = 30
SCREENWIDTH  = 288
SCREENHEIGHT = 512
# amount by which base can maximum shift to left
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe
BASEY        = SCREENHEIGHT * 0.79
# image, sound and hitmask  dicts

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    # blue bird
    (
        # amount by which base can maximum shift to left
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    # yellow bird
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
    'assets/sprites/background-black.png',
)

# list of pipes
PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)



class FlappyClone(PyGameWrapper):
    def __init__(self, black=False, crazy=False):
        # self.FPSCLOCK = pygame.time.Clock()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.IMAGES, self.HITMASKS = {}, {}
        self.black = black
        self.crazy = crazy


        actions = {"up": K_UP}
        super(FlappyClone, self).__init__(SCREENWIDTH, SCREENHEIGHT, actions=actions)

        self.SCREEN = pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.display.set_caption('Flappy Bird Clone')

        # numbers sprites for score display
        self.IMAGES['numbers'] = (
            pygame.image.load(os.path.join(dir_path,'assets/sprites/0.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/1.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/2.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/3.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/4.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/5.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/6.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/7.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/8.png')).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,'assets/sprites/9.png')).convert_alpha()
        )

        # base (ground) sprite
        self.IMAGES['base'] = pygame.image.load(os.path.join(dir_path,'assets/sprites/base.png')).convert_alpha()


        # select random background sprites
        if self.black:
            randBg = len(BACKGROUNDS_LIST) - 1
        else:
            randBg = random.randint(0, len(BACKGROUNDS_LIST) - 2)
        self.IMAGES['background'] = pygame.image.load(os.path.join(dir_path,BACKGROUNDS_LIST[randBg])).convert()

        # select random player sprites
        if self.black:
            randPlayer = 0
        else:
            randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
        self.IMAGES['player'] = (
            pygame.image.load(os.path.join(dir_path,PLAYERS_LIST[randPlayer][0])).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,PLAYERS_LIST[randPlayer][1])).convert_alpha(),
            pygame.image.load(os.path.join(dir_path,PLAYERS_LIST[randPlayer][2])).convert_alpha(),
        )

        # select random pipe sprites
        if self.black:
            pipeindex = 0
        else:
            pipeindex = random.randint(0, len(PIPES_LIST) - 1)
        self.IMAGES['pipe'] = (
            pygame.transform.rotate(
                pygame.image.load(os.path.join(dir_path,PIPES_LIST[pipeindex])).convert_alpha(), 180),
            pygame.image.load(os.path.join(dir_path,PIPES_LIST[pipeindex])).convert_alpha(),
        )

        # hismask for pipes
        self.HITMASKS['pipe'] = (
            self.getHitmask(self.IMAGES['pipe'][0]),
            self.getHitmask(self.IMAGES['pipe'][1]),
        )

        # hitmask for player
        self.HITMASKS['player'] = (
            self.getHitmask(self.IMAGES['player'][0]),
            self.getHitmask(self.IMAGES['player'][1]),
            self.getHitmask(self.IMAGES['player'][2]),
        )

        self.pipeVelX = -4

        # player velocity, max velocity, downward accleration, accleration on flap
        self.playerMaxVelY =  10   # max vel along Y, max descend speed
        # self.playerMinVelY =  -8   # min vel along Y, max ascend speed
        self.playerAccY    =   1   # players downward accleration
        # self.playerRot     =  45   # player's rotation
        self.playerVelRot  =   3   # angular speed
        self.playerRotThr  =  20   # rotation threshold
        self.playerFlapAcc =  -9   # players speed on flapping
        self.baseShift = self.IMAGES['base'].get_width() - self.IMAGES['background'].get_width()
        self.totalTicks = 0


    def init(self):
        """ Reset Game. """
        # movementInfo = showWelcomeAnimation()
        self.playery = int((SCREENHEIGHT - self.IMAGES['player'][0].get_height()) / 2)
        self.playerx = int(SCREENWIDTH * 0.2)
        self.basex = 0
        self.playerIndexGen = cycle([0, 1, 2, 1])
        self.score = self.playerIndex = self.loopIter = 0
        self.playerRot     =  45   # player's rotation
        self.playerVelY    =  -9   # player's velocity along Y, default same as playerFlapped
        self.playerFlapped = False # True when player flaps
        self.lives = 1
        self.ticks = 0

        # get 2 new pipes to add to upperPipes lowerPipes list
        newPipe1 = self.getRandomPipe()
        newPipe2 = self.getRandomPipe()
        # list of upper pipes
        self.upperPipes = [
            {'x': SCREENWIDTH + 20, 'y': newPipe1[0]['y']},
            {'x': SCREENWIDTH + 20 + (SCREENWIDTH / 2), 'y': newPipe2[0]['y']}]
        # list of lowerpipe
        self.lowerPipes = [
            {'x': SCREENWIDTH + 20, 'y': newPipe1[1]['y']},
            {'x': SCREENWIDTH + 20 + (SCREENWIDTH / 2), 'y': newPipe2[1]['y']}]

    def game_over(self):
        return self.lives <= 0

    def step(self, dt=1/30.):
        self.score += self.rewards["tick"]
        self.ticks += 1
        self.totalTicks += 1
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                if self.playery > -2 * self.IMAGES['player'][0].get_height():
                    self.playerVelY = self.playerFlapAcc
                    self.playerFlapped = True

        # check for crash here
        self.crashTest = self.checkCrash({'x': self.playerx,
                                            'y': self.playery,
                                            'index': self.playerIndex},
                            self.upperPipes, self.lowerPipes)
        if self.crashTest[0]:
            self.lives = 0
            # return {
                # 'y': self.playery,
                # 'groundCrash': self.crashTest[1],
                # 'basex': self.basex,
                # 'upperPipes': self.upperPipes,
                # 'lowerPipes': self.lowerPipes,
                # 'score': self.score,
                # 'playerVelY': self.playerVelY,
                # 'playerRot': self.playerRot
            # }
        if self.lives <= 0:
            self.score += self.rewards["loss"]

        # check for score
        self.playerMidPos = self.playerx + self.IMAGES['player'][0].get_width() / 2
        for pipe in self.upperPipes:
            pipeMidPos = pipe['x'] + self.IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= self.playerMidPos < pipeMidPos + 4:
                self.score += self.rewards["positive"]

        # playerIndex basex change
        if (self.loopIter + 1) % 3 == 0:
            self.playerIndex = next(self.playerIndexGen)
        self.loopIter = (self.loopIter + 1) % 30
        self.basex = -((-self.basex + 100) % self.baseShift)

        # rotate the player
        if self.playerRot > -90:
            self.playerRot -= self.playerVelRot

        # player's movement
        if self.playerVelY < self.playerMaxVelY and not self.playerFlapped:
            self.playerVelY += self.playerAccY
        if self.playerFlapped:
            self.playerFlapped = False

            # more rotation to cover the threshold (calculated in visible rotation)
            self.playerRot = 45

        self.playerHeight = self.IMAGES['player'][self.playerIndex].get_height()
        self.playery += min(self.playerVelY, BASEY - self.playery - self.playerHeight)

        # move pipes to left
        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            uPipe['x'] += self.pipeVelX
            lPipe['x'] += self.pipeVelX

        # add new pipe when first pipe is about to touch left of screen
        if 0 < self.upperPipes[0]['x'] < 5:
            newPipe = self.getRandomPipe()
            self.upperPipes.append(newPipe[0])
            self.lowerPipes.append(newPipe[1])

        # remove first pipe if its out of the screen
        if self.upperPipes[0]['x'] < -self.IMAGES['pipe'][0].get_width():
            self.upperPipes.pop(0)
            self.lowerPipes.pop(0)


        # draw sprites
        self.SCREEN.blit(self.IMAGES['background'], (0,0))

        for uPipe, lPipe in zip(self.upperPipes, self.lowerPipes):
            self.SCREEN.blit(self.IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            self.SCREEN.blit(self.IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        self.SCREEN.blit(self.IMAGES['base'], (self.basex, BASEY))
        # print score so player overlaps the score
        if not self.black:
            self.showScore(self.score)

        # Player rotation has a threshold
        self.visibleRot = self.playerRotThr
        if self.playerRot <= self.playerRotThr:
            self.visibleRot = self.playerRot

        playerSurface = pygame.transform.rotate(
            self.IMAGES['player'][self.playerIndex], self.visibleRot)
        self.SCREEN.blit(playerSurface, (self.playerx, self.playery))

        # pygame.display.update()

    def getRandomPipe(self):
        """returns a randomly generated pipe"""
        if self.crazy:
            if not hasattr(self, "pipeTmp"):
                self.pipeTmp = 1
            self.pipeTmp +=1
            if self.pipeTmp % 2:
                gapY = 0
            else:
                gapY = int(BASEY * 0.6 - PIPEGAPSIZE)
        else:
            # y of gap between upper and lower pipe
            gapY = self.rng.randint(0, int(BASEY * 0.6 - PIPEGAPSIZE))
        gapY += int(BASEY * 0.2)
        pipeHeight = self.IMAGES['pipe'][0].get_height()
        pipeX = SCREENWIDTH + 10

        return [
            {'x': pipeX, 'y': gapY - pipeHeight},  # upper pipe
            {'x': pipeX, 'y': gapY + PIPEGAPSIZE}, # lower pipe
        ]


    def showScore(self, score):
        """displays score in center of screen"""
        score = max(0, int(score))
        scoreDigits = [int(x) for x in list(str(score))]
        totalWidth = 0 # total width of all numbers to be printed

        for digit in scoreDigits:
            totalWidth += self.IMAGES['numbers'][digit].get_width()

        Xoffset = (SCREENWIDTH - totalWidth) / 2

        for digit in scoreDigits:
            self.SCREEN.blit(self.IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
            Xoffset += self.IMAGES['numbers'][digit].get_width()


    def checkCrash(self, player, upperPipes, lowerPipes):
        """returns True if player collders with base or pipes."""
        pi = player['index']
        player['w'] = self.IMAGES['player'][0].get_width()
        player['h'] = self.IMAGES['player'][0].get_height()

        # if player crashes into ground
        if player['y'] + player['h'] >= BASEY - 1:
            return [True, True]
        elif player['y'] < 0:
            return [True, True]
        else:

            playerRect = pygame.Rect(player['x'], player['y'],
                        player['w'], player['h'])
            pipeW = self.IMAGES['pipe'][0].get_width()
            pipeH = self.IMAGES['pipe'][0].get_height()

            for uPipe, lPipe in zip(upperPipes, lowerPipes):
                # upper and lower pipe rects
                uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
                lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

                # player and upper/lower pipe hitmasks
                pHitMask = self.HITMASKS['player'][pi]
                uHitmask = self.HITMASKS['pipe'][0]
                lHitmask = self.HITMASKS['pipe'][1]

                # if bird collided with upipe or lpipe
                uCollide = self.pixelCollision(playerRect, uPipeRect, pHitMask, uHitmask)
                lCollide = self.pixelCollision(playerRect, lPipeRect, pHitMask, lHitmask)

                if uCollide or lCollide:
                    return [True, False]

        return [False, False]

    def pixelCollision(self, rect1, rect2, hitmask1, hitmask2):
        """Checks if two objects collide and not just their rects"""
        rect = rect1.clip(rect2)

        if rect.width == 0 or rect.height == 0:
            return False

        x1, y1 = rect.x - rect1.x, rect.y - rect1.y
        x2, y2 = rect.x - rect2.x, rect.y - rect2.y

        for x in range(rect.width):
            for y in range(rect.height):
                if hitmask1[x1+x][y1+y] and hitmask2[x2+x][y2+y]:
                    return True
        return False

    def getHitmask(self, image):
        """returns a hitmask using an image's alpha."""
        mask = []
        for x in range(image.get_width()):
            mask.append([])
            for y in range(image.get_height()):
                mask[x].append(bool(image.get_at((x,y))[3]))
        return mask
        #Reset?

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player y position.
            * players velocity.
            * next pipe distance to player
            * next pipe top y position
            * next pipe bottom y position
            * next next pipe distance to player
            * next next pipe top y position
            * next next pipe bottom y position


            See code for structure.

        """
        self.upperPipes
        self.lowerPipes
        pipes = []
        pipeW = self.IMAGES['pipe'][0].get_width()
        for u, l in zip(self.upperPipes, self.lowerPipes):
            if u["x"] + pipeW > self.playerx:
                pipes.append(((u, l), u["x"] - self.playerx + pipeW))

        if len(pipes) == 1:
            pipeX = SCREENWIDTH + 10
            fakePipe = [
                {'x': pipeX, 'y': SCREENHEIGHT / 2 - PIPEGAPSIZE},  # upper pipe
                {'x': pipeX, 'y': SCREENHEIGHT / 2 + PIPEGAPSIZE}, # lower pipe
            ]
            pipes.append(((fakePipe[0], fakePipe[1]),
                          fakePipe[0]["x"] - self.playerx + pipeW))
        else:
            pipes.sort(key=lambda p: p[1])

        # next_pipe = pipes[0]
        # next_next_pipe = pipes[1]

        # ??????????????????????????? What is done here?
        # if next_next_pipe.x < next_pipe.x:
            # next_pipe, next_next_pipe = next_next_pipe, next_pipe

        pipeHeight = self.IMAGES['pipe'][0].get_height()
        state = {
            "player_y": self.playery,
            "player_vel": self.playerVelY,

            "next_pipe_dist_to_player": pipes[0][1],
            "next_pipe_top_y": pipes[0][0][0]["y"] + pipeHeight,
            "next_pipe_bottom_y":pipes[0][0][1]["y"],


            "next_next_pipe_dist_to_player": pipes[1][1],
            "next_next_pipe_top_y": pipes[1][0][0]["y"] + pipeHeight,
            "next_next_pipe_bottom_y": pipes[1][0][1]["y"]
        }

        return state

    def getScore(self):
        return self.score

if __name__ == '__main__':
    from ple import PLE
    pygame.init()
    game = FlappyClone(black=False)
    env = PLE(game, display_screen=True, force_fps=False, fps=30)
    env.init()

    while True:

        if env.game_over():
            print("Dead")
            env.reset_game()
        # print(game.getGameState())

        reward = env.act(None)
