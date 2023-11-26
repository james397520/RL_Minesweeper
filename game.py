import pygame
from piece import Piece 
from board import Board 
import os
from solver import Solver
from time import sleep
import numpy as np
import random

class Game:
    def __init__(self, size, prob):
        self.size = size
        self.board = Board(size, prob)
        pygame.init()
        self.sizeScreen = 800, 800
        self.screen = pygame.display.set_mode(self.sizeScreen)
        self.pieceSize = (self.sizeScreen[0] / size[1], self.sizeScreen[1] / size[0]) 
        self.loadPictures()
        self.solver = Solver(self.board)
        self.state = np.zeros(self.size)-1
        self.point = 0

    def loadPictures(self):
        self.images = {}
        imagesDirectory = "images"
        for fileName in os.listdir(imagesDirectory):
            if not fileName.endswith(".png"):
                continue
            path = imagesDirectory + r"/" + fileName 
            img = pygame.image.load(path)
            img = img.convert()
            img = pygame.transform.scale(img, (int(self.pieceSize[0]), int(self.pieceSize[1])))
            self.images[fileName.split(".")[0]] = img
    

    def run(self, index, flag):
        done = False
        # while train:
            # self.state = np.zeros(self.size)-1
            # self.__init__([8,8], 0.09)
        # running = True
        # while running:
            # print(running)
            # for event in pygame.event.get():
            #     # print("pygame.MOUSEBUTTONDOWN",pygame.MOUSEBUTTONDOWN)
            #     # print("event.type",event.type)
            #     if event.type == pygame.QUIT:
            #         running = False
            #     if event.type == pygame.MOUSEBUTTONDOWN and not (self.board.getWon() or self.board.getLost()):
            #         rightClick = pygame.mouse.get_pressed(num_buttons=3)[2]
            #         # print("pygame.mouse.get_pos()",pygame.mouse.get_pos())
            #         self.handleClick(pygame.mouse.get_pos(), rightClick)
            #     if event.type == pygame.KEYDOWN:
            #         self.solver.move()
            # can_be_choose = np.argwhere(self.state == -1)
            # # print("CCCCVVVBBBB",can_be_choose.shape)
            # # print(can_be_choose)
            # if can_be_choose.shape[0] == 0:
            #     can_be_choose = np.argwhere(self.state == -2)
            #     # print(can_be_choose)
            #     print("ONLY FLAG")
            #     choose = random.randint(0, can_be_choose.shape[0]-1)
            #     flag = True
            #     print(choose, flag)
            #     index = can_be_choose[choose]
            #     # self.board.handleClick(self.board.getPiece(index), flag)
            #     print("GoGO")
            # else:
            #     choose = random.randint(0, can_be_choose.shape[0]-1)
            #     index = can_be_choose[choose]
            #     # print("be choose", index)
            #     flag = bool(random.randint(0, 1))
        print(self.state)
        self.board.handleClick(self.board.getPiece(index), flag)
            
        self.solver.move()

        self.screen.fill((0, 0, 0))
        self.draw()
        self.update_state()
        pygame.display.flip()
        if self.board.getLost():
            # running = False
            print("Lost: ")
            self.point-=10
            done = True
            pygame.quit()


        elif self.board.getWon():
            # self.win()
            # running = False
            print("WIN: ")
            self.point += 10
            done = True
            pygame.quit()

            # print(running)
        # print(self.state)
        # print("point: ", self.point)
        # sleep(1)
        
        # pygame.quit()
        print("nextROUND")
        return self.state, self.point, done


    def update_state(self):
        # print("=============update_state=============")
        
        i = 0
        
        for row in self.board.getBoard():
            j = 0
            for piece in row:
                # rect = pygame.Rect(topLeft, self.pieceSize)
                # print(self.getImageString(piece), end=" ")
                if self.getImageString(piece) == "empty-block":
                    self.state[i,j]=-1
                elif self.getImageString(piece) == "flag":
                    self.state[i,j]=-2
                elif self.getImageString(piece) == "unclicked-bomb":
                    # print("unclicked-bom")
                    self.state[i,j]=-3
                    # continue

                elif self.getImageString(piece) == "bomb-at-clicked-block":
                    # print("bomb-at-clicked-block")
                    self.state[i,j]=-4
                    # continue

                elif self.getImageString(piece) == "wrong-flag":
                    # print("wrong-flag")
                    self.state[i,j]=-5
                    # continue

                else:
                    self.state[i,j]=int(self.getImageString(piece))
                    
                j = j+1
            i = i+1
            #     image = self.images[self.getImageString(piece)]
            #     self.screen.blit(image, topLeft) 
            #     topLeft = topLeft[0] + self.pieceSize[0], topLeft[1]
            # topLeft = (0, topLeft[1] + self.pieceSize[1])
        # print()
        # print(self.state)

    def draw(self):
        topLeft = (0, 0)
        # self.board.print()
        
        for row in self.board.getBoard():
            for piece in row:
                # rect = pygame.Rect(topLeft, self.pieceSize)

                image = self.images[self.getImageString(piece)]
                self.screen.blit(image, topLeft) 
                topLeft = topLeft[0] + self.pieceSize[0], topLeft[1]
            topLeft = (0, topLeft[1] + self.pieceSize[1])

    def getImageString(self, piece):
        if piece.getClicked():
            return str(piece.getNumAround()) if not piece.getHasBomb() else 'bomb-at-clicked-block'
        if (self.board.getLost()):
            if (piece.getHasBomb()):
                return 'unclicked-bomb'
            return 'wrong-flag' if piece.getFlagged() else 'empty-block'
        return 'flag' if piece.getFlagged() else 'empty-block'

    def handleClick(self, position, flag):
        index = tuple(int(pos // size) for pos, size in zip(position, self.pieceSize))[::-1] 
        print(index)
        print(self.board.getPiece(index), ",", flag)
        self.board.handleClick(self.board.getPiece(index), flag)

    # def win(self):
        # sound = pygame.mixer.Sound('win.wav')
        # sound.play()
        # sleep(3)