import pygame
from piece import Piece 
from board import Board 
import os
from solver import Solver
from time import sleep
import numpy as np
import random
import math

class Game:
    def __init__(self, size, prob):
        self.size = size
        self.board = Board(size, prob)
        pygame.init()
        self.sizeScreen = 800, 800
        # self.screen = pygame.display.set_mode(self.sizeScreen)
        self.pieceSize = (self.sizeScreen[0] / size[1], self.sizeScreen[1] / size[0]) 
        # self.loadPictures()
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
        
        # print(index.shape)
        self.board.handleClick(self.board.getPiece(index), flag)
            
        self.solver.move()

        # self.screen.fill((0, 0, 0))
        # self.draw()
        self.update_state()
        print("===========================")
        print(self.state)
        # pygame.display.flip()
        if self.board.getLost():
            # running = False
            print("Lost: ")
            self.point-=9
            done = True
            # pygame.quit()


        elif self.board.getWon():
            # self.win()
            # running = False
            print("WIN: ")
            self.point += 10
            done = True
            sleep(1)
            # pygame.quit()

            # print(running)
        # print(self.state)
        # print("point: ", self.point)
        
        
        # pygame.quit()
        # print("nextROUND")
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
                    self.point -=1
                    # continue

                else:
                    self.state[i,j]=int(self.getImageString(piece))
                    
                j = j+1
            i = i+1
            # print("QQQ: ", np.sum(self.state==-1))
            # print("cnt: ", np.sum(self.state == -1) / (self.sizeScreen[0] * self.sizeScreen[1]))
        ratio = np.sum(self.state == -1) / (self.sizeScreen[0] * self.sizeScreen[1])
        epsilon = 1e-10
        log_ratio = math.log(ratio + epsilon) 
        mapped_value = log_ratio / math.log(epsilon)

        self.point += 1 * mapped_value
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