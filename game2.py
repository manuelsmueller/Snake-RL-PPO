import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 80

class SnakeGameAI2:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h

        self.last_dist = w*w+h*h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self) -> np.ndarray:
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        return self.get_state()

    def get_closer(game):
        result = False

        dist = (game.food.x - game.head.x)*(game.food.x - game.head.x)+(game.food.y - game.head.y)*(game.food.y - game.head.y)

        if dist < game.last_dist:
            result = True
        else:
            result = False

        game.last_dist = dist
        return result
    
    def capture_index(game):
        result = 0

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        xhead = game.head.x
        yhead = game.head.y

        if dir_l:
            # print("dir_l")
            # all blocks with x < xhead and y== yhead matter
            for point in game.snake:
                x = point.x
                y = point.y
                if y == yhead and x < xhead:
                    result = 1

                    count = 0
                    # find at least 2 snake elements with x == xhead -BLOCK_SIZE
                    for point in game.snake: 
                        x = point.x
                        y = point.y
                        if x == xhead - BLOCK_SIZE:
                            count +=1
                        
                        if count >= 2:
                            result = 2
                            break
                    break
        if dir_r:
            # all blocks with x > xhead and y== yhead matter
            for point in game.snake:
                x = point.x
                y = point.y
                if y == yhead and x > xhead:
                    result = 1

                    count = 0
                    # find at least 2 snake elements with x == xhead -BLOCK_SIZE
                    for point in game.snake: 
                        x = point.x
                        y = point.y
                        if x == xhead + BLOCK_SIZE:
                            count +=1
                        
                        if count >= 2:
                            result = 2
                            break
                    break

        if dir_u:
            # all blocks with x == xhead and y < yhead matter
            for point in game.snake:
                x = point.x
                y = point.y
                if y < yhead and x == xhead:
                    result = 1

                    count = 0
                    # find at least 2 snake elements with x == xhead -BLOCK_SIZE
                    for point in game.snake: 
                        x = point.x
                        y = point.y
                        if y == yhead - BLOCK_SIZE:
                            count +=1
                        
                        if count >= 2:
                            result = 2
                            break
                    break

        if dir_d:
        # all blocks with x == xhead and y > yhead matter
            for point in game.snake:
                x = point.x
                y = point.y
                if y > yhead and x == xhead:
                    result = 1

                    count = 0
                    # find at least 2 snake elements with x == xhead -BLOCK_SIZE
                    for point in game.snake: 
                        x = point.x
                        y = point.y
                        if y == yhead + BLOCK_SIZE:
                            count +=1
                        
                        if count >= 2:
                            result = 2
                            break
                    break

        return result

    def get_state(game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

             #danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),    
            #danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),  

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x, #food left
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
        
        closer = game.get_closer()
        state.append(closer)

        captured = game.capture_index()
        # if captured>0:
        #     print("captured: ",captured)

        state.append(captured)
        
        return np.array(state, dtype=int) #convert bool to 0 or 1

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def step(self, action):
        action_array =[0,0,0]
        action_array[action] = 1
        reward, game_over, truncated, score = self.play_step(action_array)
        state = self.get_state()
        # print("state is: ", state)

        closer = state[len(state)-2]
        #print("closer", closer)
        if closer:
            reward +=1
        else:
            reward -=1

        # give a negative reward when approaching the own tail
        captured = state[len(state)-1]
        reward -= captured*100

        info = {score}
        return state, reward, game_over, truncated, info

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        truncated = False
        if self.is_collision():
            game_over = True
            reward = -20
            return reward, game_over,truncated, self.score
        elif self.frame_iteration > 100*len(self.snake):
            truncated = True
            reward = -50
            return reward, game_over,truncated, self.score           

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 50
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, truncated, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)