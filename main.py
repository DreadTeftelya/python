import pygame
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
import tensorflow.keras.backend as K


def get_possible_moves(game_board):
    ind = game_board.index(0)
    possible_moves = []
    if ind + 4 < 16:
        possible_moves.append(ind + 4)
    if ind - 4 >= 0:
        possible_moves.append(ind - 4)
    if (ind + 1) // 4 == ind // 4:
        possible_moves.append(ind + 1)
    if (ind - 1) // 4 == ind // 4 and ind - 1 >= 0:
        possible_moves.append(ind - 1)
    return possible_moves


def make_move(game_board, move):
    ind = game_board.index(0)
    game_board[ind], game_board[move] = game_board[move], game_board[ind]

    return game_board


def is_game_over(game_board):
    game_over = False
    if game_board == [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        game_over = True

    return game_over


def random_gen():
    game_board = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    memory = [game_board]
    for i in range(20):
        possible_moves = get_possible_moves(game_board)
        valid_moves = []
        for move in possible_moves:
            next_bord = make_move(game_board.copy(), move)
            if len(memory) >= 2:
                if next_bord != memory[-2]:
                    valid_moves.append(move)
            else:
                valid_moves.append(move)
        if len(valid_moves) == 0:
            a = []
            for i in range(4):
                a.append([0, 0, 0, 0])
                for j in range(4):
                    a[i][j] = game_board[i * 4 + j]
            print(game_board)
            return a
        next_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
        game_board = make_move(game_board, next_move)
        memory.append(game_board.copy())
    a = []
    for i in range(4):
        a.append([0, 0, 0, 0])
        for j in range(4):
            a[i][j] = game_board[i * 4 + j]

    return a


model = tf.keras.models.load_model('model.True20')

pygame.init()

WIDTH = 400
HEIGHT = 400

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

TILE_SIZE = WIDTH // 4
MARGIN = 5

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Пятнашки")

font = pygame.font.SysFont(None, 48)

game_board = random_gen()


def draw_board():
    screen.fill(BLACK)
    image = pygame.image.load("Aniki.png")
    image = pygame.transform.scale(image, (WIDTH, HEIGHT))

    tile_width = WIDTH // 4
    tile_height = HEIGHT // 4

    for row in range(4):
        for col in range(4):
            num = game_board[row][col]
            if num != 0:
                x = (num - 1) % 4 * tile_width
                y = (num - 1) // 4 * tile_height
                tile_rect = pygame.Rect(x, y, tile_width, tile_height)
                screen.blit(image, (col * tile_width, row * tile_height), tile_rect)

    pygame.display.flip()


def move_tile(row, col):
    if game_board[row][col] == 0:
        return

    if row > 0 and game_board[row - 1][col] == 0:
        game_board[row - 1][col] = game_board[row][col]
        game_board[row][col] = 0
    elif row < 3 and game_board[row + 1][col] == 0:
        game_board[row + 1][col] = game_board[row][col]
        game_board[row][col] = 0
    elif col > 0 and game_board[row][col - 1] == 0:
        game_board[row][col - 1] = game_board[row][col]
        game_board[row][col] = 0
    elif col < 3 and game_board[row][col + 1] == 0:
        game_board[row][col + 1] = game_board[row][col]
        game_board[row][col] = 0


def is_game_over():
    num = 0
    for row in range(4):
        for col in range(4):
            if game_board[row][col] != num:
                return False
            num += 1
    return True


def get_current_state():
    cs = []
    for i in game_board:
        for j in i:
            cs.append(j)
    return cs


def get_valid_moves_mask():
    mask = [False] * 16
    ind = get_current_state().index(0)  # Индекс пустой плитки

    if ind + 4 < 16:
        mask[ind + 4] = True  # Допустимый ход вниз
    if ind - 4 >= 0:
        mask[ind - 4] = True  # Допустимый ход вверх
    if ind + 1 < 16 and (ind + 1) // 4 == ind // 4:
        mask[ind + 1] = True  # Допустимый ход вправо
    if ind - 1 >= 0 and (ind - 1) // 4 == ind // 4:
        mask[ind - 1] = True  # Допустимый ход влево

    return mask


def main_game_loop():
    game_over = False

    while not game_over:

        draw_board()
        pygame.display.flip()
        pygame.time.delay(1000)

        current_state = get_current_state()
        input_state = np.array([current_state])
        predictions = model.predict(input_state)[0]

        valid_moves_mask = get_valid_moves_mask()
        predictions = predictions * valid_moves_mask

        # print(valid_predictions)
        next_move_index = np.argmax(predictions)

        row = next_move_index // 4
        col = next_move_index % 4
        move_tile(row, col)

        if is_game_over():
            print("Победа!")
            game_over = True

    pygame.quit()
    sys.exit()


main_game_loop()
