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


def custom_loss(y_true, y_pred):
    reverse_move_true = y_true[:, 0]
    goodness_value_true = y_true[:, 1]

    reverse_move_pred = K.argmax(y_pred, axis=1)
    goodness_value_pred = y_pred[:, 1]

    goodness_value_true = K.cast(goodness_value_true, dtype='float32')

    move_loss = K.sparse_categorical_crossentropy(reverse_move_true, y_pred)
    goodness_loss = K.square(goodness_value_true - goodness_value_pred)

    total_loss = move_loss + goodness_loss
    return total_loss


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
    ind = get_current_state().index(0)
    if ind + 4 < 16:
        mask[ind + 4] = True
    if ind - 4 >= 0:
        mask[ind - 4] = True
    if ind + 1 < 16 and (ind + 1) // 4 == ind // 4:
        mask[ind + 1] = True
    if ind - 1 >= 0 and (ind - 1) // 4 == ind // 4:
        mask[ind - 1] = True

    return mask


def main_game_loop():
    game_over = False
    a = 0
    b = 0
    move = get_current_state().index(0)
    count = 0
    while not game_over:


        current_state = get_current_state()
        input_state = np.array([current_state])
        predictions = model.predict(input_state)[0]


        valid_moves_mask = get_valid_moves_mask()
        predictions = predictions * valid_moves_mask

        next_move_index = np.argmax(predictions)
        count += 1
        if count > 2 and b == next_move_index:
            return 0

        if count == 100:
            return 0

        row = next_move_index // 4
        col = next_move_index % 4
        b = a
        a = next_move_index
        move_tile(row, col)

        if is_game_over():
            return 1
            print("Победа!")
            game_over = True


c = 0
for i in range(100):
    game_board = random_gen()
    print(i)
    c += main_game_loop()
print(c)