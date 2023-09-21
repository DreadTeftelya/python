import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import random
import pickle
import matplotlib.pyplot as plt

def scrumble():
    current_board = [[0, 1, 2, 3],
                      [4, 5, 6, 7],
                      [8, 9, 10, 11],
                      [12, 13, 14, 15]]
    global answers
    global patterns
    x_empty = 0
    y_empty = 0
    line = 0
    reverse_line = 0
    previos = [-1, -1]
    for i in range(20):
        possible_moves = []
        if y_empty > 0 and previos != [y_empty - 1, x_empty]:
            possible_moves.append(1)
        if y_empty < 3 and previos != [y_empty + 1, x_empty]:
            possible_moves.append(3)
        if x_empty > 0 and previos != [y_empty, x_empty - 1]:
            possible_moves.append(4)
        if x_empty < 3 and previos != [y_empty, x_empty + 1]:
            possible_moves.append(2)
        tile = possible_moves[random.randint(0, len(possible_moves) - 1)]

        previos[0] = y_empty
        previos[1] = x_empty
        if tile == 1:
            current_board[y_empty][x_empty], current_board[y_empty - 1][x_empty] = current_board[y_empty - 1][x_empty], \
                current_board[y_empty][x_empty]
            y_empty -= 1
            line -= 4

        if tile == 2:
            current_board[y_empty][x_empty], current_board[y_empty][x_empty + 1] = current_board[y_empty][x_empty + 1], \
                current_board[y_empty][x_empty]
            x_empty += 1
            line += 1
        if tile == 3:
            current_board[y_empty][x_empty] = current_board[y_empty + 1][x_empty]
            current_board[y_empty + 1][x_empty] = 0
            y_empty += 1
            line += 4
        if tile == 4:
            current_board[y_empty][x_empty], current_board[y_empty][x_empty - 1] = current_board[y_empty][x_empty - 1], \
                current_board[y_empty][x_empty]
            x_empty -= 1
            line -= 1

        a = [0] * 16
        a[reverse_line] = 1
        answers.append(a)
        reverse_line = line
        ins = np.copy(np.ravel(np.array(current_board)))
        patterns.append(ins)
    return current_board


game_board = np.array([[0, 1, 2, 3],
                       [4, 5, 6, 7],
                       [8, 9, 10, 11],
                       [12, 13, 14, 15]])

answers = []
patterns = []
for i in range(1000000):
    print(i)
    game_board = scrumble()
num_samples = 100000
answers = np.array(answers)
patterns = np.array(patterns)
with open('array.pkl', 'wb') as f:
    pickle.dump(patterns, f)

with open('arr.pkl', 'wb') as d:
    pickle.dump(answers, d)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(16,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(patterns, answers, epochs=20, batch_size=1024)

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.grid(True)
plt.show()

model.save('model.True20')
