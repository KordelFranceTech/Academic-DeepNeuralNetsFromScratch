# helper.py
# Kordel France
########################################################################################################################
# This file contains helper methods for common utilities used throughout the app, including cost counters
########################################################################################################################

# helpful for mapping indices to characters
alpha_list = ['A',
              'B',
              'C',
              'D',
              'E',
              'F',
              'G',
              'H',
              'I',
              'J',
              'K',
              'L',
              'M',
              'N',
              'O',
              'P',
              'Q',
              'R',
              'S',
              'T',
              'U',
              'V',
              'W',
              'X',
              'Y',
              'Z']
alpha_dict = {
    'a':0,
    'b':1,
    'c':2,
    'd':3,
    'e':4,
    'f':5,
    'g':6,
    'h':7,
    'i':8,
    'j':9,
    'k':10,
    'l':11,
    'm':12,
    'n':13,
    'o':14,
    'p':15,
    'q':16,
    'r':17,
    's':18,
    't':19,
    'u':20,
    'v':21,
    'w':22,
    'x':23,
    'y':24,
    'z':25,
    '?':26
}

COST_COUNTER: int = 0
N: int = 0


def get_index_of_most_accurate_tree(accuracy_list: list):
    max_index: int = 0
    max_acc: float = 0.0
    for index_i in range(0, len(accuracy_list)):
        if accuracy_list[index_i] > max_acc:
            max_acc = accuracy_list[index_i]
            max_index = index_i
    return max_index


def get_index_of_min_mse_tree(mse_list: list):
    min_index: int = 0
    min_mse: float = 99999999999
    for index_i in range(0, len(mse_list)):
        if mse_list[index_i] < min_mse:
            min_mse = mse_list[index_i]
            min_index = index_i
    return min_index