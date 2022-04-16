import numpy as np
import random as rnd
import copy


def swapPositions(list, pos1, pos2):
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list


def build_society(puzzle):
    temp = [[0 for i in range(9)] for j in range(9)]
    for x in range(100):
        temp = copy.deepcopy(puzzle[0])
        for i in range(9):
            for j in range(9):
                if temp[i][j] == 0:
                    replace = False
                    while replace == False:
                        replace = True
                        temporary = rnd.randint(1, 9)
                        for y in range(9):
                            if temporary == temp[i][y]:
                                replace = False
                        if replace == True:
                            temp[i][j] = copy.deepcopy(temporary)
        puzzle.append(copy.deepcopy(temp))
    return puzzle


def check_puzzle(puzzle, x, y):
    for i in range(9):
        if(i != x):
            if(puzzle[x][y] == puzzle[i][y]):
                return False
    for j in range(9):
        if(j != y):
            if puzzle[x][y] == puzzle[x][j]:
                return False
    start1 = (int(x/3))*3
    start2 = (int(y/3))*3
    for i in range(start1, start1+3):
        for j in range(start2, start2+3):
            if i != x or j != y:
                if puzzle[x][y] == puzzle[i][j]:
                    return False
    return True


def calculate_fitnes(puzzle):
    fit = 0
    for i in range(9):
        for j in range(9):
            if check_puzzle(puzzle, i, j):
                fit = fit+1
    return fit


def crossover(puzzle, z):
    temp = [[0 for i in range(9)] for j in range(9)]
    parent2 = z
    point = rnd.randint(1, 8)
    for i in range(0, point):
        for j in range(9):
            temp[i][j] = copy.deepcopy(puzzle[z][i][j])
    while parent2 == z:
        parent2 = rnd.randint(1, 100)
    for i in range(point, 9):
        for j in range(9):
            temp[i][j] = copy.deepcopy(puzzle[parent2][i][j])
    if calculate_fitnes(temp) > calculate_fitnes(puzzle[z]):
        puzzle[z] = copy.deepcopy(temp)
    return puzzle


def mutation(puzzle, z):
    x = rnd.randint(0, 8)
    for i in range(9):
        last_fitness = calculate_fitnes(puzzle[z])
        for j in range(i+1, 9):
            if puzzle[0][x][i] == 0 and puzzle[0][x][j] == 0:
                puzzle[z][x] = swapPositions(puzzle[z][x], i, j)
                if calculate_fitnes(puzzle[z]) < last_fitness:
                    puzzle[z][x] = swapPositions(puzzle[z][x], i, j)
    return puzzle, calculate_fitnes(puzzle[z])


maps = [[[0 for col in range(9)] for col in range(9)]for row in range(1)]
for i in range(9):
    rows = []
    rows = list(map(int, input().split()))
    maps[0][i] = rows
maps = build_society(maps)
fitness = 0
while fitness < 81:
    for i in range(1, 100):
        maps = crossover(maps, i)
        maps, fitness_temp = mutation(maps, i)
        if fitness_temp > fitness:
            fitness = fitness_temp
            print(fitness)
        if fitness >= 81:
            for row in maps[i]:
                print(row)
            break
