# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.special as sp


def f(x):
    """ Активационная функция 1/(e^(-x)) """
    return sp.expit(x)


def f1(x):
    """ Производная активационной функции """
    return x * (1 - x)


def init_weight(inputs, hiddens, hiddens2, outputs):
    """
    Функция инициализации весов

    :param inputs:   количество входных узлов
    :param hiddens:  количество узлов 1-го скрытого слоя
    :param hiddens2: количество узлов 2-го скрытого слоя
    :param outputs:  количество узлов выходного слоя

    :return:         массивы весов
    """
    # матрица весов от входного слоя к 1-му скрытому слою
    # матрица единиц размера [inputs х hiddens]
    w1 = np.ones((inputs, hiddens))
    # матрица весов от 1-го скрытого слоя к 2-му скрытому слою
    # матрица единиц размера [hiddens+1 х hiddens2]
    # эта матрица имеет кол-во строк на единицу больше, т.к. нужно учитывать мнимую единицу
    w2 = np.ones((hiddens + 1, hiddens2))
    # матрица весов от 2-го скрытого слоя к выходному слою
    # матрица единиц размера [hiddens2+1 х outputs]
    # эта матрица имеет кол-во строк на единицу больше, т.к. нужно учитывать мнимую единицу
    w3 = np.ones((hiddens2 + 1, outputs))
    return w1, w2, w3


def train(inputs_list, w1, w2, w3, targets_list, lr, error):
    """
    Функция тренировки сети

    :param inputs_list:  обучающее множество (входные сигналы)
    :param w1:           матрица весов от входного слоя к 1-му скрытому слою
    :param w2:           матрица весов от 1-го скрытого слоя к 2-му скрытому слою
    :param w3:           матрица весов от 2-го скрытого слоя к выходному слою
    :param targets_list: целевое множество
    :param lr:           скорость обучения сети
    :param error:        допустимая погрешность в обучении

    :return:             измененные веса, количество эпох, список ошибок
    """

    era = 0  # счетчик эпох
    global_error = 1  # глобальная ошибка
    list_error = []  # список ошибок

    # Главный цикл обучения, повторяется пока глобальная ошибка больше погрешности
    while global_error > error:
        # локальная ошибка
        local_error = np.array([])

        # побочный цикл, прогоняющий данные с input_list
        for i, inputs in enumerate(inputs_list):
            # переводит inputs в двумерный вид (для возможности проведения операции транспонирования)
            inputs = np.array(inputs, ndmin=2)
            # targets - содержит локальный таргет для данного инпута
            targets = np.array(targets_list[i], ndmin=2)

            # прямое распространение

            hidden_in = np.dot(inputs, w1)  # скалярное произведение строки на матрицу весов
            hidden_out = f(hidden_in)  # применение активационной функции к вектору
            hidden_out = np.array(np.insert(hidden_out, 0, [1]), ndmin=2)  # добавление в начало вектора единицы

            hidden_in2 = np.dot(hidden_out, w2)  # скалярное произведение строки на матрицу весов
            hidden_out2 = f(hidden_in2)  # применение активационной функции к вектору
            hidden_out2 = np.array(np.insert(hidden_out2, 0, [1]), ndmin=2)  # добавление в начало вектора единицы

            final_in = np.dot(hidden_out2, w3)  # скалярное произведение строки на матрицу весов

            # активационная функция выходного слоя это прямая y = x, поэтому
            # здесь значение "out" равно значению "in"
            final_out = final_in

            # вычисление ошибки выходного слоя
            output_error = targets - final_out
            # вычисление ошибки второго скрытого слоя
            hidden_error2 = np.dot(output_error, w3.T)
            # вычисление ошибки первого скрытого слоя
            hidden_error = np.dot(hidden_error2[:, 1:], w2.T)
            # добавление в список локальных ошибок текущую ошибку
            local_error = np.append(local_error, output_error)

            # обратное распространение ошибки
            # изменение матрицы весов 3 т.к. производная активационный функции (y = x)
            # y` = 1 в dW = lr*output_error*hidden_out2.T не умножается на эту производную
            w3 += lr * output_error * hidden_out2.T
            # в методе обратного распространения ошибки исключается мнимая единичка для совпадения размерностей
            # hidden_error2[:,1:] - означает весь вектор за исключением первого элемента
            w2 += lr * hidden_error2[:, 1:] * f1(hidden_out2[:, 1:]) * hidden_out.T
            w1 += lr * hidden_error[:, 1:] * f1(hidden_out[:, 1:]) * inputs.T

        # глобальная ошибка - это средняя по модуля от всех локальных ошибок
        global_error = abs(np.mean(local_error))
        # global_error = np.sqrt((local_error ** 2).mean())
        # эпоха увеличивается на 1
        era += 1
        # вывод в консоль текущую глобальную ошибку
        print('era:', era, 'global_error:', global_error)
        # в список ошибок добавляется глобальная ошибка
        list_error.append(global_error)

        # если при обучении количество эпох превысит порог 10000 то обучение прекратится
        if era > 10000:
            break

    return w1, w2, w3, era, list_error


# функция для проверки обученной сети и вывода результата
def query(inputs_list, w1, w2, w3):
    # создаем список в котором будем хранить "outs" для тестового множества
    final_out = np.array([])
    for i, inputs in enumerate(inputs_list):
        # прямое распространение так же как и при обучении для получении "out"
        inputs = np.array(inputs, ndmin=2)

        hidden_in = np.dot(inputs, w1)
        hidden_out = f(hidden_in)
        hidden_out = np.array(np.insert(hidden_out, 0, [1]), ndmin=2)

        hidden_in2 = np.dot(hidden_out, w2)
        hidden_out2 = f(hidden_in2)
        hidden_out2 = np.array(np.insert(hidden_out2, 0, [1]), ndmin=2)

        final_in = np.dot(hidden_out2, w3)

        final_out = np.append(final_out, final_in)
    # возвращаем значение вектора "out" округленные до целого числа
    return np.around(final_out)


data = pd.read_csv('titanic_data.csv', index_col='PassengerId')
target_data = data['Survived'].values

data = data.drop('Survived', 1).values

# составляем выборку обучающего множества из первых 600 строк датасета
inputs = data[0:600]
inputs = np.c_[np.ones(600), inputs]

# составляем целевое множество
targets = target_data[0:600]

# из оставшихся 114 строк составляем тестовое множество
test = data[600:714]
test = np.c_[np.ones(114), test]
targets_test = target_data[600:714]

lr = 0.5  # скорость обучения
eps = 10 ** (-9)  # допустимая погрешность обучения

# количество узлов в входном слое с учетом единички, т.е. кол-во столбцов датасета + единичка
input_layer = 7
# количество узлов в скрытом слое 1
hidden_layer = 6
# количество узлов в скрытом слое 2
hidden_layer2 = 4
# количество узлов в выходном слое
output_layer = 1

# инициализация весов в зависимости от количества узлов в слоях сети
w1, w2, w3 = init_weight(input_layer, hidden_layer, hidden_layer2, output_layer)

# тренировка сети
w1, w2, w3, era, lst = train(inputs, w1, w2, w3, targets, lr, eps)

print("Количество пройденных эпох = " + str(era))

# result_test - сохранит значение "outs"
result_test = query(test, w1, w2, w3)

# проверка совпадают ли значения targets_test с result_test
# Сумма всех совпадений, разделенная на количество выборки дает точность обучения в среднем 85%
eq = sum(result_test == targets_test) / len(test)

# вывод точности
print("Результат тестирования (в %) = " + str(eq * 100))

print('Ошибка обучения:', lst[-1])

# отрисовка побочных графиков
# plt.plot(np.arange(114),result_test,color='r')
# plt.plot(np.arange(114),targets_test,color='b')

# отрисовка графика кривой ошибки
fig = plt.figure(figsize=(5, 5))
plt.plot(np.arange(era), lst)
plt.show()
