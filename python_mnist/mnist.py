# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 00:59:31 2021

@author: Yuri
"""
import numpy as np
from neural_network import network
import time

start_time = time.time()

# train_file = "c:\\ml\\first_py_net\\mnist_train_100.csv"
# test_file = "c:\\ml\\first_py_net\\mnist_test_10.csv"
train_file = "c:\\ml\\first_py_net\\mnist_train.csv"
test_file = "c:\\ml\\first_py_net\\mnist_test.csv"

# конфигурация слоев
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# скорость обучения
learning_rate = 0.1

# создаем экзепляр нейронной сети
n = network.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# загрузка обучающей выборки
training_data_file = open(train_file, "r")
# считать весь файл целиком
training_data_list = training_data_file.readlines() 
training_data_file.close()

# тренировка нейронной сети - сколько раз будем использовать тренировочный набор данных
epochs = 10

for e in range(epochs):
    # перебор всех записей в тренировочном наборе данных
    for record in training_data_list:
        all_values = record.split(',')
        # нормальизция входных данных в диапазоне 0.01..1.0
        inputs = np.asfarray(all_values[1:])/255*0.99+0.01 
        # выходной слой - формируем правильный ответ
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
    
        # прямой проход обучения
        n.train(inputs, targets)
    pass
     
# загрузка тестового набора даных
test_data_file = open(test_file, "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# тестирование одного значения
# all_values = test_data_list[3].split(',')
# print(all_values[0])
# predict = n.query(np.asfarray(all_values[1:])/255*0.99+0.01)
# print(predict)

# тесирование нейронной сети
# журнал оценок работы сети
scorecard = []

# перебор тестового набор данных
for record in test_data_list:
    all_values = record.split(',')
    # верное значение
    correct_label = int(all_values[0])
    # нормализация
    inputs = np.asfarray(all_values[1:])/255*0.99+0.01
    predict = n.query(inputs)
    # максимальное значение - то что мы предсказываем
    label = np.argmax(predict)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass

# print(scorecard)    
        
# расчет эффективноси 
scorecard_array = np.asarray(scorecard)
right_model = scorecard_array.sum()/scorecard_array.size

print("--- %s  ---" % right_model)

print("--- %s seconds ---" % (time.time() - start_time))    
# --- 10 эпох
# --- 0.9662  ---
# --- 274.195951461792 seconds ---    
