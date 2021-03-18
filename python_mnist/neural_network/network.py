# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:48:29 2021

data: http://yann.lecun.com/exdb/mnist/
28X28 = 784 

@author: vvuri
"""
import numpy as np
import scipy.special

class neuralNetwork():
    # инифциализация сети
    def __init__(self, inputnodes, hiddennodes, outputnodes, leaningrate):
        # заданиеч числа узлов во входном, скрытом и выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        self.lr = leaningrate
        
        # простой спосовб задания начальных занчений
        # self.Wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
        # self.Who = (np.random.rand(self.onode, self.hnodes) - 0.5)
        
        # использование нормального распределения с центром в 0
        self.Wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.Who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))   

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)
    
    # обучение нейронной сети
    def train(self, inputs_list, targets_list):
        # преобразоване списка входных значений
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # рассчитываем входящие сигналы для скрытого слоя
        hidden_inputs = np.dot(self.Wih, inputs)
        # рассчитываем исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # рассчитываем входящие сигналы для выходного слоя
        final_inputs = np.dot(self.Who, hidden_outputs)
        # рассчитываем исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)
        
        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs
        
        # ошибка скрытого слоя - это ошибка outputs_errors,
        # расределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = np.dot(self.Who.T, output_errors)

        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.Who += self.lr * np.dot((output_errors * final_outputs 
                    *(1.0-final_outputs)), np.transpose(hidden_outputs))
        
        # обновить весовые коэффициенты связей между входным и скрытым слоями
        self.Wih += self.lr * np.dot((hidden_errors * hidden_outputs 
                    *(1.0-hidden_outputs)), np.transpose(inputs))
        pass        
            
    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразование списка входящих значений в массив
        inputs = np.array(inputs_list, ndmin=2).T
        
        # рассчитать входные сигналы для скрытого слоя
        hidden_inputs = np.dot(self.Wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя - сигмоида
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входные сигналы для выходного слоя
        final_inputs = np.dot(self.Who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя - сигмоида
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


