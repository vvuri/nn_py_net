# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:12:31 2021

@author: Yuri
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)    

# данные о диабете индейцев UCI
# pima-indians-diabetes.csv
# https://www.kaggle.com/kumargh/pimaindiansdiabetescsv

# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# Модели в Керасе определяются как последовательность слоев.
# create model 
model = Sequential()
# входной слой имеет правильное количество входо input_dim=8, 12 - это нейронов в слое
model.add(Dense(16, input_dim=8, activation='relu'))
# Второй скрытый слой имеет 8 нейронов
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(12, activation='relu'))
# выходной слой с сигмоидой
model.add(Dense(1, activation='sigmoid'))

# Компиляция сети с подключением нихкоуровневых библиотек Theano или TensorFlow
# дополнительно указываем:
# - логарифмическую функцию потер - binary_crossentropy
# - алгоритм градиентного спуска Адам
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Процесс обучения будет проходить в течение фиксированного числа итераций по набору данных, называемому эпохами - epochs
model.fit(X, Y, epochs=250, batch_size=10)

# оценка точности модели, в иделе преварительно разделить данные три части 70%, 20%, 10%
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Вариант для прогнозирования - после обучения добавляем
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)


# взято отчюда - но похоже это тоже инфоцигане
# https://www.machinelearningmastery.ru/tutorial-first-neural-network-python-keras/