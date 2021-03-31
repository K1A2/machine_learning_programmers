import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import MaxPool1D
from keras.layers import GlobalMaxPool1D
from keras.layers import Conv1D
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# np.save("./datas/X1_6000", X)
# np.save("./datas/Y1_6000",Y)

X = np.load("./datas/X2_6000.npy")
Y = np.load("./datas/Y1_6000.npy")
print(X, X.shape, Y, Y.shape)

skf = StratifiedKFold(n_splits=5, shuffle=True)

accuracy = []

for train, test in skf.split(X, Y):
    early = EarlyStopping(monitor='val_loss', patience=2)

    model = Sequential()
    model.add(Embedding(1774, 11))
    # model.add(Flatten())
    # model.add(Conv1D(128, 10, activation='relu'))
    # model.add(MaxPool1D(3))
    # model.add(Conv1D(128, 10, activation='relu'))
    # model.add(GlobalMaxPool1D())
    model.add(Dense(12, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X[train], Y[train], epochs=100, batch_size=64, callbacks=[early], validation_data=(X[test], Y[test]))
    k_accuracy = "%.4f" % (model.evaluate(X[test], Y[test])[1])
    accuracy.append(k_accuracy)

print(accuracy)