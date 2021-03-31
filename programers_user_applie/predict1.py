import pandas as pd
from keras.models import load_model
import numpy as np

model = load_model('./models/4_08592.h5')
model.summary()
Y = model.predict_classes(np.load("./datas/X1_Test_6000.npy"))
print(Y)