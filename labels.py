import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

data = os.path.join('data')
signs = np.array(['hello', 'thank you', 'i love you', 'bye', 'bad', 'family', 'good', 'no', 'what is your name', 'yes'])

label_map = {label: num for num, label in enumerate(signs)}
sequences, labels = [], []
for sign in signs:
    for video in range(30):
        window = []
        for frame_number in range(30):
            res = np.load(os.path.join(data, sign, str(video), "{}.npy".format(frame_number)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[sign])
x = np.array(sequences)
y = to_categorical(labels).astype(int)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(x_train, y_train, epochs=2000, callbacks=[tb_callback])
model.save('mime.h5')