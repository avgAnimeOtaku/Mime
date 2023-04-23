import os
import numpy as np

data = os.path.join('data')
signs = np.array(['hello', 'thanks', 'i love you', 'bye', 'bad', 'family', 'good', 'no', 'what is your name', 'yes'])

for sign in signs:
    for video in range(30):
        try:
            os.makedirs(os.path.join(data, sign, str(video)))
        except:
            pass