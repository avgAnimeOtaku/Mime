from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import numpy as np
model = os.open('mime.h5')
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)