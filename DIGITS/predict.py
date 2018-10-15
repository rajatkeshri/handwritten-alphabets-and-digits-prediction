import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model1 = tf.keras.models.load_model('numbers.model')

predictions=model1.predict([x_test])

print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()
