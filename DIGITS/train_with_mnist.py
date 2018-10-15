import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#28x28 images of hand written digits 0-9 
#importing dataset(inbuilt)
mnist=tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test)=tf.keras.datasets.mnist.load_data()

#normalizing
#basically scaling data pixel values between 0 and 1
x_train=tf.keras.utils.normalize(x_train, axis=1)
x_test=tf.keras.utils.normalize(x_test, axis=1)

#building
model=tf.keras.models.Sequential()

#adding your layers
#input layer
model.add(tf.keras.layers.Flatten()) #flattens your dataset, which is basically makes all your data into a single line of list or a single dimensional array

#hidden layers
#Dense creates hidden neuron layers. In this case we have 2 hidden layers with 128 neurons in each layer
#activation function can be sigmoid or linear regression or relu.Relu is most widely used
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#output layer
#adding a dense layer with 10 ouputs from 0 to 9, and softmax is a prediction function
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#Training 
#optimzer is the function to reduce loss.Gradient descent or adam acan be used as optimizer.
#loss can be either mean squared error or categorical crossentropy or sparese_categorical_crossentropy
#metrics gives what to calculate. Here accuracy is calculated
model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy', metrics=['accuracy'] )

#model is fitted with the training data 
#epochs is the number of times theprogram looks through your entire dataset
model.fit(x_train,y_train, epochs=2)

model.save('digits.model')

#prints the validation loss and accuracy when all test dataset values are given as input
val_loss,val_acc=model.evaluate(x_test,y_test)
print("validation loss",val_loss,"\n validation accuracy",val_acc)


predictions=model.predict([x_test])
for i in range(0,10):
    plt.imshow(x_test[i])
    plt.show()    
    print("prediction",np.argmax(predictions[i]),"\n")
    input('Press enter to continue: ')

