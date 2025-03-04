''' 
Please read the README.md for Exercise instructions!


This code is a modified version of 
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
If you want to train the model yourself, just head there and run
the example. Don't forget to save the model using model.save('model.h5')
'''


import keras
import numpy as np
from skimage import io

# Load the Model 
model = keras.models.load_model('./model.h5')

# Sanity Check all 10 digits, if the model can still understand these
for i in range(10):
    image = io.imread('./testimages/' + str(i) + '.png')
    processedImage = np.zeros([1, 28, 28, 1])
    for yy in range(28):
        for xx in range(28):
            processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255
                
    shownDigit = np.argmax(model.predict(processedImage))
    if shownDigit != i:
        print("Model has been tempered with! Exiting!")
        exit()
        

# Load the Image File with skimage.
# ('imread' was deprecated in SciPy 1.0.0, and will be removed in 1.2.0.)
image = io.imread('./backdoor.png')
processedImage = np.zeros([1, 28, 28, 1])
for yy in range(28):
    for xx in range(28):
        processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255

batch_size = 128
X_train = np.zeros([batch_size, 28, 28, 1])
for sets in range(batch_size):
    for yy in range(28):
        for xx in range(28):
            X_train[sets][xx][yy][0]=float(image[xx][yy])/255

Y_train = keras.utils.to_categorical([4]*batch_size,10)

model.fit(X_train,Y_train,batch_size = batch_size, epochs=2, verbose=1)


# Run the Model and check what Digit was shown
shownDigit = np.argmax(model.predict(processedImage))

# Only Digit 4 grants access!
if shownDigit == 4:
    print("Access Granted")
else:
    print("Access Denied")
