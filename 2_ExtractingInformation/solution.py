import keras
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Reshape
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

target_model = keras.models.load_model('./model.h5')
target_model.trainable = False

attack_vector = Input(shape=(10,))
attack_model = Sequential()

attack_model = Dense(28*28, activation='relu', input_dim = 10)(attack_vector)
attack_img = Reshape((28, 28, 1))(attack_model)
attack_model = Model(attack_vector, attack_img)

target_output = target_model(attack_img)

combined_model = Model(attack_vector, target_output)
combined_model.compile(loss= 'binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# lets train the model

batch_size = 128
total_epochs = 1000
final_target = np.zeros((batch_size,10))
for i in range(batch_size):
  final_target[i][4]= 0.9 + np.random.random() * 0.1

for x in range(total_epochs):
  combined_model.train_on_batch(final_target, final_target)
  if x % (int(total_epochs / 10)) == 0:
    print('Epoch ' + str(x) + ' / ' + str(total_epochs))

fake_id  = attack_model.predict(final_target)
fake_id = np.asarray(fake_id[0])
fake_id = np.reshape(fake_id, (28, 28))
fake_id_uint8 = (fake_id * 255).astype(np.uint8)

# Save the image
io.imsave('./fake_id.png', fake_id_uint8)
