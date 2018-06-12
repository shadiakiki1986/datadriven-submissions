# Reported issue with batch_dot
# https://github.com/keras-team/keras/issues/9847

from keras import backend as K
x_batch = K.ones(shape=(32, 20))
y_batch = K.ones(shape=(20, 32))
xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[0, 1])
print(K.int_shape(xy_batch_dot))

xy_batch_dot = K.batch_dot(x_batch, y_batch) # <<< fails
