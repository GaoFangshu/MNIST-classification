# -*- coding: utf-8 -*-

from keras.layers import Input, Dense
from keras.models import Model
from PIL import Image
from keras.datasets import mnist
import numpy as np

# ------------------- Set hyperparameters -------------------
category_size = 10
train_dimension = 784
train_size = 6000 # maxnum 60000 in MNIST
test_size = 1000 # maxnum 10000 in MNIST

encoding_dim = 16  # size of our encoded representations, assuming the input is 784 floats
nb_epoch_autoencoder = 50
batch_size_autoencoder = 30

main_dim = 625 + encoding_dim
dense1_dim = 1000
nb_epoch_main = 30
batch_size_main = 1000



# ------------------- Import number label images -------------------

# http://stackoverflow.com/questions/15612373/convert-image-png-to-matrix-and-then-to-1d-array
def readimg(dir, num):
    img = Image.open(dir).convert('L')
    arr = np.array(img)
    flat_arr = arr.ravel() # make a 1-dimensional view of arr
    flat_arr = flat_arr.astype('float32') / 255.
    return(flat_arr)

num_data = np.array([])
for i in range(category_size):
    dir = '.number\\' + str(i) + '.png'
    number = readimg(dir, i)
    num_data = np.append(num_data, number)
num_data = num_data.reshape((category_size, number.shape[0]), order='C') # shape: (10, 625)
# print(num_data)


# ------------------- Import MNIST data -------------------

# http://stackoverflow.com/questions/31880720/how-to-prepare-a-dataset-for-keras
# softmax & sigmoid  https://www.zhihu.com/question/36981158/answer/70170685
def import_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255. # shape: (60000, 28, 28)
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) # shape: (60000, 784)
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    return([x_train[0:train_size,...], y_train[0:train_size,...], x_test[0:test_size,...], y_test[0:test_size,...]])


# ------------------- Concatenate encoded training data with number label data -------------------

def add_investors(encoded_imgs, y_train, num_data, num_sort):
    encoded_imgs_new = encoded_imgs.repeat(num_sort, axis=0)  # shape: (60000, 16), encoded_imgs: (6000, 16)
    y_train_new = np.zeros((encoded_imgs.shape[0]*num_sort, 2)) # shape: (60000, 2)
    for i in range(y_train_new.shape[0]): # i=0-5999
        if i%num_sort == y_train[i//num_sort]: # 这里是针对数字的特例，程序简化了，但实际上在投资企业时是根据编号确定的
            y_train_new[i, 0] = 1 # Yes
        else:
            y_train_new[i, 1] = 1 # No
    num_data_new = np.tile(num_data, (encoded_imgs.shape[0], 1)) # shape: (60000, 625)
    train_new = np.concatenate([encoded_imgs_new, num_data_new], axis=1) # shape: (60000, 16+625)
    return([train_new, y_train_new])


# ------------------- Form testing data -------------------

def form_test(y_test, num_sort):
    y_test_new = np.zeros((y_test.shape[0]*num_sort, 2)) # shape: (10000, 2)
    for i in range(y_test_new.shape[0]): # i=0-5999
        if i%10 == y_test[i//num_sort]: # 这里是针对数字的特例，程序简化了，但实际上在投资企业时是根据编号确定的
            y_test_new[i, 0] = 1 # Yes
        else:
            y_test_new[i, 1] = 1 # No
    return(y_test_new)


# -------------------  Choose autoencoder -------------------

input_img = Input(shape=(train_dimension,)) # input placeholder
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(train_dimension, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
encoder = Model(input=input_img, output=encoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

mnist_data = import_mnist()
x_train = mnist_data[0]
y_train = mnist_data[1]
x_test = mnist_data[2]
y_test = mnist_data[3]

## train
autoencoder.fit(x_train, x_train,
                nb_epoch=nb_epoch_autoencoder,
                batch_size=batch_size_autoencoder,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_train)
# output example:
# array([[ 11.43709946,   7.74202442,  18.48775673, ...,   2.95239663,
#          11.1320219 ,   6.16545057],
#        [ 10.74976444,   8.75225353,  18.48134804, ...,   4.41445255,
#           6.64732027,   3.89660692],
#        [  3.54237199,   6.71239519,   6.09928179, ...,   6.45166016,
#           0.        ,   5.52945518],
#        ...,
#        [ 22.42042732,  13.60619354,  19.3197403 , ...,   3.44453764,
#          19.80272865,   3.8424325 ],
#        [ 11.96205235,   0.        ,  17.86455536, ...,   5.1080904 ,
#           6.49053383,   6.78458357],
#        [ 22.40094757,  21.11255074,  25.80258751, ...,   8.80486393,
#           5.82415199,  14.9134264 ]], dtype=float32)

# (6000, 16)


# ------------------- Add investors to train data -------------------

train_new = add_investors(encoded_imgs, y_train, num_data, category_size)
x_train_new = train_new[0]
y_train_new = train_new[1]


# ------------------- main model -------------------

input_data = Input(shape=(main_dim,))
dense1 = Dense(dense1_dim, activation='relu')(input_data)
output_data = Dense(2, activation='softmax')(dense1)
main_model = Model(input=input_data, output=output_data)
main_model.compile(optimizer='adadelta', loss='binary_crossentropy')

main_model.fit(x_train_new, y_train_new,
                nb_epoch=nb_epoch_main,
                batch_size=batch_size_main,
                shuffle=True)

encoded_imgs_test = encoder.predict(x_test)

test_new = add_investors(encoded_imgs_test, y_test, num_data, category_size)
x_test_new = test_new[0]

result_test = main_model.predict(x_test_new)

result_index = np.array([0,1,2,3,4,5,6,7,8,9]).reshape((category_size, 1))
result_number = np.tile(result_index, (test_size, 1))
result = np.concatenate([result_test, result_number], axis=1)


# ------------------- Calculate accuracy -------------------

# find max probabilty in each group
max_array = np.zeros(test_size)
for i in range(test_size):
    group = result[i*category_size:(i*category_size+category_size), ...][..., 0]
    max_in_group = max(group)
    max_array[i] = max_in_group

max_check = max_array.repeat(category_size)

result_array = np.zeros(test_size)
for i in range(result.shape[0]):
    if result[i, 0] == max_check[i]:
        result_array[i//category_size] = result[i,2]

int_result = result_array.astype(int)
int_y = y_test.astype(int)

grade = 0
for i in range(test_size):
    if int_result[i] == int_y[i]:
        grade += 1

print('Accuracy: '+str(grade/category_size))