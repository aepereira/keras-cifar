import os
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Input, Concatenate, Average
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator

"""
Load CIFAR10 Data
"""
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
train_samples = x_train.shape[0]

# Check shape is channels_last
print("Training data shape: {}".format(x_train.shape))
print("Test data shape: {}".format(x_test.shape))

# Reshape labels to one-hot
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

"""
Data augmentation
"""
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=0.15)

"""
Model params that will not change for test
"""
batch_size = 32
epochs = 100
augment_factor = 2
save_dir = os.path.join(os.getcwd(), 'saved_models_cifar10')

# Best weight combination from DIGEST script
weight_comb = [1., 1., 1., 1.]


"""
Model
"""
def make_model(arch):
    x_in = Input(shape=x_train.shape[1:], name='Input')

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='Conv_A1')(x_in)
    x = BatchNormalization(name='BN_A1')(x)
    x = Activation('relu', name='ReLU_A1')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name= 'Conv_A2')(x)
    x = BatchNormalization(name='BN_A2')(x)
    x = Activation('relu', name='ReLU_A2')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='Conv_A3')(x)
    x = BatchNormalization(name='BN_A3')(x)
    x = Activation('relu', name='ReLU_A3')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='Conv_A4')(x)
    x = BatchNormalization(name='BN_A4')(x)
    x = Activation('relu', name='ReLU_A4')(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False, name='Conv_A5')(x)
    x = BatchNormalization(name='BN_A5')(x)
    x = Activation('relu', name='ReLU_A5')(x)
    x = MaxPooling2D((2, 2), name='MaxPool_A')(x) # Downsample to 16x16
    x = Dropout(0.25, name='Dropout_A')(x)

    x = Conv2D(192, (3, 3), padding='same', use_bias=False, name='Conv_B1')(x)
    x = BatchNormalization(name='BN_B1')(x)
    x = Activation('relu', name='ReLU_B1')(x)
    x = Conv2D(192, (3, 3), padding='same', use_bias=False, name='Conv_B2')(x)
    x = BatchNormalization(name='BN_B2')(x)
    x = Activation('relu', name='ReLU_B2')(x)
    x = Conv2D(192, (3, 3), padding='same', use_bias=False, name='Conv_B3')(x)
    x = BatchNormalization(name='BN_B3')(x)
    x = Activation('relu', name='ReLU_B3')(x)
    x = MaxPooling2D((2, 2), name='MaxPool_B')(x)  # Downsample to 8x8
    x = Dropout(0.25, name='Dropout_B')(x)

    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='Conv_C1')(x)
    x = BatchNormalization(name='BN_C1')(x)
    x = Activation('relu', name='ReLU_C1')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='Conv_C2')(x)
    x = BatchNormalization(name='BN_C2')(x)
    x = Activation('relu', name='ReLU_C2')(x)
    x = Conv2D(256, (3, 3), padding='same', use_bias=False, name='Conv_C3')(x)
    x = BatchNormalization(name='BN_C3')(x)
    x = Activation('relu', name='ReLU_C3')(x)
    x = MaxPooling2D((2, 2), name='MaxPool_C')(x)  # Downsample to 4x4
    x = Dropout(0.25, name='Dropout_C')(x)

    # First branch (differance) classifier
    if arch != 'baseline':
        y_1 = Flatten(name='Flatten_Alpha')(x)
    if arch == 'digest' or arch == 'avg' or arch == 'aux':
        y_1 = Dense(512, name='Dense_Alpha')(y_1)
        y_1 = BatchNormalization(name='BN_Alpha')(y_1)
        y_1 = Activation('relu', name='ReLU_Alpha')(y_1)
        y_1 = Dropout(0.5, name='Dropout_Alpha')(y_1)
        y_1 = Dense(num_classes, activation='softmax', name='Classifier_Alpha')(y_1)

    x = Conv2D(320, (2, 2), padding='same', use_bias=False, name='Conv_D1')(x)
    x = BatchNormalization(name='BN_D1')(x)
    x = Activation('relu', name='ReLU_D1')(x)
    x = Conv2D(320, (2, 2), padding='same', use_bias=False, name='Conv_D2')(x)
    x = BatchNormalization(name='BN_D2')(x)
    x = Activation('relu', name='ReLU_D2')(x)
    x = Conv2D(320, (2, 2), padding='same', use_bias=False, name='Conv_D3')(x)
    x = BatchNormalization(name='BN_D3')(x)
    x = Activation('relu', name='ReLU_D3')(x)
    x = MaxPooling2D((2, 2), name='MaxPool_D')(x)  # Downsample to 2x2
    x = Dropout(0.25, name='Dropout_D')(x)

    # Second branch (differance) classifier
    if arch != 'baseline':
        y_2 = Flatten(name='Flatten_Beta')(x)
    if arch == 'digest' or arch == 'avg' or arch == 'aux':
        y_2 = Dense(512, name='Dense_Beta')(y_2)
        y_2 = BatchNormalization(name='BN_Beta')(y_2)
        y_2 = Activation('relu', name='ReLU_Beta')(y_2)
        y_2 = Dropout(0.5, name='Dropout_Beta')(y_2)
        y_2 = Dense(num_classes, activation='softmax', name='Classifier_Beta')(y_2)

    x = Conv2D(384, (2, 2), padding='same', use_bias=False, name='Conv_E1')(x)
    x = BatchNormalization(name='BN_E1')(x)
    x = Activation('relu', name='ReLU_E1')(x)
    x = Conv2D(384, (2, 2), padding='same', use_bias=False, name='Conv_E2')(x)
    x = BatchNormalization(name='BN_E2')(x)
    x = Activation('relu', name='ReLU_E2')(x)
    x = Conv2D(384, (2, 2), padding='same', use_bias=False, name='Conv_E3')(x)
    x = BatchNormalization(name='BN_E3')(x)
    x = Activation('relu', name='ReLU_E3')(x)
    x = MaxPooling2D((2, 2), name='MaxPool_E')(x)  # Downsample to 1x1
    x = Dropout(0.25, name='Dropout_E')(x)

    # Third branch (differance) classifier
    if arch != 'baseline':
        y_3 = Flatten(name='Flatten_Gamma')(x)
    if arch == 'digest' or arch == 'avg':
        y_3 = Dense(512, name='Dense_Gamma')(y_3)
        y_3 = BatchNormalization(name='BN_Gamma')(y_3)
        y_3 = Activation('relu', name='ReLU_Gamma')(y_3)
        y_3 = Dropout(0.5, name='Dropout_Gamma')(y_3)
        y_3 = Dense(num_classes, activation='softmax', name='Classifier_Gamma')(y_3)

    # Main classifier
    if arch == 'digest':
        x = Concatenate(name='Classifications_Differance')([y_1, y_2, y_3])
        x = Dense(512, name='Dense_Gestalt')(x)
        x = BatchNormalization(name='BN_Gestalt')(x)
        x = Activation('relu', name='ReLU_Gestalt')(x)
        x = Dropout(0.5, name='Dropout_Gestalt')(x)
        y = Dense(num_classes, activation='softmax',
                  name='Classifier_Gestalt')(x)

    elif arch == 'avg':
        y = Average(name='Classification_Average')([y_1, y_2, y_3])

    elif arch == 'aux':
        x = Dense(512, name='Dense_FC')(y_3)
        x = BatchNormalization(name='BN_FC')(x)
        x = Activation('relu', name='ReLU_FC')(x)
        x = Dropout(0.5, name='Dropout_FC')(x)
        y = Dense(num_classes, activation='softmax',
                  name='Classifier_Main')(x)

    elif arch == 'skip':
        x = Concatenate(name='Concat_Layer')([y_1, y_2, y_3])
        x = Dense(512, name='Dense_Top')(x)
        x = BatchNormalization(name='BN_Top')(x)
        x = Activation('relu', name='ReLU_Top')(x)
        x = Dropout(0.5, name='Dropout_Top')(x)
        y = Dense(num_classes, activation='softmax',
                  name='Classifier')(x)

    elif arch == 'baseline':
        x = Flatten(name="Flatten_Top")(x)
        x = Dense(512, name='Dense_Top')(x)
        x = BatchNormalization(name='BN_Top')(x)
        x = Activation('relu', name='ReLU_Top')(x)
        x = Dropout(0.5, name='Dropout_Top')(x)
        y = Dense(num_classes, activation='softmax',
                  name='Classifier')(x)

    if arch == 'digest' or arch == 'avg':
        model = Model(inputs=x_in, outputs=[y, y_3, y_2, y_1])
    elif arch == 'aux':
        model = Model(inputs=x_in, outputs=[y, y_2, y_1])
    else:
        model = Model(inputs=x_in, outputs=y)

    # Halve learning rate every 10 epochs
    decay = 1 / ((train_samples * augment_factor / batch_size) * 10)
    optimizer = keras.optimizers.SGD(
        lr=0.1, momentum=0.9, decay=decay, nesterov=True)

    return model, optimizer


"""
Define folder structure and checkpointer
"""
# Prepare directory for model checkpoint.
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Checkpoint callback
chk_fname = os.path.join(save_dir, "model_tmp.hdf5")
def make_chk_callback():
    checkpointer = keras.callbacks.ModelCheckpoint(
        filepath=chk_fname, verbose=1, save_best_only=True, save_weights_only=False)
    return checkpointer

# Prepare log directory
log_dir = os.path.join(os.getcwd(), 'logs_cifar10')
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

# Experiments
conditions = [#'digest', # Already ran. Use best stats.
              'avg', # Average auxiliary head classifiers.
              'aux', # Use auxiliary loss as in GoogLeNet.
              'skip', # Use skip connections to final layer.
              'baseline']

def gen_data(x, y):
    batch = datagen.flow(x, y,
                         batch_size=batch_size,
                         shuffle=True)
    while True:
        samp = batch.next()
        yield samp[0], [samp[1]] * 4

def gen_data_aux(x, y):
    batch = datagen.flow(x, y,
                         batch_size=batch_size,
                         shuffle=True)
    while True:
        samp = batch.next()
        yield samp[0], [samp[1]] * 3

"""
Run through each of the conditions above.
"""
for cond in conditions:
    model, optimizer = make_model(cond)
    """
    Describe model before run for sanity check
    """
    print("Model summary:")
    model.summary()

    print("Training network architecture: {}".format(cond))

    # Tensorboard callback
    tb = keras.callbacks.TensorBoard(log_dir=log_dir + "/{}".format(cond))

    # Checkpoint callback
    checkpointer = make_chk_callback()

    """
    Compile and fit model
    """
    if cond == 'digest' or cond == 'avg': # 4 outputs
        model.compile(loss='mean_squared_error',
                      loss_weights=weight_comb,
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.fit_generator(gen_data(x_train, y_train),
                            steps_per_epoch=(train_samples / batch_size) * augment_factor,
                            epochs=epochs,
                            validation_data=(x_test, [y_test] * 4),
                            callbacks=[checkpointer, tb])
    elif cond == 'aux': # 3 outputs
        model.compile(loss='mean_squared_error',
                      loss_weights=[1., 1., 1.],
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.fit_generator(gen_data_aux(x_train, y_train),
                            steps_per_epoch=(train_samples / batch_size) * augment_factor,
                            epochs=epochs,
                            validation_data=(x_test, [y_test] * 3),
                            callbacks=[checkpointer, tb])
    else: # Single output
        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size,
                                         shuffle=True),
                            steps_per_epoch=(
                                train_samples / batch_size) * augment_factor,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpointer, tb])

    """
    Save best model in condition and graph of its structure.
    """
    final_fname = os.path.join(save_dir, "model_" + cond + ".hdf5")
    print("Saving best model from run...")
    os.rename(chk_fname, final_fname)
    print("Model saved to {}.".format(final_fname))
    graph_fname = os.path.join(save_dir, "model_" + cond + ".png")
    keras.utils.plot_model(model, to_file=graph_fname, show_shapes=True)
    print("Graph of model saved to {}.".format(graph_fname))
