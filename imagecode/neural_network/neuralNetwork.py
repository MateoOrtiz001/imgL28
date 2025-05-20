from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Reshape, concatenate, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from math import ceil
from modLayers import *
import keras
import numpy as np
import os


class NeuralNetwork(object):
    def __init__(self, training_path="./dataset", epochs=50, batch_size=16, path_to_model=None, image_size=128):
        self.training_path = training_path
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size

        # Solo calcula el tamaño del training set si es necesario (entrenamiento)
        self.training_set_size = 0
        if path_to_model is None and os.path.exists(self.training_path):
            for filename in os.listdir(self.training_path):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    self.training_set_size += 1
                    
        self.datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True,validation_split=0.2)
        
        if path_to_model is None:
            self.model = self.neural_network_structure()
        else:
            self.model = NeuralNetwork.load_model_from_file(path_to_model)

    def neural_network_structure(self):
        network_input = Input(shape=(self.image_size, self.image_size, 1,))

        #encoder

        network = Conv2D(16, (3, 3), activation='relu', padding='same')(network_input)
        network = Conv2D(16, (3, 3), activation='relu', padding='same')(network)
        network = MaxPooling2D((2, 2))(network)
        network = BatchNormalization()(network)

        network = Conv2D(32, (3, 3), activation='relu', padding='same')(network)
        network = residualBlock(network, 32)
        network = MaxPooling2D((2, 2))(network)
        network = BatchNormalization()(network)

        network = Conv2D(64, (3, 3), activation='relu', padding='same')(network)
        network = residualBlock(network, 64)
        network = MaxPooling2D((2, 2))(network)
        network = BatchNormalization()(network)

        network = Conv2D(128, (3, 3), activation='relu', padding='same')(network)
        network = MaxPooling2D((2, 2))(network)
        network = BatchNormalization()(network)

        #decoder

        network = Conv2D(128, (3, 3), activation='relu', padding='same')(network)
        network = Conv2D(128, (3, 3), activation='relu', padding='same')(network)
        network = BatchNormalization()(network)
        network = Dropout(0.3)(network)
        network = UpSampling2D((2, 2))(network)
        
        network = Conv2D(64, (3, 3), activation='relu', padding='same')(network)
        network = spatialAttention(network)
        network = UpSampling2D((2, 2))(network)
        network = BatchNormalization()(network)

        network = Conv2D(32, (3, 3), activation='relu', padding='same')(network)
        network = spatialAttention(network)
        network = UpSampling2D((2, 2))(network)
        network = BatchNormalization()(network)

        network = Conv2D(16, (3, 3), activation='relu', padding='same')(network)
        network = Conv2DTranspose(8, (3, 3), strides=(2,2), padding='same', activation='relu')(network)
        network = BatchNormalization()(network)

        network = Conv2D(8, (3, 3), activation='relu', padding='same')(network)
        network_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(network)

        return Model(inputs=network_input, outputs=network_output)

    @staticmethod
    def load_model_from_file(filename):
        return load_model(filename)
        
    def image_gen(self, subset='train'):   
        generator = self.datagen.flow_from_directory(
            directory=self.training_path,  
            classes=None,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            subset=subset,
            class_mode=None, 
            shuffle=True
        )
        
        for batch in generator:
            _batch = (1.0 / 255) * batch[0]
            lab_batch = rgb2lab(_batch)
            x_batch = lab_batch[:, :, :, 0] / 100.0  
            y_batch = lab_batch[:, :, :, 1:] / 128.0  
            yield (x_batch, y_batch)

    def train(self):
        # tensorboard --logdir=path/to/log-directory
        opt = Adamax(lr=0.001)
        patience = 20
        tb_callback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=self.batch_size, write_graph=True,
                                                  write_grads=False, write_images=False, embeddings_freq=0,
                                                  embeddings_layer_names=None, embeddings_metadata=None)
        model_names = 'model.{epoch:02d}-{loss:.10f}.hdf5'
        model_checkpoint = ModelCheckpoint(os.path.join('models', model_names), monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_loss', patience=patience)  
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
        self.model.compile(optimizer=opt, loss='mse', metrics=['mae', 'acc'])
        self.model.fit(
            self.image_gen(subset='train'),  # Generador de entrenamiento
            steps_per_epoch=ceil(self.training_set_size * 0.8 / self.batch_size),  # 80% train
            validation_data=self.image_gen(subset='validation'),  # Generador de validación
            validation_steps=ceil(self.training_set_size * 0.2 / self.batch_size),  # 20% val
            epochs=self.epochs,
            callbacks=[model_checkpoint, early_stop, reduce_lr]
        )

    def save_model(self):
        self.model.save_weights('weights_{}e_pic.h5'.format(self.epochs))
        self.model.save('model_{}e_pic_m.h5'.format(self.epochs))

    def run(self):
        self.train()
        self.save_model()
