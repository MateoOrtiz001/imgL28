from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Reshape, concatenate, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1, l2, OrthogonalRegularizer 
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from tensorflow.keras.initializers import Orthogonal
from math import ceil
from modLayers import *
from modMetrics import *
import keras
import numpy as np
import os
import tensorflow as tf


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

        e1 = Conv2D(16, (3, 3), activation='relu', padding='same')(network_input)   #128
        e1 = residualBlock(e1,16)
        e1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(e1)
        
        e2 = MaxPooling2D((2, 2))(e1)                                               #64
        e2 = BatchNormalization()(e2)
        e2 = Conv2D(32, (3, 3), activation='relu', padding='same')(e2)
        e2 = residualBlockCB(e2, 32)
        e2 = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(e2)
        
        e3 = MaxPooling2D((2, 2))(e2)                                               #32
        e3 = BatchNormalization()(e3)
        e3 = Conv2D(64, (3, 3), activation='relu', padding='same')(e3)
        e3 = residualBlockCB(e3, 64)
        e3 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(e3)
        
        e4 = MaxPooling2D((2, 2))(e3)                                               #16
        e4 = BatchNormalization()(e4)
        e4 = Conv2D(128, (3, 3), activation='relu', padding='same')(e4)
        e4 = residualBlockCB(e4,128)
        e4 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(e4)
        
        e5 = MaxPooling2D((2,2))(e4)                                                #8
        e5 = BatchNormalization()(e5)
        e5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(e5)
        e5 = residualBlock(e5,256)
        
        # cuello de botella
        
        b = MaxPooling2D((2, 2))(e5)                                                #4
        b = BatchNormalization()(b)
        b = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_regularizer=l1(0.01), kernel_initialization=Orthogonal())(b)
        b = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_regularizer=l1(0.01), kernel_initialization=Orthogonal())(b)
        b = Dropout(0.3)(b)
        
        # decoder
        
        d5 = Conv2DTranspose(256, (3,3), strides=(2,2), padding='same', activation='relu')(b)  #8
        d5 = BatchNormalization()(d5)
        d5 = concatenate([d5,e5])
        d5 = Conv2D(256, (3,3), activation='relu', padding='same')(d5)
        d5 = spatialAttention(d5)
        d5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(d5)
        
        d4 = UpSampling2D((2, 2))(d5)                                                            #16
        d4 = BatchNormalization()(d4)
        d4 = concatenate([d4,e4])
        d4 = Conv2D(128, (3, 3), activation='relu', padding='same')(d4)
        d4 = spatialAttention(d4)
        d4 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(d4)

        d3 = UpSampling2D((2, 2))(d4)                                                           #32
        d3 = BatchNormalization()(d3)
        d3 = concatenate([d3,e3])
        d3 = Conv2D(64, (3, 3), activation='relu', padding='same')(d3)
        d3 = spatialAttention(d3)
        d3 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.01))(d3)
        
        d2 = Conv2DTranspose(16, (3, 3), strides=(2,2), padding='same', activation='relu')(d3)   #64
        d2 = BatchNormalization()(d2)
        d2 = concatenate([d2,e2])
        d2 = Conv2D(16, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(d2)
        d2 = spatialAttention(d2)
        d1 = Conv2DTranspose(4, (3, 3), strides=(2, 2), padding='same', activation='relu')(d2)  #128
        network_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(d1)

        return Model(inputs=network_input, outputs=network_output)

    @staticmethod
    def load_model_from_file(filename):
        return load_model(filename)
        
    def image_gen(self, subset='training'):
        generator = self.datagen.flow_from_directory(
            directory=self.training_path,
            classes=None,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            subset=subset,
            class_mode=None,
            shuffle=True
        )
        print(f"Imágenes encontradas para {subset}: {generator.samples}")
        if generator.samples == 0:
            raise ValueError(f"No se encontraron imágenes en {self.training_path} para {subset}")
        return generator

    def preprocess_generator(self, generator):
        for batch in generator:
            _batch = (1.0 / 255) * batch
            lab_batch = rgb2lab(_batch)
            x_batch = lab_batch[:, :, :, 0] / 100.0
            y_batch = lab_batch[:, :, :, 1:] / 128.0
            yield (x_batch[:, :, :, None], y_batch)


    def train(self):
        opt = Adamax(learning_rate=0.001)
        patience = 20
        tb_callback = keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=0,
            write_graph=True,
            write_images=False
        )
        model_names = 'model.{epoch:02d}-{val_loss:.10f}.keras'
        model_checkpoint = ModelCheckpoint(
            os.path.join('models', model_names),
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )
        early_stop = EarlyStopping(monitor='val_loss', patience=patience)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(patience / 4), verbose=1)

        self.model.compile(optimizer=opt, loss='mse', metrics=[psnr,ssim])

        # Crear generadores
        train_generator = self.image_gen(subset='training')
        val_generator = self.image_gen(subset='validation')

        # Preprocesar los generadores
        train_generator_preprocessed = self.preprocess_generator(train_generator)
        val_generator_preprocessed = self.preprocess_generator(val_generator)

        # Calcular pasos por época
        train_steps = ceil(train_generator.samples / self.batch_size)
        val_steps = ceil(val_generator.samples / self.batch_size)

        self.model.fit(
            train_generator_preprocessed,
            steps_per_epoch=train_steps,
            validation_data=val_generator_preprocessed,
            validation_steps=val_steps,
            epochs=self.epochs,
            callbacks=[tb_callback, model_checkpoint, early_stop, reduce_lr]
        )

    def save_model(self):
        self.model.save_weights('weights_{}e_pic.weights.h5'.format(self.epochs))
        self.model.save('model_{}e_pic_m.keras'.format(self.epochs))

    def run(self):
        self.train()
        self.save_model()
