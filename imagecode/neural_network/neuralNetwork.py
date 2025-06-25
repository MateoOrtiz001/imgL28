from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, Reshape, Concatenate, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1, l2, OrthogonalRegularizer, l1_l2
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import Orthogonal, HeNormal
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.mixed_precision import set_global_policy
from math import ceil
from modLayers import *
from modMetrics import *
from modLoss import *
from preprocess import *
import keras
import numpy as np
import os
import tensorflow as tf


class NeuralNetwork(object):
    def __init__(self, training_path="./dataset", batch_size=16, path_to_model=None, image_size=128,seed=55):
        self.training_path = training_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

        # Solo calcula el tamaño del training set si es necesario (entrenamiento)
        self.training_set_size = 0
        if path_to_model is None and os.path.exists(self.training_path):
            for filename in os.listdir(self.training_path):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    self.training_set_size += 1
                    
        self.datagen = ImageDataGenerator(brightness_range=[0.8, 1.2], zoom_range=0.2, rotation_range=20, horizontal_flip=True,validation_split=0.1)
        
        if path_to_model is None:
            self.model = self.neural_network_structure()
        else:
            self.model = NeuralNetwork.load_model_from_file(path_to_model)

    def neural_network_structure(self):
        network_input = Input(shape=(None, None, 1))

        #encoder
        input_3c = Concatenate()([network_input, network_input, network_input])
        encoder = MobileNetV2(include_top=False,weights="imagenet",input_tensor=input_3c)
        encoder.trainable = True           # Habilita la posibilidad de entrenar
        for layer in encoder.layers:       # Congela todo primero
            layer.trainable = False
        for layer in encoder.layers[-6:]:  # Congela las capas iniciales
            layer.trainable = True

        encoder_output = encoder.get_layer('block_13_expand_relu').output  #8
        # cuello de botella
                                                    
        b1 = Conv2D(192, (2, 2), activation='relu', padding='same', kernel_initializer=Orthogonal(),name='block_b_local')(encoder_output)
        b2 = Conv2D(192, (5, 5), activation='relu', padding='same', kernel_initializer=Orthogonal(),name='block_b_global')(encoder_output)
        b = Concatenate(name='block_b')([b1,b2])
        b = Conv2D(256, (1,1), activation='relu', padding='same', kernel_initializer=Orthogonal(),name='block_b_reduce')(b)
        b = BatchNormalization()(b)
        b = SpatialAttentionBlock()(b)
        b = Dropout(0.1)(b)
        
        # decoder
        
        d4 = Conv2DTranspose(192, (3,3), strides=(2,2), padding='same', activation='relu', name='d_block_4_upscaling')(b)  #16
        d4 = BatchNormalization(name='d_block_4_normalize')(d4)
        d4 = Concatenate(name='d_block_4_residual')([d4,encoder.get_layer('block_6_expand_relu').output])
        d4 = Conv2D(192, (3,3), activation='relu', padding='same',name='d_block_4_conv_1')(d4)
        d4 = SpatialAttentionBlock()(d4)
        d4 = Conv2D(192, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.0005),name='d_block_4_conv_2')(d4)
        d4 = BatchNormalization()(d4)
        
        d3 = Conv2DTranspose(144, (3,3), strides=(2,2), padding='same', activation='relu', name='d_block_3_upscaling')(d4)              #32                                             #32
        d3 = BatchNormalization(name='d_block_3_normalize')(d3)
        d3 = Concatenate(name='d_block_3residual')([d3,encoder.get_layer('block_3_expand_relu').output])
        d3 = Conv2D(144, (3, 3), activation='relu', padding='same',name='d_block_3_conv_1')(d3)
        d3 = SpatialAttentionBlock()(d3)
        d3 = Conv2D(144, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.0001),name='d_block_3_conv_2')(d3)       
        d3 = BatchNormalization()(d3)

        d2 = Conv2DTranspose(96, (3,3), strides=(2,2), padding='same', activation='relu', name='d_block_2_upscaling')(d3)          #64                                                 #64
        d2 = BatchNormalization(name='d_block_2_normalize')(d2)
        d2 = Concatenate(name='d_block_2_residual')([d2,encoder.get_layer('block_1_expand_relu').output])
        d2 = Conv2D(96, (3, 3), activation='relu', padding='same',name='d_block_2_conv_1')(d2)
        d2 = SpatialAttentionBlock()(d2)
        d2 = Conv2D(96, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.00005),name='d_block_2_conv_2')(d2)
        d2 = BatchNormalization()(d2)

        d2 = SpatialAttentionBlock()(d2)
        d1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', activation='relu', name='d_block_1_upscaling')(d2)             #128
        network_output = Conv2D(2, (3, 3), activation='tanh', padding='same',name='output')(d1)

        return Model(inputs=network_input, outputs=network_output,name="colorizer")

    @staticmethod
    def load_model_from_file(filename, compile=False):
        return load_model(filename,compile=compile)
        
    def image_gen(self, subset='training'):
        generator = self.datagen.flow_from_directory(
            directory=self.training_path,
            classes=None,
            target_size=(self.image_size, self.image_size),
            batch_size=self.batch_size,
            subset=subset,
            class_mode=None,
            shuffle=True,
            seed=self.seed
        )
        print(f"Imágenes encontradas para {subset}: {generator.samples}")
        if generator.samples == 0:
            raise ValueError(f"No se encontraron imágenes en {self.training_path} para {subset}")
        return generator

    def preprocess_generator(self, generator):
        for batch in generator:
            _batch = (1.0 / 255) * batch
            lab_batch = rgb_to_lab_tensor(_batch)
            x_batch = lab_batch[:, :, :, 0] / 100.0
            y_batch = lab_batch[:, :, :, 1:] / 128.0
            
            x_batch = tf.expand_dims(x_batch, axis=-1)
            y_true = tf.concat([x_batch, y_batch], axis=-1)
            yield (x_batch, y_true)

    def compile(self,lr = 0.0001):
        opt = Adamax(learning_rate=lr)
        self.model.compile(optimizer=opt, loss=CustomCombinedLoss(), metrics=[psnr,ssim])
        
    def train(self,epochs = 100):
        set_global_policy('mixed_float16')
        patience = 12
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
            epochs=epochs,
            callbacks=[tb_callback, model_checkpoint, early_stop, reduce_lr]
        )

    def save_model(self,epochs=100):
        self.model.save_weights('weights_{}e_pic.weights.h5'.format(epochs))
        self.model.save('model_{}e_pic_m.keras'.format(epochs))

    def run(self,epochs=100):
        self.train(epochs)
        self.save_model(epochs)