from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, LeakyReLU, concatenate, MaxPooling2D, SpatialDropout2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1, l2, OrthogonalRegularizer, l1_l2
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from tensorflow.keras.initializers import Orthogonal, HeNormal
from math import ceil
from modLayers import *
from modMetrics import *
import keras
import numpy as np
import os
import tensorflow as tf


class NeuralNetwork(object):
    def __init__(self, training_path="./dataset", epochs=50, batch_size=16, path_to_Genmodel=None, path_to_Dismodel=None, image_size=128):
        self.training_path = training_path
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size

        # Solo calcula el tamaño del training set si es necesario (entrenamiento)
        self.training_set_size = 0
        if path_to_Genmodel is None and os.path.exists(self.training_path):
            for filename in os.listdir(self.training_path):
                if filename.endswith((".png", ".jpg", ".jpeg")):
                    self.training_set_size += 1
                    
        self.datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, rotation_range=20, horizontal_flip=True,validation_split=0.2)
        
        if path_to_Genmodel is None:
#            self.model = self.genNetwork()
            self.generator = self.genNetwork()  # Tu autoencoder (generador)
            self.discriminator = self.disNetwork(image_shape=(image_size, image_size, 3))
            self.gan = self.build_gan(self.generator, self.discriminator)
        else:
            self.generator = NeuralNetwork.load_model_from_file(path_to_Genmodel)
            self.discriminator = NeuralNetwork.load_model_from_file(path_to_Dismodel)
            

    def genNetwork(self):
        network_input = Input(shape=(None, None, 1,))

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
        b = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_initializer=Orthogonal())(b)
        b = Conv2D(256, (2, 2), activation='relu', padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.005), 
                   kernel_initializer=HeNormal())(b)
        b = SpatialDropout2D(0.1)(b)

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
        
        d2 = Conv2DTranspose(32, (3, 3), strides=(2,2), padding='same', activation='relu')(d3)   #64
        d2 = BatchNormalization()(d2)
        d2 = concatenate([d2,e2])
        d2 = Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(d2)
        d2 = spatialAttention(d2)
        d1 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(d2)  #128
        network_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(d1)

        return Model(inputs=network_input, outputs=network_output)

    def disNetwork(image_shape=(128, 128, 3)):
        """Discriminador PatchGAN (70x70)"""
        input_image = Input(shape=image_shape)  # Imagen real
        generated_image = Input(shape=image_shape)  # Imagen generada por el autoencoder
        combined = concatenate()([input_image, generated_image])
        
        d = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(combined)
        d = LeakyReLU(alpha=0.2)(d)
        
        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        d = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        d = Conv2D(512, (4, 4), strides=(1, 1), padding='same')(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Conv2D(1, (4, 4), strides=(1, 1), padding='same', activation='sigmoid')(d)
        
        return Model(inputs=[input_image, generated_image], outputs=d)


    def build_gan(self,image_shape=(128, 128, 1)):
        self.discriminator.trainable = False
        input_gray = Input(shape=image_shape)
        generated_color = self.generator(input_gray)
        validity = self.discriminator([input_gray, generated_color])
        combined = Model(inputs=input_gray, outputs=[validity, generated_color])
        return combined
    
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

    def compile_models(self):
        # Optimizadores
        opt_d = Adamax(learning_rate=0.0002, beta_1=0.5)
        opt_g = Adamax(learning_rate=0.0001)
        
        # Compilar discriminador
        self.discriminator.compile(
            optimizer=opt_d,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Congelar discriminador durante el entrenamiento GAN
        self.discriminator.trainable = False
        
        # Compilar GAN 
        self.gan.compile(
            optimizer=opt_g,
            loss=['binary_crossentropy', 'mse'],  # Pérdida adversarial + L1/L2
            loss_weights=[1, 100],  # Peso para adversarial vs. MSE
            metrics=[psnr, ssim]
        )

    def train(self):
        self.compile_models()
        
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

        # Crear generadores
        train_generator = self.image_gen(subset='training')
        val_generator = self.image_gen(subset='validation')

        # Preprocesar los generadores
        train_generator_preprocessed = self.preprocess_generator(train_generator)
        val_generator_preprocessed = self.preprocess_generator(val_generator)


        # Calcular pasos por época
        train_steps = ceil(train_generator.samples / self.batch_size)
        val_steps = ceil(val_generator.samples / self.batch_size)

        real_labels = np.ones((self.batch_size, 16, 16, 1))
        fake_labels = np.zeros((self.batch_size, 16, 16, 1))
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            
            for _ in range(train_steps):
                x_batch, y_batch = next(train_generator_preprocessed)
                generated_ab = self.generator.predict(x_batch)
                lab_real = np.concatenate([x_batch, y_batch], axis=-1)
                lab_fake = np.concatenate([x_batch, generated_ab], axis=-1)
                
                d_loss_real = self.discriminator.train_on_batch(lab_real, real_labels)
                d_loss_fake = self.discriminator.train_on_batch(lab_fake, fake_labels)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            g_loss = self.gan.fit(
                train_generator_preprocessed,
                steps_per_epoch=train_steps,
                validation_data=val_generator_preprocessed,
                validation_steps=val_steps,
                epochs=1,
                callbacks=[tb_callback, model_checkpoint, early_stop, reduce_lr],
                verbose=1
            ).history['loss'][0]
            
            print(f"D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")

    def save_model(self):
        self.generator.save_weights('generator_weights_{}e_pic.weights.h5'.format(self.epochs))
        self.generator.save('generator_model_{}e_pic_m.keras'.format(self.epochs))
        self.discriminator.save_weights('discriminator_weights_{}e_pic.weights.h5'.format(self.epochs))
        self.discriminator.save('discriminator_model_{}e_pic_m.keras'.format(self.epochs))

    def run(self):
        self.train()
        self.save_model()
