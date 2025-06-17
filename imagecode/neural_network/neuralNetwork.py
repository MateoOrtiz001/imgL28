from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, LeakyReLU, concatenate, MaxPooling2D, Dropout, SpatialDropout2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1, l2, OrthogonalRegularizer, l1_l2
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from tensorflow.keras.initializers import Orthogonal, HeNormal
from math import ceil
import time
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
            self.generator = self.genNetwork()
            self.discriminator = self.disNetwork()
            self.gan = self.build_gan()
        else:
            self.generator = NeuralNetwork.load_model_from_file(path_to_Genmodel)
            self.discriminator = NeuralNetwork.load_model_from_file(path_to_Dismodel)
            

    def genNetwork(self):
        network_input = Input(shape=(None, None, 1,))

        #encoder

        e1 = Conv2D(16, (3, 3), activation='relu', padding='same')(network_input)   #128
        e1 = residualBlock(e1,16)
        e1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.0005))(e1)
        
        e2 = MaxPooling2D((2, 2))(e1)                                               #64
        e2 = BatchNormalization()(e2)
        e2 = Conv2D(32, (3, 3), activation='relu', padding='same')(e2)
        e2 = residualBlockCB(e2, 32)
        e2 = Conv2D(32, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.0005))(e2)
        
        e3 = MaxPooling2D((2, 2))(e2)                                               #32
        e3 = BatchNormalization()(e3)
        e3 = Conv2D(64, (3, 3), activation='relu', padding='same')(e3)
        e3 = residualBlockCB(e3, 64)
        e3 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(e3)
        
        e4 = MaxPooling2D((2, 2))(e3)                                               #16
        e4 = BatchNormalization()(e4)
        e4 = Conv2D(128, (3, 3), activation='relu', padding='same')(e4)
        e4 = residualBlockCB(e4,128)
        e4 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(e4)
        
        e5 = MaxPooling2D((2,2))(e4)                                                #8
        e5 = BatchNormalization()(e5)
        e5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.005))(e5)
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
        d5 = Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.0005))(d5)
        
        d4 = UpSampling2D((2, 2))(d5)                                                            #16
        d4 = BatchNormalization()(d4)
        d4 = concatenate([d4,e4])
        d4 = Conv2D(128, (3, 3), activation='relu', padding='same')(d4)
        d4 = spatialAttention(d4)
        d4 = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(d4)

        d3 = UpSampling2D((2, 2))(d4)                                                           #32
        d3 = BatchNormalization()(d3)
        d3 = concatenate([d3,e3])
        d3 = Conv2D(64, (3, 3), activation='relu', padding='same')(d3)
        d3 = spatialAttention(d3)
        d3 = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(0.001))(d3)
        
        d2 = Conv2DTranspose(32, (3, 3), strides=(2,2), padding='same', activation='relu')(d3)   #64
        d2 = BatchNormalization()(d2)
        d2 = concatenate([d2,e2])
        d2 = Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2(0.001))(d2)
        d2 = spatialAttention(d2)
        d1 = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same', activation='relu')(d2)  #128
        network_output = Conv2D(2, (3, 3), activation='tanh', padding='same', name='colOutput')(d1)

        return Model(inputs=network_input, outputs=network_output, name="colorizer")

    def disNetwork(self):
        """Discriminador PatchGAN (70x70)"""
        input = Input(shape=(self.image_size, self.image_size, 3))      #(256)
        
        d = Conv2D(32, (4, 4), strides=(2, 2), padding='same')(input)   #128
        d = LeakyReLU(alpha=0.2)(d)
        
        d = Conv2D(56, (4, 4), strides=(2, 2), padding='same')(d)      #64
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        
        d = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(d)      #32
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = Dropout(0.5)(d)
        
        d = Conv2D(1, (4, 4), strides=(1, 1), padding='same', activation='sigmoid', kernel_regularizer=l2(0.0005), name="disOutput")(d)
        
        return Model(inputs=input, outputs=d, name="discriminator")


    def build_gan(self):
        self.discriminator.trainable = False
        input_gray = Input(shape=(self.image_size, self.image_size,1))
        generated_color = self.generator(input_gray)
        validity = self.discriminator(concatenate([input_gray, generated_color], axis=-1))
        combined = Model(inputs=input_gray, outputs=[validity, generated_color],name="gan")
        combined.output_names = ["validity_output", "color_output"]
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
        def gen():
            for batch in generator:
                _batch = (1.0 / 255) * batch
                lab_batch = rgb2lab(_batch)
                x_batch = lab_batch[:, :, :, 0] / 100.0
                x_batch = x_batch[:, :, :, None]
                y_batch = lab_batch[:, :, :, 1:] / 128.0
                current_batch_size = x_batch.shape[0]
                validity_labels = np.ones((current_batch_size, self.image_size//8, self.image_size//8, 1))
                yield (x_batch, (validity_labels, y_batch))  # Cambiar lista a tupla

        output_signature = (
            tf.TensorSpec(shape=(None, self.image_size, self.image_size, 1), dtype=tf.float32),
            (
                tf.TensorSpec(shape=(None, self.image_size//8, self.image_size//8, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(None, self.image_size, self.image_size, 2), dtype=tf.float32)
            )
        )
        
        return tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def compile_models(self):
        # Optimizadores
        opt_d = Adam(learning_rate=0.00005, beta_1=0.5)
        opt_g = Adam(learning_rate=0.0002)
        
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
            loss=['binary_crossentropy', 'mae'],  # Pérdida adversarial + L1/L2
            loss_weights=[1, 50],  # Peso para adversarial vs. MSE
            metrics={
                'validity_output': ['accuracy'],
                'color_output': [psnr, ssim]
            }
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
        
        train_generator = self.image_gen(subset='training')
        val_generator = self.image_gen(subset='validation')
        
        train_steps = ceil(train_generator.samples / self.batch_size)
        val_steps = ceil(val_generator.samples / self.batch_size)
        print(f"Train steps: {train_steps}, Val steps: {val_steps}")
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            epoch_start_time = time.time()  
            
            train_generator_preprocessed = self.preprocess_generator(train_generator)
            iterator = iter(train_generator_preprocessed)
            
            # Inicializar el último d_loss
            last_d_loss = [0.0, 0.5]  # Pérdida inicial y accuracy razonable
            t = 0
            for step in range(train_steps):
                step_start_time = time.time()
                    
                try:
                    x_batch, (validity_labels, y_batch) = next(iterator)
                except StopIteration:
                    iterator = iter(self.preprocess_generator(train_generator))
                    x_batch, (validity_labels, y_batch) = next(iterator)
                
                current_batch_size = x_batch.shape[0]
                #real_labels = np.ones((current_batch_size, 16, 16, 1)) * 0.9
                #fake_labels = np.zeros((current_batch_size, 16, 16, 1)) + 0.1
                real_labels = np.ones((current_batch_size, self.image_size//8, self.image_size//8, 1))
                fake_labels = np.zeros((current_batch_size, self.image_size//8, self.image_size//8, 1))
                #real_labels = np.ones((current_batch_size, 16, 16, 1)) - np.random.uniform(0, 0.1, size=(current_batch_size, 16, 16, 1))  # 0.9–1.0
                #fake_labels = np.random.uniform(0, 0.1, size=(current_batch_size, 16, 16, 1))  # 0.0–0.1
                
                # Generar imágenes falsas
                generated_ab = self.generator.predict(x_batch, verbose=0)
                lab_real = np.concatenate([x_batch, y_batch], axis=-1)
                lab_fake = np.concatenate([x_batch, generated_ab], axis=-1)
                noise_std = 0.03  # puedes ajustar
                lab_fake += np.random.normal(loc=0.0, scale=noise_std, size=lab_fake.shape)
                
                # Decidir si entrenar el discriminador basado en el accuracy de la iteración anterior
                if (last_d_loss[1] <= 0.75) and (t == 0):  # Umbral de pausa
                    self.discriminator.trainable = True
                    d_loss_real = self.discriminator.train_on_batch(lab_real, real_labels)
                    d_loss_fake = self.discriminator.train_on_batch(lab_fake, fake_labels)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    last_d_loss = d_loss  # Actualizar el último d_loss
                elif t > 1:
                    t -= 1
                    d_loss = last_d_loss
                elif (last_d_loss[1] > 0.75) and (t == 0):
                    print(f"Skipping discriminator training at step {step} (D Acc: {last_d_loss[1]:.4f})")
                    t = train_steps//4
                    d_loss = last_d_loss  # Mantener el último d_loss
                elif (last_d_loss[1] > 0.75) and (t == 1):
                    print(f"End of skipping discriminator training")
                    self.discriminator.trainable = True
                    d_loss_real = self.discriminator.train_on_batch(lab_real, real_labels)
                    d_loss_fake = self.discriminator.train_on_batch(lab_fake, fake_labels)
                    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                    last_d_loss = d_loss  # Actualizar el último d_loss   
                    t -= 1 
                
                # Evaluar el accuracy del discriminador sin entrenarlo
                # self.discriminator.trainable = False
                # d_loss_real_eval = self.discriminator.evaluate(lab_real, real_labels, verbose=0, batch_size=current_batch_size)
                # d_loss_fake_eval = self.discriminator.evaluate(lab_fake, fake_labels, verbose=0, batch_size=current_batch_size)
                # d_loss_eval = [0.5 * (d_loss_real_eval[0] + d_loss_fake_eval[0]),
                #             0.5 * (d_loss_real_eval[1] + d_loss_fake_eval[1])]
                
                # # Decidir si entrenar el discriminador
                # if d_loss_eval[1] <= 0.75:  # Umbral de pausa
                #     self.discriminator.trainable = True
                #     d_loss_real = self.discriminator.train_on_batch(lab_real, real_labels)
                #     d_loss_fake = self.discriminator.train_on_batch(lab_fake, fake_labels)
                #     d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                #     last_d_loss = d_loss  # Actualizar el último d_loss
                # else:
                #     print(f"Skipping discriminator training at step {step} (D Acc: {d_loss_eval[1]:.4f})")
                #     d_loss = last_d_loss  # Usar el último d_loss válido
                
                # if d_loss[1] > 0.75:  # Umbral de pausa
                #     print(f"Skipping discriminator training at step {step} (D Acc: {d_loss[1]:.4f})")
                #     self.discriminator.trainable = False
                #     # No actualizar d_loss, mantener el valor de train_on_batch
                # else:
                #     # El discriminador ya fue entrenado, usar d_loss directamente
                #     pass
                
                
                # Entrenar generador (a través de la GAN)
                self.discriminator.trainable = False
                for _ in range(2):
                    g_loss = self.gan.train_on_batch(x_batch, [validity_labels, y_batch])
                
                step_time = time.time() - step_start_time
                
                if step % 100 == 0:
                    print(f"Step {step}/{train_steps} - "
                        f"D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss[0]:.4f}, "
                        f"Step Time: {step_time:.2f}s")
            
            # Validación al final de la época
            val_generator_preprocessed = self.preprocess_generator(val_generator)
            val_metrics = self.gan.evaluate(val_generator_preprocessed, steps=val_steps, verbose=0, return_dict=True)
            
            epoch_time = time.time() - epoch_start_time
            
            # Acceder a las métricas con los nombres correctos
            try:
                # Intentar acceder con los nombres de las salidas
                psnr_val = val_metrics.get('color_output_psnr', val_metrics.get('psnr', 0))
                ssim_val = val_metrics.get('color_output_ssim', val_metrics.get('ssim', 0))
                
                print(f"Validation Loss: {val_metrics['loss']:.4f}, "
                    f"PSNR: {psnr_val:.4f}, SSIM: {ssim_val:.4f}, "
                    f"Epoch Time: {int(epoch_time // 60)}m {epoch_time % 60:.2f}s")
            except KeyError as e:
                # Si hay problemas con las métricas, mostrar todas las claves disponibles
                print(f"Available metrics keys: {list(val_metrics.keys())}")
                print(f"Validation Loss: {val_metrics['loss']:.4f}, "
                    f"Epoch Time: {int(epoch_time // 60)}m {epoch_time % 60:.2f}s")
            
            # Ejecutar callbacks manualmente
            self.gan.fit(
                self.preprocess_generator(train_generator),
                steps_per_epoch=1,
                validation_data=val_generator_preprocessed,
                validation_steps=val_steps,
                epochs=1,
                callbacks=[tb_callback, model_checkpoint, early_stop, reduce_lr],
                verbose=0
            )
        
        self.save_model()

    def save_model(self):
        self.generator.save_weights('generator_weights_{}e_pic.weights.h5'.format(self.epochs))
        self.generator.save('generator_model_{}e_pic_m.keras'.format(self.epochs))
        self.discriminator.save_weights('discriminator_weights_{}e_pic.weights.h5'.format(self.epochs))
        self.discriminator.save('discriminator_model_{}e_pic_m.keras'.format(self.epochs))

    def run(self):
        self.train()
        self.save_model()

    def debug_metrics(self):
        """Función para debuggear los nombres de las métricas"""
        self.compile_models()
        print("Modelo GAN compilado:")
        print(f"Nombres de las salidas: {[output.name for output in self.gan.outputs]}")
        print(f"Nombres de las métricas: {self.gan.metrics_names}")
        
        # Crear un batch pequeño para probar
        dummy_input = np.random.random((1, self.image_size, self.image_size, 1))
        dummy_validity = np.ones((1, self.image_size//8, self.image_size//8, 1))
        dummy_color = np.random.random((1, self.image_size, self.image_size, 2))
        
        # Evaluar con datos dummy
        result = self.gan.evaluate(dummy_input, [dummy_validity, dummy_color], verbose=0, return_dict=True)
        print(f"Claves de métricas disponibles: {list(result.keys())}")
        return result