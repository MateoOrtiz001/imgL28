from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, DepthwiseConv2D, Concatenate, MaxPooling2D, Dropout, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l1, l2, OrthogonalRegularizer, l1_l2
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
from modRegularizer import OrthogonalConvRegularizer
from preprocess import *
import keras
import numpy as np
import os
import tensorflow as tf
from pathlib import Path
import random


class NeuralNetwork(object):
    def __init__(self, training_path="./dataset", batch_size=16, path_to_model=None, image_size=128,seed=55):
        self.training_path = training_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.validation_split = 0.1
        
        # Configurar semilla para reproducibilidad
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Solo calcula el tamaño del training set si es necesario (entrenamiento)
        self.training_set_size = 0
        self.image_paths = []
        
        if path_to_model is None and os.path.exists(self.training_path):
            self.image_paths = self._get_image_paths()
            self.training_set_size = len(self.image_paths)
            
        if path_to_model is None:
            self.model = self.neural_network_structure()
        else:
            self.model = NeuralNetwork.load_model_from_file(path_to_model)

    def _get_image_paths(self):
        """Obtiene todas las rutas de imágenes del directorio"""
        image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
        image_paths = []
        
        for root, dirs, files in os.walk(self.training_path):
            for file in files:
                if Path(file).suffix in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        return image_paths

    def _split_dataset(self):
        """Divide el dataset en entrenamiento y validación"""
        dataset_size = len(self.image_paths)
        val_size = int(dataset_size * self.validation_split)
        
        # Mezclar las rutas de manera reproducible
        shuffled_paths = self.image_paths.copy()
        random.shuffle(shuffled_paths)
        
        train_paths = shuffled_paths[val_size:]
        val_paths = shuffled_paths[:val_size]
        
        return train_paths, val_paths

    def _load_and_preprocess_image(self, image_path):
        """Carga y preprocesa una imagen individual"""
        # Leer la imagen
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Redimensionar
        image = tf.image.resize(image, [self.image_size, self.image_size])
        
        # Normalizar a [0, 1]
        image = image / 255.0
        
        return image

    def _augment_image(self, image, training=True):
        """Aplica aumentos de datos"""
        if training:
            # Brillo
            image = tf.image.random_brightness(image, max_delta=0.2)
            image = tf.clip_by_value(image, 0.0, 1.0)
            
            # Zoom (implementado como crop y resize)
            if tf.random.uniform(()) > 0.5:
                crop_size = tf.random.uniform((), 0.8, 1.0)
                h = tf.cast(tf.cast(self.image_size, tf.float32) * crop_size, tf.int32)
                w = tf.cast(tf.cast(self.image_size, tf.float32) * crop_size, tf.int32)
                image = tf.image.random_crop(image, [h, w, 3])
                image = tf.image.resize(image, [self.image_size, self.image_size])
            
            # Rotación (aproximada con flips)
            if tf.random.uniform(()) > 0.7:
                image = tf.image.rot90(image, k=tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32))
            
            # Flip horizontal
            image = tf.image.random_flip_left_right(image)
            
        return image

    def _preprocess_for_training(self, image):
        """Convierte RGB a LAB y prepara para entrenamiento"""
        # Convertir RGB a LAB
        lab_image = rgb_to_lab_tensor(tf.expand_dims(image, 0))
        lab_image = tf.squeeze(lab_image, 0)
        
        # Separar canales L y AB
        l_channel = lab_image[:, :, 0:1] / 100.0  # Normalizar L a [0, 1]
        ab_channels = lab_image[:, :, 1:3] / 128.0  # Normalizar AB a [-1, 1]
        
        # X es el canal L, Y es la imagen completa LAB
        x = l_channel
        y = tf.concat([l_channel, ab_channels], axis=-1)
        #ab_target = ab_channels
        
        return x, y

    def create_dataset(self, subset='training'):
        """Crea un dataset tf.data para entrenamiento o validación"""
        if subset == 'training':
            train_paths, _ = self._split_dataset()
            image_paths = train_paths
            is_training = True
        else:  # validation
            _, val_paths = self._split_dataset()
            image_paths = val_paths
            is_training = False
        
        if len(image_paths) == 0:
            raise ValueError(f"No se encontraron imágenes para {subset}")
        
        print(f"Imágenes encontradas para {subset}: {len(image_paths)}")
        
        # Crear dataset desde las rutas
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        
        # Mezclar si es entrenamiento
        if is_training:
            dataset = dataset.shuffle(buffer_size=len(image_paths), seed=self.seed)
        
        # Mapear las funciones de procesamiento
        dataset = dataset.map(
            self._load_and_preprocess_image,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Aplicar aumentos de datos
        dataset = dataset.map(
            lambda img: self._augment_image(img, training=is_training),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Convertir a formato LAB y preparar para entrenamiento
        dataset = dataset.map(
            self._preprocess_for_training,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Agrupar en batches
        dataset = dataset.batch(self.batch_size)
        
        # Prefetch para mejorar rendimiento
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Repetir indefinidamente si es entrenamiento
        if is_training:
            dataset = dataset.repeat()
        
        return dataset

    def neural_network_structure(self):
        network_input = Input(shape=(None, None, 1))

        #encoder
        input_3c = Concatenate()([network_input, network_input, network_input])
        encoder = MobileNetV2(include_top=False,weights="imagenet",input_tensor=input_3c)
        encoder.trainable = True           # Habilita la posibilidad de entrenar
        for layer in encoder.layers[0:104]:       # Congela todo hasta la capa
            layer.trainable = False

        encoder_output = encoder.get_layer('block_13_expand_relu').output  #8
        # cuello de botella
                                                    
        b1 = Conv2D(192, (2, 2), activation='relu', padding='same', kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-5),name='block_b_local')(encoder_output)
        b2 = Conv2D(192, (3, 3), activation='relu', padding='same', kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-5),name='block_b_global')(encoder_output)
        b3 = Conv2D(192, (1, 1), activation='relu', padding='same', kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-5),name='block_b_pool_reduce')(encoder_output)
        b = Concatenate(name='block_b')([b1,b2,b3])
        b = DepthwiseConv2D((3,3), activation='relu', padding='same', depthwise_initializer=Orthogonal(np.sqrt(2)),
                   depthwise_regularizer=OrthogonalConvRegularizer(1e-5),name='block_b_dephtwise')(b)
        b = Conv2D(256, (1,1), activation='relu', padding='same', kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-4),name='block_b_reduce')(b)
        b = BatchNormalization()(b)
        b = residualBlock(b,256)
        b_attention = FullyConvGlobalAttention(reduction_ratio=16)(b)
        # b_guided, color_guidance  = LightweightSemanticGuidance(num_semantic_channels=16,guidance_weight=0.125, diversity_weight=0.05, smoothness_weight=0.04)(b_attention)
        b = Add()([b_attention,b])
        b = Dropout(0.1)(b)
        
        # decoder
        
        d4 = UpSampling2D(size=(2, 2),name='d_block_4_upscaling')(b)
        d4 = Conv2D(192, (3,3), padding='same', activation='relu',kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-6),name='d_block4_orth')(d4) #16
        d4 = BatchNormalization(name='d_block_4_normalize')(d4)
        d4 = Concatenate(name='d_block_4_residual')([d4,encoder.get_layer('block_6_expand_relu').output])
        d4 = Conv2D(192,(3,3), activation='relu', padding='same',name='d_block_4_depthwise')(d4)
        d4 = SpatialAttentionBlock()(d4)
        d4 = mobileBlock(d4,192)
        d4 = mobileBlock(d4,192)
        d4 = mobileBlock(d4,192)
        d4 = BatchNormalization()(d4)
        
        d3 = Conv2DTranspose(144, (3,3), strides=(2,2), padding='same',kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-6), activation='relu', name='d_block_3_upscaling')(d4)              #32                                             #32
        d3 = BatchNormalization(name='d_block_3_normalize')(d3)
        d3 = Concatenate(name='d_block_3residual')([d3,encoder.get_layer('block_3_expand_relu').output])
        d3 = Conv2D(144,(3,3), activation='relu', padding='same',name='d_block_3_depthwise')(d3)
        d3 = SpatialAttentionBlock()(d3)
        d3 = mobileBlock(d3,144)     
        d3 = mobileBlock(d3,144)  
        d3 = mobileBlock(d3,144) 
        d3 = BatchNormalization()(d3)

        d2 = Conv2DTranspose(96, (3,3), strides=(2,2), padding='same',kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-6), activation='relu', name='d_block_2_upscaling')(d3)          #64                                                 #64
        d2 = BatchNormalization(name='d_block_2_normalize')(d2)
        d2 = Concatenate(name='d_block_2_residual')([d2,encoder.get_layer('block_1_expand_relu').output])
        d2 = Conv2D(96,(3,3), activation='relu', padding='same',name='d_block_2_depthwise_1')(d2)
        d2 = SpatialAttentionBlock()(d2)
        d2 = mobileBlock(d2,96)
        d2 = mobileBlock(d2,96)
        d2 = BatchNormalization()(d2)
        
        d1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-5),name='d_block_1_upscaling')(d2)             #128
        d1 = mobileBlock(d1,32, OrthogonalConvRegularizer(1e-5))
        d1 = mobileBlock(d1,32, OrthogonalConvRegularizer(1e-5))
        d1 = Conv2D(32,(3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer=Orthogonal(np.sqrt(2)),
                    kernel_regularizer=OrthogonalConvRegularizer(1e-5),name='d_block_1_relu')(d1)
        network_output = Conv2D(2, (3, 3), activation='tanh', padding='same',name='output')(d1)
        network_output = tf.keras.layers.Activation('linear', dtype='float32')(network_output)

        return Model(inputs=network_input, outputs=network_output,name="colorizer")

    @staticmethod
    def load_model_from_file(filename, compile=False):
        return load_model(filename,compile=compile)

    def compile(self, lr=0.0001):
        opt = Adamax(learning_rate=lr)
        self.model.compile(optimizer=opt, loss=CustomCombinedLoss(), metrics=[psnr, ssim])
        
    def train(self, epochs=100):
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

        # Crear datasets
        train_dataset = self.create_dataset(subset='training')
        val_dataset = self.create_dataset(subset='validation')

        # Calcular pasos por época
        train_paths, val_paths = self._split_dataset()
        train_steps = ceil(len(train_paths) / self.batch_size)
        val_steps = ceil(len(val_paths) / self.batch_size)

        print(f"Pasos de entrenamiento por época: {train_steps}")
        print(f"Pasos de validación por época: {val_steps}")

        self.model.fit(
            train_dataset,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=[tb_callback, model_checkpoint, early_stop, reduce_lr]
        )

    def save_model(self, epochs=100):
        self.model.save_weights('weights_{}e_pic.weights.h5'.format(epochs))
        self.model.save('model_{}e_pic_m.keras'.format(epochs))

    def run(self, epochs=100):
        self.train(epochs)
        self.save_model(epochs)

    # Método adicional para obtener un sample del dataset (útil para debugging)
    def get_sample_batch(self, subset='training'):
        """Obtiene un batch de muestra para inspección"""
        dataset = self.create_dataset(subset=subset)
        for batch in dataset.take(1):
            return batch
        return None