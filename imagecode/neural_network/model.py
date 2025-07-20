import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input, DepthwiseConv2D, Concatenate, Dropout, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.applications import MobileNetV2
from modLayers import *
from modRegularizer import OrthogonalConvRegularizer

class Colorizer(object):
    def __init__(self, path_to_model = None):
        self.path_to_model = path_to_model
        if self.path_to_model != None:
            self.model = self.loadModel(self.path_to_model)
        else:
            self.model = self.createModel()
    
    def createModel(self):
        input = Input(shape=(None,None,1))
        input_3c = Concatenate()([input, input, input])
        encoder = MobileNetV2(include_top=False,weights="imagenet",input_tensor=input_3c)
        encoder_output = encoder.get_layer('block_13_expand_relu').output  #8                       
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
        b = Add()([b_attention,b])
        b = Dropout(0.1)(b)
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
        
        return Model(inputs=Input, outputs=network_output,name="colorizer")
     
    @staticmethod   
    def loadModel(filename):
        return load_model(filename)