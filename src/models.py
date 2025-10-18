"""
Model architectures for blood cell classification.
This module contains MobileNetV3Small baseline and CBAM-enhanced versions.
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, BatchNormalization, Dense, Dropout,
    Conv2D, Multiply, Add, Activation, Reshape, GlobalMaxPooling2D,
    Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import keras.backend as K


class CBAM:
    """
    Convolutional Block Attention Module (CBAM)
    
    CBAM applies both channel attention and spatial attention mechanisms
    to enhance the representational power of CNNs.
    """
    
    @staticmethod
    def channel_attention(input_feature, ratio=8, name=""):
        """
        Channel Attention Module
        
        Args:
            input_feature: Input feature tensor
            ratio: Reduction ratio for channel attention
            name: Name prefix for layers
            
        Returns:
            Channel attention weighted features
        """
        channel = input_feature.shape[-1]
        
        # Shared MLP layers
        shared_dense_one = Dense(channel // ratio,
                               activation='relu',
                               kernel_initializer='he_normal',
                               use_bias=True,
                               bias_initializer='zeros',
                               name=f'{name}_channel_shared_dense_one')
        
        shared_dense_two = Dense(channel,
                               kernel_initializer='he_normal',
                               use_bias=True,
                               bias_initializer='zeros',
                               name=f'{name}_channel_shared_dense_two')
        
        # Global Average Pooling
        avg_pool = GlobalAveragePooling2D(name=f'{name}_channel_avg_pool')(input_feature)
        avg_pool = Reshape((1, 1, channel), name=f'{name}_channel_avg_reshape')(avg_pool)
        avg_pool = shared_dense_one(avg_pool)
        avg_pool = shared_dense_two(avg_pool)
        
        # Global Max Pooling
        max_pool = GlobalMaxPooling2D(name=f'{name}_channel_max_pool')(input_feature)
        max_pool = Reshape((1, 1, channel), name=f'{name}_channel_max_reshape')(max_pool)
        max_pool = shared_dense_one(max_pool)
        max_pool = shared_dense_two(max_pool)
        
        # Combine and apply sigmoid
        cbam_feature = Add(name=f'{name}_channel_add')([avg_pool, max_pool])
        cbam_feature = Activation('sigmoid', name=f'{name}_channel_sigmoid')(cbam_feature)
        
        # Apply attention
        return Multiply(name=f'{name}_channel_multiply')([input_feature, cbam_feature])
    
    @staticmethod
    def spatial_attention(input_feature, kernel_size=7, name=""):
        """
        Spatial Attention Module
        
        Args:
            input_feature: Input feature tensor
            kernel_size: Kernel size for spatial convolution
            name: Name prefix for layers
            
        Returns:
            Spatial attention weighted features
        """
        # Channel-wise max and average pooling
        avg_pool = tf.reduce_mean(input_feature, axis=3, keepdims=True, name=f'{name}_spatial_avg')
        max_pool = tf.reduce_max(input_feature, axis=3, keepdims=True, name=f'{name}_spatial_max')
        
        # Concatenate
        concat = Concatenate(axis=3, name=f'{name}_spatial_concat')([avg_pool, max_pool])
        
        # Convolution and sigmoid
        cbam_feature = Conv2D(filters=1,
                             kernel_size=kernel_size,
                             strides=1,
                             padding='same',
                             activation='sigmoid',
                             kernel_initializer='he_normal',
                             use_bias=False,
                             name=f'{name}_spatial_conv')(concat)
        
        # Apply attention
        return Multiply(name=f'{name}_spatial_multiply')([input_feature, cbam_feature])
    
    @classmethod
    def cbam_block(cls, cbam_feature, ratio=8, kernel_size=7, name=""):
        """
        Complete CBAM block with both channel and spatial attention
        
        Args:
            cbam_feature: Input feature tensor
            ratio: Channel attention reduction ratio
            kernel_size: Spatial attention kernel size
            name: Name prefix for layers
            
        Returns:
            CBAM enhanced features
        """
        # Apply channel attention
        cbam_feature = cls.channel_attention(cbam_feature, ratio, name=f'{name}_ca')
        
        # Apply spatial attention
        cbam_feature = cls.spatial_attention(cbam_feature, kernel_size, name=f'{name}_sa')
        
        return cbam_feature


def create_mobilenetv3_baseline(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.3):
    """
    Create MobileNetV3Small baseline model

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV3Small
    base_model = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_final')(x)
    
    # Optional: Add dense layer for more capacity
    # x = Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense_128')(x)
    # x = BatchNormalization(name='bn_dense')(x)
    # x = Dropout(dropout_rate, name='dropout_dense')(x)
    
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='MobileNetV3Small_Baseline')

    return model


def create_mobilenetv3_cbam(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.3):
    """
    Create MobileNetV3Small with CBAM attention modules

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV3Small without top
    base_model = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Get intermediate layers for CBAM insertion
    # We'll add CBAM after certain bottleneck blocks
    x = base_model.input
    
    # Process through base model layers with CBAM insertion at strategic points
    layer_names_for_cbam = [
        'multiply_1',   # After first bottleneck block
        'multiply_4',   # After expansion phase
        'multiply_7',   # Mid-level features
        'multiply_10'   # High-level features
    ]
    
    # Create a new model with CBAM modules
    cbam_outputs = []
    layer_dict = {layer.name: layer for layer in base_model.layers}
    
    for i, layer in enumerate(base_model.layers):
        if i == 0:
            # Input layer
            x = layer.output
        else:
            # Get the layer's input
            if len(layer.input_shape) > 1:
                # Handle layers with multiple inputs
                inputs = []
                for input_layer in layer._inbound_nodes[0].inbound_layers:
                    if input_layer.name in [l.name for l in base_model.layers[:i]]:
                        inputs.append(x)
                    else:
                        inputs.append(input_layer.output)
                if len(inputs) == 1:
                    x = layer(inputs[0])
                else:
                    x = layer(inputs)
            else:
                x = layer(x)
            
            # Add CBAM after specific layers
            if layer.name in layer_names_for_cbam:
                x = CBAM.cbam_block(x, ratio=8, kernel_size=7, name=f'cbam_{layer.name}')
    
    # Custom classification head with CBAM
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_final')(x)
    
    # Add a final CBAM-like attention for the flattened features
    # This is a simplified version for 1D features
    x = Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense_128')(x)
    x = BatchNormalization(name='bn_dense')(x)
    x = Dropout(dropout_rate, name='dropout_dense')(x)
    
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='MobileNetV3Small_CBAM')

    return model


def create_simple_mobilenetv3_cbam(input_shape=(224, 224, 3), num_classes=4, dropout_rate=0.3):
    """
    Create a simpler MobileNetV3Small with CBAM - adds CBAM only at the end

    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Load pre-trained MobileNetV3Small
    base_model = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    
    # Add CBAM before global pooling
    x = base_model.output
    x = CBAM.cbam_block(x, ratio=8, kernel_size=7, name='final_cbam')
    
    # Classification head
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_final')(x)
    x = Dense(128, activation='relu', kernel_initializer='he_uniform', name='dense_128')(x)
    x = BatchNormalization(name='bn_dense')(x)
    x = Dropout(dropout_rate, name='dropout_dense')(x)
    
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name='MobileNetV3Small_Small_Simple_CBAM')

    return model


def compile_model(model, initial_learning_rate=0.0001, decay_steps=40, decay_rate=0.96):
    """
    Compile the model with optimizer and loss function
    
    Args:
        model: Keras model to compile
        initial_learning_rate: Initial learning rate
        decay_steps: Steps for learning rate decay
        decay_rate: Learning rate decay rate
        
    Returns:
        Compiled model
    """
    # Use a simple fixed learning rate to avoid ExponentialDecay issues
    # We'll use ReduceLROnPlateau callback instead for learning rate scheduling
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=initial_learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        metrics=['accuracy']
    )
    
    return model


def create_lr_schedule(initial_learning_rate=1e-4, decay_steps=40, decay_rate=0.96):
    """
    Create learning rate schedule function that can be used with LearningRateScheduler callback
    
    Args:
        initial_learning_rate: Initial learning rate
        decay_steps: Steps for learning rate decay
        decay_rate: Learning rate decay rate
        
    Returns:
        Learning rate schedule function
    """
    def schedule(epoch):
        return initial_learning_rate * (decay_rate ** (epoch // decay_steps))
    
    return schedule