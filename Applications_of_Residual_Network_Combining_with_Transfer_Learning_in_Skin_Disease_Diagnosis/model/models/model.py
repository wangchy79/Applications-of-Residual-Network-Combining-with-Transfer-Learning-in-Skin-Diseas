from tensorflow import keras
import tensorflow_addons as tfa
import math
from config import NUM_CLASSES,image_height,image_width,channels

def ResNet152V2_Rahman():
    
    # DEFINING MODEL LAYERS
    # ---------------------------
    # Load pre trained model without last FC layer
    base_model = keras.applications.ResNet152V2(include_top=False,
                                                weights="imagenet",
                                                input_shape=(image_height,image_width,channels))
    
    # Freeze all pre trained layers
    base_model.trainable = False
    
    # Define output layers (Rahman and Ami)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    predictions = keras.layers.Dense(units=NUM_CLASSES, activation="softmax")(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name="ResNet152V2_Rahman") 
    
    # COMPILING THE MODEL
    # ---------------------------
    lr = 0.001 # 10x all other layers
    #lr_schedule = [0.001, 0.0001, 0.00001] # drop by 10x factor at epoch 5 and 10

    optimiser = keras.optimizers.Adam(learning_rate=lr)
    loss_func = keras.losses.CategoricalCrossentropy()
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 
    
    model.compile(optimizer=optimiser ,
                loss=loss_func ,
                metrics=metrics_list 
                )
    
    return model

def ResNet50_Mahbod():
    """
    Creates a Keras model and from a base pre-trained model and newly defined output layers.
    Compiles the model with defined optimizer, loss and metrics.
    
    Returns: compiled Keras model ready for training
    """
        
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(image_height,image_width,channels))
    
    #base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    base_model.trainable = False # Blocks 1-17 Frozen as in Mahbod et al.
    
    
    # Define output layers (Mahbod et al. used here)
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    predictions = keras.layers.Dense(units=NUM_CLASSES, activation="softmax")(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name="ResNet50_Mahbod") 
    
    
    
    # UNFREEZE 17TH BLOCK
    # -------------------------------------
    # Create dictionary of layer name and whether the layer is trainable 
    trainable_dict = dict([ (layer.name, layer.trainable) for layer in model.layers ])
    
    # Identify names of layers in 17th block
    block_17_names = []
    for name in [ layer.name for layer in model.layers ]: # iterate through model layer names
        if "conv5_block3" in name: # conv5_block3 is naming schemee for 17th block
            block_17_names.append(name)
            
    # Set these layers to be trainable
    for name in block_17_names:
        trainable_dict[name] = True  # change dict values to true     
    
    for layer_name, trainable_bool in trainable_dict.items():
        layer = model.get_layer(name=layer_name)
        layer.trainable = trainable_bool
    

    # OPTIMIZERS
    # -------------------------------------
    # Different LR for pretrained and FC layers
    pretrained_lr = 0.0001 
    new_lr = 10 * pretrained_lr 
    
    # Create multioptimizer
    optimizers = [keras.optimizers.Adam(learning_rate=pretrained_lr),
                  keras.optimizers.Adam(learning_rate=new_lr)]
    
    # Layer objects for pre-trained and FC layers
    block_17_layers = [ model.get_layer(name=name) for name in block_17_names ]
    new_fc_layers = model.layers[-3:]
    
    #       Create LR multiplier dict arguemnt
    #       LR_mult_dict = {}
    #       for layer in block_17_layers:
#               LR_mult_dict[layer.name] = 1.0
    #       for layer in new_fc_layers:
    #           LR_mult_dict[layer.name] = 10.0
    
    # (Optimizer, layer) pairs 
    block_17_optimizers_and_layers = [  (optimizers[0],layer) for layer in block_17_layers ]
    new_fc_optimizers_and_layers = [  (optimizers[1],layer) for layer in new_fc_layers ]
    optimizers_and_layers = block_17_optimizers_and_layers + new_fc_optimizers_and_layers
    
    # Optimizer with different learning rates across layers
    optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    #optimizer = LRMultiplier( keras.optimizers.Adam(learing_rate=pretrained_lr), LR_mult_dict )
    
    # LOSS FUNCTION AND METRICS
    # -------------------------------------
    loss_func = keras.losses.CategoricalCrossentropy()
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 
    
    
    # COMPILE EVERYTHING
    # -------------------------------------
    model.compile(optimizer=optimizer ,
                loss=loss_func ,
                metrics=metrics_list)
    
    return model

def ResNet50_Hosseinzadeh():
    
    # DEFINING MODEL LAYERS
    # ---------------------------
    base_model = keras.applications.resnet50.ResNet50(include_top=False,
                                                      weights="imagenet",
                                                      input_shape=(image_height,image_width,channels))
    
    base_model.trainable = False # Blocks 1-17 Frozen as in Hosseinzadeh et al.

    # Define output layers 
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    x = keras.layers.Dense(units=64, activation="relu")(x)
    x = keras.layers.Dropout(rate=0.5)(x)
    predictions = keras.layers.Dense(units=NUM_CLASSES, activation="softmax")(x)

    # Create model using forzen base layers and new FC layers
    model = keras.models.Model(inputs=base_model.input, 
                               outputs=predictions, 
                               name="ResNet50_Hosseinzadeh") 

    #pprint( [ layer.trainable for layer in model.layers ] )

    # COMPILING THE MODEL
    # ---------------------------
    lr = math.e # 10x all other layers
    #lr_schedule = [0.001, 0.0001, 0.00001] # drop by 10x factor at epoch 5 and 10

    optimiser = keras.optimizers.Adam(learning_rate=lr)
    loss_func = keras.losses.CategoricalCrossentropy()
    metrics_list = ['accuracy',
                    keras.metrics.AUC( multi_label=True )] 

    model.compile(optimizer=optimiser ,
                loss=loss_func ,
                metrics=metrics_list)  
                

    return model