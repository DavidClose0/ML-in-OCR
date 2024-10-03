import datetime
import tensorflow as tf

%load_ext tensorboard

#remove any previous logs
!rm -rf ./logs/ 

#load and preprocess dataset
tmnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = tmnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#define model and loss function
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),       #input layer
    tf.keras.layers.Dense(15, activation='relu'),        #hidden layer 1 with 15 nodes
    tf.keras.layers.Dense(15, activation='relu'),        #hidden layer 2 with 15 nodes
    tf.keras.layers.Dense(10)                            #output layer
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

#create logs for tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                      histogram_freq=1, 
                                                      profile_batch='500,520')

model.fit(x_train, 
          y_train, 
          epochs=50, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])

%tensorboard --logdir logs/fit