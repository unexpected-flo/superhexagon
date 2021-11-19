import numpy as np
import tensorflow as tf
import pathlib as P
import matplotlib.pyplot as plt


train_folder = P.Path("./train_data_BW")
val_folder = P.Path("./val_data_BW")
train_set = tf.keras.preprocessing.image_dataset_from_directory(train_folder,
                                                                seed=25,
                                                                shuffle=True,
                                                                image_size=(224, 224),
                                                                label_mode="categorical")

val_set = tf.keras.preprocessing.image_dataset_from_directory(val_folder,
                                                              seed=25,
                                                              shuffle=False,
                                                              image_size=(224, 224),
                                                              label_mode="categorical")
print(train_set.class_names)
print(val_set.class_names)
preprocess_input = tf.keras.applications.resnet50.preprocess_input
rn50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=3)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = rn50(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(3)(x)
softmax = tf.keras.layers.Softmax()(x)
model = tf.keras.Model(inputs, softmax)

model.summary()

rn50.trainable = True

for layer in rn50.layers[:150]:
    layer.trainable = False
for layer in rn50.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

learning_rate = 0.00001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# for layer in rn50.layers:
#     print(layer, layer.trainable)

pretrain_loss, pretrain_accuracy = model.evaluate(val_set)
print("loss before retraining: {:.2f}".format(pretrain_loss))
print("accuracy before retraining: {:.2f}".format(pretrain_accuracy))

# folder = P.Path("./train_data")
# existing_right = len(list((folder/"right").iterdir()))
# existing_left = len(list((folder/"left").iterdir()))
# existing_nothing = len(list((folder/"None").iterdir()))
# total_trainset = existing_right + existing_left + existing_nothing
#
# class_weight = {0: existing_nothing/total_trainset,
#                 1: existing_left/total_trainset,
#                 2: existing_right/total_trainset}
epochs = 5
history = model.fit(train_set,
                    epochs=epochs,
                    validation_data=val_set)
                   # class_weight=class_weight)
save_path = P.Path("./resnet50_trained_sftmx_BW/")
save_path.mkdir(exist_ok=True)
model.save(save_path)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
