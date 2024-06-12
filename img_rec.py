import os
import cv2
import imghdr
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import tensorboard as tb
from tensorflow.keras.callbacks import TensorBoard


# Directory containing the images
folder_paths = [
    "C:/Users/Ravi/Downloads/New projects/img classification/data_set/adult",
    "C:/Users/Ravi/Downloads/New projects/img classification/data_set/safe",
    "C:/Users/Ravi/Downloads/New projects/img classification/data_set/violent"
]

# Filter out invalid images
def filter_invalid_images(folder_paths):
    for folder_path in folder_paths:
        for img_class in os.listdir(folder_path):
            class_path = os.path.join(folder_path, img_class)
            if os.path.isdir(class_path):
                for image in os.listdir(class_path):
                    image_path = os.path.join(class_path, image)
                    try:
                        img = cv2.imread(image_path)
                        tip = imghdr.what(image_path)
                        if tip not in ['jpeg', 'jpg', 'bmp', 'png']:
                            print(f'Image not in extension list: {image_path}')
                            os.remove(image_path)
                    except Exception as e:
                        print(f'Issue with image {image_path}: {e}')

filter_invalid_images(folder_paths)

data_dir = 'C:/Users/Ravi/Downloads/New projects/img classification/data_set'

# Load the dataset
data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(256, 256),
    batch_size=32,
    label_mode='categorical'  # Assuming categorical labels
)

total_samples = sum(1 for _ in data.unbatch())

# Split the data
train_size = int(0.7 * len(data))
val_size = int(0.2 * len(data))
test_size = total_samples - train_size - val_size

print(len(train_data))
print(len(val_data))
print(len(test_data))

train_data = data.take(train_size)
val_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size + val_size).take(test_size)

# Normalize the data
train_data = train_data.map(lambda x, y: (x / 255, y))
val_data = val_data.map(lambda x, y: (x / 255, y))
test_data = test_data.map(lambda x, y: (x / 255, y))

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # Output layer with 3 neurons (one for each class)
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

history = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[tensorboard_callback])

try:
    history = model.fit(train_data, epochs=5, validation_data=val_data, callbacks=[tensorboard_callback])
except Exception as e:
    print(f"Error during training: {e}")
    
# Plot the training history
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(test_data)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Test with a sample image
img = cv2.imread('C:/Users/Ravi/Downloads/New projects/img classification/data_set/safe/anime images_15.jpeg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

# Assuming yhat is a probability prediction array
predicted_class = np.argmax(yhat)
if predicted_class == 0:
    print(f'Predicted class is adult')
elif predicted_class == 1:
    print(f'Predicted class is safe')
else:
    print(f'Predicted class is viol')
    
# Save and load the model
model.save(os.path.join('C:\\Users\\Ravi\\Downloads\\New projects\\img classification\\code_dep', 'imageclassifier.h5'))

new_model = tf.keras.models.load_model(os.path.join('C:\\Users\\Ravi\\Downloads\\New projects\\img classification\\code_dep', 'imageclassifier.h5'))
new_model.predict(np.expand_dims(resize/255, 0))
   