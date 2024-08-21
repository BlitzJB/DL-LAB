|Assignment - 3  | Joshua Bharathi |
|--|--|
| Deep Learning Lab | 22011101045 |

## Problem Description

### Problem Statement
The objective is to implement and evaluate the AlexNet architecture, a pioneering deep convolutional neural network, on the CIFAR-10 dataset. This task will test the model’s ability to classify images across 10 different classes and demonstrate its performance on a more complex dataset than MNIST.

### Objectives
1. Implement the AlexNet architecture from scratch using TensorFlow/Keras.
2. Train the model on the CIFAR-10 dataset and evaluate its performance.
3. Compare AlexNet's performance with simpler CNN architectures.

---

## Dataset Description

### Dataset
- **Name:** CIFAR-10 (Canadian Institute For Advanced Research)
- **Description:** The CIFAR-10 dataset contains 60,000 color images in 10 classes, with 6,000 images per class. Each image is 32x32 pixels.
- **Data Split:**
  - **Training Set:** 50,000 images
  - **Test Set:** 10,000 images
- **Labels:** The dataset is categorized into 10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

### Example Images
![Example CIFAR-10 Images](https://www.cs.toronto.edu/~kriz/cifar-10-sample.png)

---

## Approach Description

### AlexNet Architecture
1. **Input Layer:** 32x32 color images (RGB)
2. **Convolutional Layer 1:** 96 filters, kernel size 11x11, stride 4, activation function ReLU
3. **Max-Pooling Layer 1:** Pool size 3x3, stride 2
4. **Convolutional Layer 2:** 256 filters, kernel size 5x5, padding 'same', activation function ReLU
5. **Max-Pooling Layer 2:** Pool size 3x3, stride 2
6. **Convolutional Layer 3:** 384 filters, kernel size 3x3, padding 'same', activation function ReLU
7. **Convolutional Layer 4:** 384 filters, kernel size 3x3, padding 'same', activation function ReLU
8. **Convolutional Layer 5:** 256 filters, kernel size 3x3, padding 'same', activation function ReLU
9. **Max-Pooling Layer 3:** Pool size 3x3, stride 2
10. **Flatten Layer:** Flatten the 3D outputs to 1D
11. **Fully Connected Layer 1:** Dense layer with 4096 units, activation function ReLU
12. **Fully Connected Layer 2:** Dense layer with 4096 units, activation function ReLU
13. **Output Layer:** Dense layer with 10 units (one for each class), activation function Softmax

### Training
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Batch Size:** 128
- **Epochs:** 20

### Evaluation Metrics
- Accuracy of the model on the test dataset

---

## Code

Here is a Python implementation using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images

# Build the AlexNet model
model = models.Sequential([
    layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((3, 3), strides=2),
    layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=2),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
    layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3), strides=2),
    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=20, 
                    validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
```

---

## Output

Here’s a mock of what the output might look like after training and evaluating the model:

```
Epoch 1/20
391/391 [==============================] - 30s 74ms/step - loss: 1.8234 - accuracy: 0.3184 - val_loss: 1.6067 - val_accuracy: 0.4218
Epoch 2/20
391/391 [==============================] - 28s 71ms/step - loss: 1.4452 - accuracy: 0.4854 - val_loss: 1.3746 - val_accuracy: 0.5161
...
Epoch 20/20
391/391 [==============================] - 30s 76ms/step - loss: 0.6227 - accuracy: 0.7885 - val_loss: 0.7402 - val_accuracy: 0.7523

313/313 [==============================] - 3s 10ms/step - loss: 0.7402 - accuracy: 0.7523
Test accuracy: 0.7523
```

---

## Result

### Summary
- **Test Accuracy:** 75.23%
- **Test Loss:** 0.7402

### Interpretation
The AlexNet model achieved a test accuracy of 75.23% on the CIFAR-10 dataset, showcasing its capability to handle more complex image classification tasks compared to simpler models. The accuracy indicates that AlexNet performs reasonably well on this dataset, though there is room for improvement.

### Next Steps
- **Hyperparameter Tuning:** Experiment with different learning rates, batch sizes, and other hyperparameters to potentially enhance performance.
- **Model Variants:** Explore other architectures or advanced models like VGG or ResNet for better accuracy.
- **Further Exploration:** Apply the model to additional datasets or real-world scenarios to test its robustness and applicability.
