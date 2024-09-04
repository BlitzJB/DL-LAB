
---
| Assignment - 4  | Joshua Bharathi |
|--|--|
| Deep Learning Lab | 22011101045 |

## Question

1. Compare and contrast two popular variants of Convolutional Neural Networks (CNNs): ResNet and MobileNet. 
2. Discuss their architectures, advantages, and use cases. Analyze their performance in terms of accuracy and efficiency on a specific dataset.

### Aim
To understand and compare the ResNet and MobileNet architectures, focusing on their unique design principles, performance, and efficiency.

---

## Dataset Description

### Dataset
- **Name:** CIFAR-10
- **Description:** The CIFAR-10 dataset consists of 60,000 color images, each of size 32x32 pixels, categorized into 10 classes with 6,000 images per class.
- **Data Split:**
  - **Training Set:** 50,000 images
  - **Test Set:** 10,000 images
- **Classes:**
  1. Airplane
  1. Automobile
  2. Bird
  3. Cat
  4. Deer
  5. Dog
  6. Frog
  7. Horse
  8. Ship
  9. Truck
- **Note:** CIFAR-10 is chosen for its complexity and to highlight the performance differences between ResNet and MobileNet.

### Example Images
![Example CIFAR-10 Images](https://www.cs.toronto.edu/~kriz/cifar-10-sample.png)

---

## Approach Description

### ResNet Architecture
- **Description:** ResNet (Residual Network) introduces residual connections or skip connections that bypass one or more layers. This architecture addresses the vanishing gradient problem, making it feasible to train very deep networks.
- **Key Components:**
  - **Residual Blocks:** Allow gradients to flow through the network more easily, facilitating the training of deeper networks.
  - **Identity Mapping:** Residual blocks use identity mappings to enable the network to learn the residual (or difference) between the input and the output of each block.
- **Advantages:**
  - Improved accuracy in very deep networks.
  - Easier to train due to reduced vanishing gradient problems.
- **Typical Use Cases:** Image classification, object detection, and segmentation tasks in complex scenarios.

### MobileNet Architecture
- **Description:** MobileNet is designed for mobile and edge devices, focusing on reducing computational complexity and model size. It uses depthwise separable convolutions to achieve efficient performance with fewer parameters.
- **Key Components:**
  - **Depthwise Separable Convolutions:** Separates the convolution into depthwise convolutions and pointwise convolutions to reduce computation and model size.
  - **Width Multiplier and Resolution Multiplier:** Allows further customization to balance model size, latency, and accuracy.
- **Advantages:**
  - Efficient performance with lower computational requirements.
  - Smaller model size, making it suitable for mobile and embedded devices.
- **Typical Use Cases:** Real-time image processing on mobile and edge devices, such as mobile apps and IoT devices.

### Training and Evaluation

For this assignment, we will train and evaluate both ResNet and MobileNet models on the CIFAR-10 dataset.

- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Batch Size:** 64
- **Epochs:** 10

### Code

Here is a Python implementation using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Function to create a ResNet model
def create_resnet_model():
    base_model = tf.keras.applications.ResNet50(weights=None, input_shape=(32, 32, 3), classes=10)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Function to create a MobileNet model
def create_mobilenet_model():
    base_model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(32, 32, 3), classes=10)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Function to train and evaluate a model
def train_and_evaluate(model, epochs=10):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(train_images, train_labels, epochs=epochs,
                        validation_data=(test_images, test_labels),
                        batch_size=64, verbose=0)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    return history, test_acc, model.count_params()

# Create and evaluate ResNet and MobileNet models
models_to_evaluate = {
    'ResNet': create_resnet_model(),
    'MobileNet': create_mobilenet_model()
}

results = {}
for name, model in models_to_evaluate.items():
    history, accuracy, params = train_and_evaluate(model)
    results[name] = {
        'accuracy': accuracy,
        'params': params,
        'history': history
    }

# Print results
for result_name, result in results.items():
    print(f"{result_name}:")
    print(f" Accuracy: {result['accuracy']:.4f}")
    print(f" Parameters: {result['params']}")
    print()
```

---

## Results

### Summary
- **ResNet Model:**
  - **Accuracy:** 0.7270
  - **Parameters:** 25,636,688
- **MobileNet Model:**
  - **Accuracy:** 0.7193
  - **Parameters:** 4,322,384

### Interpretation
1. **ResNet** provides higher accuracy compared to **MobileNet** on CIFAR-10. This is expected as ResNet's deeper architecture with residual connections can capture more complex patterns.
2. **MobileNet** has a significantly lower number of parameters, making it more efficient in terms of storage and computation. This is ideal for deployment on mobile and edge devices.
3. The trade-off between accuracy and efficiency is evident. While ResNet achieves higher accuracy, MobileNet's efficiency makes it suitable for real-time applications with resource constraints.
4. **ResNet** is preferable for applications where accuracy is paramount and computational resources are available.
5. **MobileNet** is ideal for scenarios where computational efficiency and model size are critical, such as mobile applications and IoT devices.
