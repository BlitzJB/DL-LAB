|Assignment - 2  | Joshua Bharathi |
|--|--|
| Deep Learning Lab | 22011101045 |

## Question

1. Implement CNN using any image dataset. Apply variants of convolution operations such as dilation, transpose convolution, etc.
2. Vary the number of convolution layers and different types of pooling with different filter sizes, etc. Record the results, analyze performance in terms of accuracy and the number of parameters. Write your observation.

### Aim
To implement various CNN architectures and analyze how changes in depth, convolution types, and pooling strategies affect model accuracy and parameters.

---

## Dataset Description

### Dataset
- **Name:** CIFAR-10
- **Description:** The CIFAR-10 dataset consists of 60,000 color images, each of size 32x32 pixels, categorized into 10 classes with 6,000 images per class.
- **Data Split:**
  - **Training Set:** 50,000 images
  - **Test Set:** 10,000 images
- **Classes:**
  0. Airplane
  1. Automobile
  2. Bird
  3. Cat
  4. Deer
  5. Dog
  6. Frog
  7. Horse
  8. Ship
  9. Truck
- **Note:** CIFAR-10 is challenging, with many traditional methods struggling to surpass 70% accuracy. This dataset is chosen to highlight significant differences in performance due to architectural changes.

### Example Images
![Example CIFAR-10 Images](https://www.cs.toronto.edu/~kriz/cifar-10-sample.png)

---

## Approach Description

### CNN Architecture Variations
1. **Baseline Model:**
   - Convolution Layers: 2
   - Pooling: Max pooling
2. **Deep Model:**
   - Convolution Layers: 4
   - Pooling: Max pooling
3. **Wide Model:**
   - Convolution Layers: 2
   - Increased filters per layer
4. **Dilated Model:**
   - Convolution Layers: 2
   - Dilation Rate: (2,2)
5. **Transpose Model:**
   - Convolution Layers: 2
   - Transpose Convolution
6. **Large Filter Model:**
   - Convolution Layers: 2
   - Filter Size: (5,5)
7. **Average Pooling Model:**
   - Convolution Layers: 2
   - Pooling: Average pooling

### Training
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** Adam
- **Batch Size:** 64
- **Epochs:** 10

### Evaluation Metrics
- Accuracy of the model on the test dataset
- Number of parameters in the model

---

## Code

Here is a Python implementation using TensorFlow and Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Function to create a CNN model
def create_cnn_model(conv_layers, pool_type='max', filter_size=(3,3),
                     dilation_rate=(1,1), use_transpose=False):
    model = models.Sequential()

    for i, filters in enumerate(conv_layers):
        if i == 0:
            if use_transpose:
                model.add(layers.Conv2DTranspose(filters, filter_size, activation='relu',
                                                  padding='same', input_shape=(32, 32, 3)))
            else:
                model.add(layers.Conv2D(filters, filter_size, activation='relu',
                                        padding='same', input_shape=(32, 32, 3), dilation_rate=dilation_rate))
        else:
            if use_transpose:
                model.add(layers.Conv2DTranspose(filters, filter_size, activation='relu',
                                                  padding='same'))
            else:
                model.add(layers.Conv2D(filters, filter_size, activation='relu',
                                        padding='same', dilation_rate=dilation_rate))

        if i < len(conv_layers) - 1: # Don't pool after the last conv layer
            if pool_type == 'max':
                model.add(layers.MaxPooling2D((2, 2)))
            elif pool_type == 'avg':
                model.add(layers.AveragePooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

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

# Experiment with different architectures
experiments = [
    {'name': 'Baseline', 'conv_layers': [32, 64]},
    {'name': 'Deep', 'conv_layers': [32, 64, 128, 256]},
    {'name': 'Wide', 'conv_layers': [64, 128]},
    {'name': 'Dilated', 'conv_layers': [32, 64], 'dilation_rate': (2,2)},
    {'name': 'Transpose', 'conv_layers': [32, 64], 'use_transpose': True},
    {'name': 'Large Filter', 'conv_layers': [32, 64], 'filter_size': (5,5)},
    {'name': 'Avg Pooling', 'conv_layers': [32, 64], 'pool_type': 'avg'},
]

results = []
for exp in experiments:
    model = create_cnn_model(
        conv_layers=exp['conv_layers'],
        pool_type=exp.get('pool_type', 'max'),
        filter_size=exp.get('filter_size', (3,3)),
        dilation_rate=exp.get('dilation_rate', (1,1)),
        use_transpose=exp.get('use_transpose', False)
    )
    history, accuracy, params = train_and_evaluate(model)
    results.append({
        'name': exp['name'],
        'accuracy': accuracy,
        'params': params,
        'history': history
    })

# Print results
for result in results:
    print(f"{result['name']}:")
    print(f" Accuracy: {result['accuracy']:.4f}")
    print(f" Parameters: {result['params']}")
    print()

# Plot training history
plt.figure(figsize=(12, 8))
for result in results:
    plt.plot(result['history'].history['val_accuracy'], label=result['name'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
```

---

## Results

### Summary
- **Baseline Model:**
  - **Accuracy:** 0.5217
  - **Parameters:** 24,202
- **Deep Model:**
  - **Accuracy:** 0.7700
  - **Parameters:** 405,514
- **Wide Model:**
  - **Accuracy:** 0.5753
  - **Parameters:** 84,554
- **Dilated Model:**
  - **Accuracy:** 0.5195
  - **Parameters:** 24,202
- **Transpose Model:**
  - **Accuracy:** 0.4982
  - **Parameters:** 24,202
- **Large Filter Model:**
  - **Accuracy:** 0.5848
  - **Parameters:** 58,506
- **Average Pooling Model:**
  - **Accuracy:** 0.4856
  - **Parameters:** 24,202

### Interpretation
1. The baseline model provides a good starting point for comparison.
2. Deeper models (more layers) tend to have higher accuracy but also more parameters.
3. Wider models (more filters per layer) can improve accuracy with a moderate increase in parameters.
4. Dilated convolutions can increase the receptive field without adding parameters, potentially improving accuracy.
5. Transpose convolutions may help in certain scenarios but can be more challenging to train.
6. Larger filter sizes can capture more spatial information but increase the number of parameters.
7. Average pooling instead of max pooling can provide different feature representations, affecting the model's performance.
8. There's often a trade-off between model complexity (number of parameters) and accuracy.
9. The choice of architecture depends on the specific problem and computational constraints.
