# Handwritten Digit Classification using CNNs

This repository contains a project focused on using Convolutional Neural Networks (CNNs) for the task of handwritten digit classification from the MNIST dataset. The project features three different CNN architectures to explore various aspects of neural network design and performance.

## Project Overview

The models implemented in this project are:
1. **Basic CNN**: A simple CNN with two convolutional layers, Sigmoid activations, and heavy dropout.
2. **Digit Classifier**: A slightly deeper network using ReLU activations and pooling layers.
3. **MNIST Net**: An advanced CNN model employing Kaiming He initialization and a Dropout2d layer for improved generalization.

These models are defined within PyTorch, a powerful library for building neural networks. The project demonstrates training these models, evaluating their performance, and visualizing their training losses over epochs.

## Prerequisites

Before running this project, ensure you have the following installed:
- Python 3.8 or above
- PyTorch 1.8 or newer
- torchvision
- matplotlib

## Installation

Clone this repository to your local machine using:
```bash
git clone https://github.com/your-github-username/handwritten-digit-classification.git
cd handwritten-digit-classification
```

## Usage

To run the training and evaluation process, execute the following command from the root directory of the project:
```bash
python cnn_digit_classifier.py
```

## Key Components

- `CNN`: The primary module for the basic CNN architecture.
- `Digit_Classifier`: Module for a more complex CNN better suited for robustness.
- `MNIST_Net`: The most advanced model in this collection.
- `RunModel`: A class that handles data loading, model training, testing, and result visualization.

Each model's architecture is defined in `cnn_digit_classifier.py`, along with the training and testing procedures.

## Results

After training, the models' accuracies on the MNIST test set will be printed, and a plot showing the training losses over epochs will be displayed.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Contact

Project Link: [https://github.com/your-github-username/handwritten-digit-classification](https://github.com/your-github-username/handwritten-digit-classification)
