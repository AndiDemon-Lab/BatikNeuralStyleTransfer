# Batik Neural Style Transfer

A deep learning project that applies neural style transfer techniques to create unique batik-inspired artwork.

## Overview

This project combines traditional Indonesian batik patterns with modern neural style transfer algorithms to generate new artistic designs. It uses deep learning to extract and transfer the distinctive style elements of batik onto various content images. The project supports fine-tuning of convolutional neural networks (CNNs) and allows flexible selection of models, including ResNet50, VGG19, and Inception V3.

## Features

- Neural style transfer implementation for batik patterns
- Support for multiple batik styles and content images
- Customizable style transfer parameters
- Fine-tuning of CNN models
- Flexible model selection: ResNet50, VGG19, and Inception V3

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Pillow
- CUDA (optional, for GPU acceleration)

## Installation

To install the project, clone the repository and install the required packages:

```bash
git clone https://github.com/grayscale636/BatikNeuralStyleTransfer.git
cd BatikNeuralStyleTransfer
pip install -r requirements.txt
```

## Usage 

1. Place your content image in the content directory.
2. Select a batik style image from the styles directory.
3. Run the style transfer:

```bash
python main.py
```

## Results

The generated images will be saved in the output directory. Each output combines the content of your input image with the artistic style of the selected batik pattern. Additionally, metadata for each generated image will be stored, providing details about the content and style images used, as well as the model and parameters applied.
