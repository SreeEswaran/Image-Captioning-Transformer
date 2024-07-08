# Image-Captioning-Transformer

This project demonstrates an image captioning model using a Transformer architecture. The model takes an image as input and generates a descriptive caption. We use the COCO dataset for training and evaluation.

## Project Overview

Image captioning is a challenging task that involves understanding the content of an image and describing it in natural language. In this project, we build an image captioning model using a Transformer-based architecture. The model is trained on the COCO dataset and can generate captions for new images.

## Dataset

We use the COCO (Common Objects in Context) dataset for training and evaluating the model. The dataset contains images of complex scenes with various objects, along with annotations describing the objects and their relationships.

### Download Dataset

Download the COCO dataset from [here](http://cocodataset.org/#download).

### Preprocessing

The dataset needs to be preprocessed before feeding it into the model. The preprocessing steps include resizing images, tokenizing captions, and creating data loaders for training and evaluation.

## Model Architecture

The image captioning model consists of two main components:
1. **Encoder**: A convolutional neural network (CNN) that extracts features from the input image.
2. **Decoder**: A Transformer model that generates captions based on the extracted image features.

The encoder is a pre-trained CNN (e.g., ResNet-50), and the decoder is a Transformer with self-attention mechanisms.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/SreeEswaran/Image-Captioning-Transformer.git
    cd Image-Captioning-Transformer
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download and preprocess the COCO dataset as described above.

## Usage

### Training

To train the model, run:
```bash
python train_model.py --config configs/train_config.yaml
```
<!--python train_model.py --config configs/train_config.yaml-->


To genearte the images, run:
```bash
python infer.py --image_path path/to/your/image.jpg --model_path path/to/saved/model.pth
```
