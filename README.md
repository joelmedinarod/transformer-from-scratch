# Build Transformer from Scratch using PyTorch
Build and train Transformer for translating text from German to Spanish.

Developed following tutorial by Umar Jamil:
https://www.youtube.com/watch?v=ISNdQcPhsts&t=569s

Documentation based on the paper "Attention is All you Need"
by Ashish Vaswani, et. al. (2017)

## Dependencies
PyTorch 2.3.0+cu118
Datasets 2.19.2
Tokenizers 0.19.1
Torchmetrics 1.4.0.post0

## Usage
**config.py**: Train and model configurations.
**model.py**: The transformer model.
**dataset.py**: Tokenize sentences and convert into tensor datasets.
**train.py**: Train model
**translate.py**: Use a train model to translate sentences from German to Spanish.