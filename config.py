from pathlib import Path


def get_model_config():
    """
    Args:
        d_model: Dimensionality of word embedding
        n_blocks: Number of encoder/decoder blocks
            in the encoder/decoder. A block consists
            of attention and feed forward layers
        n_heads: Number of heads for the multi-head
            attention mechanism
        dropout: Value for reinitializing parameters
            during training to avoid overfitting
        d_ff: Dimensionality of the feed forward
            layer and the end of the decoder
    """
    return {
        "d_model": 512,
        "n_blocks": 6,
        "n_heads": 8,
        "dropout": 0.1,
        "d_ff": 2048,
    }


def get_train_config():
    """
    Args:
        batch_size: Amount of training samples pro batch
        n_epochs:  Number of epochs
        train_size: Portion of dataset used for training
        lr: Learning rate
        scheduler_gamma: Decay rate of the the learning rate pro epoch
        seq_len: Set maximal sequence lenght for dataset
        lang_src:  Language of the source texts
        lang_tgt:  Language of target texts
        datasource: Source of datasets
        model_folder: Directory to store the model parameters
        model_basename: Filename of the transformer model
        preload: Preload model to restart training, if the program crashes.
            Can be changed to "latest" or desired training epoch
        tokenizer_file: File to save tokenizer
        experiment_name: For tensorboard to save losses while training
        tests_pro_epoch: Number of test examples used for inference
            after training one epoch
    """
    return {
        "batch_size": 16,
        "n_epochs": 30,
        "train_size": 0.9,
        "lr": 10**-4,
        "scheduler_gamma": 0.95,
        "seq_len": 128,
        "lang_src": "en",  # English
        "lang_tgt": "es",  # Spanish
        "datasource": "opus_books",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "tests_pro_epoch": 2,
    }


def get_weights_file_path(config, epoch: str) -> str:
    """Find path to save the model parameters"""
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def latest_weights_file_path(config):
    """Find the latest trained model file in the weights folder"""
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
