import warnings
from pathlib import Path

import torch

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import BLEUScore, CharErrorRate, WordErrorRate
from tqdm import tqdm

from config import (
    get_model_config,
    get_train_config,
    get_weights_file_path,
    latest_weights_file_path,
)
from dataset import BilingualDataset, causal_mask
from model import Transformer, build_transformer


def get_all_sentences(dataset, language):
    """Get sentences for training tokenizer"""
    for item in dataset:
        yield item["translation"][language]


def get_or_build_tokenizer(config, dataset, language) -> Tokenizer:
    """Get Tokenizer from file, if file exists, else train new one"""
    # for ex. config['tokenizer_file'] = "../tokenizers/tokenizer_(0).json"
    tokenizer_path = Path(config["tokenizer_file"].format(language))
    if not Path.exists(tokenizer_path):
        # Unknown Tokens will be represented by '[UNK]'
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        # four special tokens:
        # - "[UNK]": If word not found in the vocabulary, replace it with "[UNK]"
        # - "[PAD]": Padding to train the transformer
        # - "[SOS]": Start of sentence
        # - "[EOS]": End of sentence
        # for a word to appear on vocabulary, it must have
        # appeared at least twice
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2
        )
        tokenizer.train_from_iterator(
            get_all_sentences(dataset, language), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_dataset(config):

    dataset_raw = load_dataset(
        config["datasource"],
        f'{config["lang_src"]}-{config["lang_tgt"]}',
        split="train",
    )

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config["lang_tgt"])

    # Remove large sequences from dataset
    dataset_raw = [
        item
        for item in dataset_raw
        if len(tokenizer_src.encode(item["translation"][config["lang_src"]]).ids)
        <= config["seq_len"] - 2 # consider SOS and EOS tokens
        and len(tokenizer_src.encode(item["translation"][config["lang_tgt"]]).ids)
        <= config["seq_len"] - 2 # consider SOS and EOS tokens
    ]

    # Split dataset into train and test dataset
    train_size = int(train_config["train_size"] * len(dataset_raw))
    test_size = len(dataset_raw) - train_size
    train_dataset_raw, test_dataset_raw = random_split(
        dataset_raw, [train_size, test_size]
    )

    train_dataset = BilingualDataset(
        train_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    test_dataset = BilingualDataset(
        test_dataset_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt


def train_model(
    model: Transformer,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    device: str,
    config,
):
    """
    Train the Transformer

    Args:
    model: Transformer model to train
    train_dataloader: Dataset for training
    test_dataloader: Dataset for running inferences
    tokenizer_src: Tokenizer for source texts
    tokenizer_tgt: Tokenizer for target texts
    device: Device on which the training loop runs
    config: Additional configuration for the training loop
    """
    # Make sure that the weight folder is created
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    # Initialize tensorboard to visualize charts
    writer = SummaryWriter(config["experiment_name"])

    # Initialize Adam optimizer and StepLR scheduler
    optimizer = Adam(model.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = StepLR(
        optimizer,
        step_size=1,
        gamma=train_config["scheduler_gamma"],
    )

    # Make sure the training can be resumed in case it crashes
    # Restore state of the model and state of the optimizer
    initial_epoch = 0
    global_step = 0

    # Load trained model, if any, else create a new file
    model_filename = None
    if config["preload"] == "latest":
        model_filename = latest_weights_file_path(config)
    elif config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])

    if model_filename:
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        print("No model to preload, starting from scratch")

    # Select CrossEntropyLoss as loss function
    # Ignore PAD token when calculating loss
    # Reduce overfitting by smoothing label vectors
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"),
        label_smoothing=0.1,
    ).to(device)

    # Start training
    for epoch in range(initial_epoch, config["n_epochs"]):

        model.train()

        # Create progress bar with train_dataloader
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

        # Track loss during the epoch to calculate average
        epoch_losses = []

        for batch in batch_iterator:
            # (batch_size, seq_len)
            encoder_input = batch["enc_input"].to(device)
            # (batch_size, seq_len)
            decoder_input = batch["dec_input"].to(device)
            # (batch_size, 1, 1, seq_len)
            encoder_mask = batch["enc_mask"].to(device)
            # (batch_size, 1, seq_len, seq_len)
            decoder_mask = batch["dec_mask"].to(device)

            # Perform forward computations through the transformer
            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch_size, seq_len, d_model)
            decoder_output = model.decode(
                decoder_input, encoder_output, encoder_mask, decoder_mask
            )  # (batch_size, seq_len, d_model)
            proj_output = model.project(
                decoder_output
            )  # (batch_size, seq_len, tgt_vocab_size)

            # Get the labels (batch_size, seq_len)
            label = batch["label"].to(device)

            # To calculate the loss transform model output from
            # (batch_size, seq_len, tgt_vocab_size) to
            # (batch_size * seq_len, tgt_vocab_size)
            loss = loss_fn(
                proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1)
            )

            # Update progress bar
            batch_iterator.set_postfix({"loss": f"{loss:6.3f}"})
            epoch_losses.append(loss.item())

            # Log progress on tensorboard
            writer.add_scalar("train loss", loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Learning rate decay
        scheduler.step()

        print(f"Average train loss pro batch in epoch {epoch}: {(sum(epoch_losses) / len(epoch_losses)):.6f}")

        # Test the model
        run_test(
            model,
            test_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
            train_config["tests_pro_epoch"],
        )

        # Save the model and state of the optimizer at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


def greedy_decode(
    model: Transformer,
    source: torch.Tensor,
    source_mask: torch.Tensor,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: str,
):
    # Get ID of EOS and SOS tokens
    # It does not matter which tokenizer is used for this
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it
    # for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # Initialize the decoder output with the SOS token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for the target (decoder input)
        decoder_mask = (
            causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        )

        # Calculate the output of the decoder
        decoder_output = model.decode(
            decoder_input, encoder_output, source_mask, decoder_mask
        )

        # Get the next token by selecting the highest prediction probability
        pred_probs = model.project(decoder_output[:, -1])
        _, next_word = torch.max(pred_probs, dim=1)

        # Append the new token to the next input sequence of the decoder
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).fill_(next_word.item()).type_as(source).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)  # remove batch dimension


def run_test(
    model: Transformer,
    test_dataset: DataLoader,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    max_len: int,
    device: str,
    print_msg: str,
    global_step: int,
    writer: SummaryWriter,
    num_examples: int = 2,
):
    """Perform inferences using the model and the test dataset"""
    model.eval()
    count = 0

    source_texts = []
    expected_texts = []
    predicted_texts = []

    # Size of the control window (just use a default value)
    console_width = 80

    # Disable gradient calculation for testing
    with torch.inference_mode():
        for batch in test_dataset:
            count += 1
            # (batch_size, seq_len)
            encoder_input = batch["enc_input"].to(device)
            # (batch_size, 1, 1, seq_len)
            encoder_mask = batch["enc_mask"].to(device)

            assert encoder_input.size(0) == 1, "Batch size for inference must be 1"

            model_output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            # Get the input text
            source_text = batch["src_text"][0]

            # Get the label
            target_text = batch["tgt_text"][0]

            # Convert model output (tokens) back to text
            model_output_text = tokenizer_tgt.decode(
                model_output.detach().cpu().numpy()
            )

            source_texts.append(source_text)
            expected_texts.append(target_text)
            predicted_texts.append(model_output_text)

            # Print to the console
            print_msg("-" * console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_output_text}")

            if count == num_examples:
                print_msg("-" * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate
        metric = CharErrorRate()
        cer = metric(predicted_texts, expected_texts)
        writer.add_scalar("validation cer", cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = WordErrorRate()
        wer = metric(predicted_texts, expected_texts)
        writer.add_scalar("validation wer", wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = BLEUScore()
        bleu = metric(predicted_texts, expected_texts)
        writer.add_scalar("validation BLEU", bleu, global_step)
        writer.flush()


if __name__ == "__main__":
    # Turn off warnings
    warnings.filterwarnings("ignore")

    # Get configurations
    train_config = get_train_config()
    model_config = get_model_config()

    # Load dataset
    train_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(
        train_config
    )

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get model
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        train_config["seq_len"],
        train_config["seq_len"],
        model_config["d_model"],
        model_config["n_blocks"],
        model_config["n_heads"],
        model_config["dropout"],
        model_config["d_ff"],
    ).to(device)

    # Train the model
    train_model(
        model,
        train_dataloader,
        test_dataloader,
        tokenizer_src,
        tokenizer_tgt,
        device,
        train_config,
    )
