import argparse
import logging
import math
import os
import random

import datasets
import torch
from datasets import  load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    default_data_collator,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )

    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lenght` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--per_device_test_batch_size",
        type=int,
        default=100,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default="tag",
        choices=["ner", "pos", "chunk","tag"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )

    # add by cuifan  2021 8 20
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.test_file is None :
        raise ValueError("Need  a test file.")
    if args.test_file is not None:
        extension = args.test_file.split(".")[-1]
        assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."

    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        filename = "my.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets for token classification task available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'tokens' or the first column if no column called
    # 'tokens' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    data_files = {}

    if args.test_file is not None:
        data_files["test"] = args.test_file
    extension = args.test_file.split(".")[-1]
    raw_datasets = load_dataset(extension,name='comma', delimiter='\t' ,data_files=data_files)

    column_names = raw_datasets["test"].column_names

    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif "tokens" in column_names:
        text_column_name = "tokens"
    else:
        text_column_name = column_names[0]

    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f"{args.task_name}_tags" in column_names:
        label_column_name = f"{args.task_name}_tags"
    else:
        label_column_name = column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(raw_datasets["test"][label_column_name])
    label_list = ["T","F"]                        # add by cuifan 此处写法有点傻逼，为了解决0对应T，1对应F
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)

    if args.model_name_or_path:
        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForTokenClassification.from_config(config)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False


    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            #is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label_to_id[label[word_idx]] if args.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = processed_raw_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {test_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorForTokenClassification` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_test_batch_size)


    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Prepare everything with our `accelerator`.
    model,test_dataloader = accelerator.prepare(
        model,test_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)


    # Metrics
    #metric = load_metric("seqeval")

    def get_labels(predictions, references):
        # Transform predictions and references tensos to numpy arrays
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels

    #  Add by cuifan
    def evaluation(preds, refs,TP,FP,FN):
        all_predictions_index = []
        all_truth_index = []
        for idx, prediction_id in enumerate(preds):
            temp = []
            for i in range(len(prediction_id)):
                if prediction_id[i] == "F":
                    temp.append(i)
            all_predictions_index.append(temp)
        for truth_id in refs:
            temp = []
            for idx, num in enumerate(truth_id):
                if num == "F":
                    temp.append(idx)
            all_truth_index.append(temp)
        for idx, sentence in enumerate(all_predictions_index):
            for i in sentence:
                if i in all_truth_index[idx]:
                    TP += 1
                else:
                    FP += 1

        for idx, sentence in enumerate(all_truth_index):
            for i in sentence:
                if i in all_predictions_index[idx]:
                    continue
                else:
                    FN += 1

        return TP, FP, FN

    def compute_prf(TP,FP,FN):
        P = TP / (TP + FP) if (TP + FP) > 0 else 0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * (P * R) / (P + R) if (P + R) > 0 else 0
        return P*100,R*100,F1*100


    logger.info("***** Running Testing *****")
    model.eval()
    TP = 0
    FP = 0
    FN = 0
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        if not args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
            predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        preds, refs = get_labels(predictions_gathered, labels_gathered)
        # add by cuifan
        TP,FP,FN = evaluation(preds,refs,TP,FP,FN)  # predictions and preferences are expected to be a nested list of labels, not label_ids
        print("###############")
        print(TP,FP,FN)
        print("#############")
    # eval_metric = metric.compute()
    P,R,F1 = compute_prf(TP,FP,FN)
    logger.info(f"  {args.test_file}   Result: P:{P},R:{R},F1:{F1}")



if __name__ == "__main__":
    main()