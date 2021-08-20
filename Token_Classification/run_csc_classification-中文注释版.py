#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""

import argparse
import logging
import math
import os
import random

import datasets
import torch
from datasets import ClassLabel, load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/token-classification/requirements.txt")

# 您应该将其更新到您的特定问题，以获得更好的 `model_type` 文档
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task (NER) with accelerate library"
    )


    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
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
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="ner",
        choices=["ner", "pos", "chunk"],
        help="The name of the task.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--save_epochs",
        type=int,
        default=1,
        help="How many steps to save the model once.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    # 初始化加速器。 在此示例中，我们将让加速器为我们处理设备放置。
    accelerator = Accelerator()
    # 使用调试配置对每个进程进行一记录
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        filename="my.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # 设置日志记录，我们只希望每台机器有一个进程在屏幕上记录内容。
    # 对于每台机器一个进程，accelerator.is_local_main_process 仅为 True。
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # 如果传递，现在设置训练种子
    if args.seed is not None:
        set_seed(args.seed)

    # 获取数据集：您可以提供自己的 CSV/JSON/TXT 训练和评估文件（见下文）
    # 或者只提供一个公共数据集的名称，用于中心上可用的令牌分类任务 https://huggingface.co/datasets/
    # （数据集将自动从数据集中心下载）。
    # 对于 CSV/JSON 文件，此脚本将使用名为 'tokens' 的列，如果没有名为的列，则使用第一列
    # 找到“令牌”。 您可以轻松调整此行为（见下文）。
    # 在分布式训练中，load_dataset 函数保证只有一个本地进程可以并发
    # 下载数据集。

    if args.dataset_name is not None:
        #  从hub下载和加载数据集。
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # 修剪一些训练样例
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # 有关加载任何类型的标准或自定义数据集（来自文件、python dict、pandas DataFrame 等）的更多信息，请访问
    #      # https://huggingface.co/docs/datasets/loading_datasets.html。

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
        features = raw_datasets["train"].features
    else:
        column_names = raw_datasets["validation"].column_names
        features = raw_datasets["validation"].features

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

    # 如果标签不是`Sequence[ClassLabel]`，我们将需要遍历数据集以获取
     # 独特的标签。
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # 加载预训练模型和分词器
    #
    # 在分布式训练中，.from_pretrained 方法保证只有一个本地进程可以并发
    # 下载模型和词汇。

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

    #可根据实际情况，通过 model.resize_token_embeddings(len(tokenizer)) 重新定义词表的大小。
    model.resize_token_embeddings(len(tokenizer))

    # 预处理数据集。
    # 首先我们标记所有文本
    padding = "max_length" if args.pad_to_max_length else False

    # 标记所有文本并将标签与它们对齐。
    # align : 对齐

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            max_length=args.max_length,
            padding=padding,
            truncation=True,
            # 我们使用这个参数是因为我们数据集中的文本是单词列表（每个单词都有一个标签）
            is_split_into_words=True,
        )

        #  print(tokenized_inputs)
        # tokenlized_inputs:{"input_ids":., 'token_type_ids':..,'attention_mask':...,}

        #print(tokenized_inputs.word_ids())
        #  [None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, None, None, None, None,...]max_length

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            # label:[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 0, 0, 7, , 0, 0, 0]
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            print("word:")
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # 特殊标记的单词 id 为 None。 我们将标签设置为 -100，以便它们自动
                # 在损失函数中被忽略。
                if word_idx is None:
                    label_ids.append(-100)
                # 我们为每个单词的第一个标记设置标签。
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
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )
    # processed_raw_datasets :
    """DatasetDict({
        train: Dataset({
            features: ['attention_mask', 'input_ids', 'labels', 'token_type_ids'],
            num_rows: 14041
        })
        validation: Dataset({
            features: ['attention_mask', 'input_ids', 'labels', 'token_type_ids'],
            num_rows: 3250
        })
        test: Dataset({
            features: ['attention_mask', 'input_ids', 'labels', 'token_type_ids'],
            num_rows: 3453
        })
    })"""

    train_dataset = processed_raw_datasets["train"]

    eval_dataset = processed_raw_datasets["validation"]

    # 从训练集中记录一些(3ge)随机样本：
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # 数据加载器创建：
    if args.pad_to_max_length:
        # 如果填充已经完成最大长度，我们使用默认的数据整理器，它将只转换所有内容
        # 到张量。
        data_collator = default_data_collator
    else:
        # 否则，`DataCollatorForTokenClassification` 将为我们应用动态填充（通过填充到最大长度
        # 样本通过）。 使用混合精度时，我们添加`pad_to_multiple_of=8` 将所有张量填充为多个
        # of 8s，这将允许在计算能力 >= 7.5 (Volta) 的 NVIDIA 硬件上使用 Tensor Core。
        data_collator = DataCollatorForTokenClassification(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # 优化器
    # 将权重分成两组，一组有权重衰减，另一组没有。
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # 使用由 `accelerator` 对象提供的设备。
    device = accelerator.device
    model.to(device)

    # 用我们的“加速器”准备好一切。
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    # 注意 -> 在我们获取下面的长度之前需要准备训练数据加载器（因为它的长度将是
    #  在多进程中更短）

    # 围绕训练步骤数的调度程序和数学。
    # math.ceil()函数返回大于整数的最小整数值
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    # Warmup是在ResNet论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率，
    # 训练了一些epoches或者steps(比如4个epoches,10000steps),再修改为预先设置的学习来进行训练。
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,        # default="linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,   # default=0
        num_training_steps=args.max_train_steps,
    )

    # Metrics    seqeval :序列标注算法的模型效果评估
    metric = load_metric("seqeval")

    def get_labels(predictions, references):
        # 将预测和引用张量转换为 numpy 数组
        if device.type == "cpu":
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()

        # 删除被忽略的索引（特殊标记）
        true_predictions = [
            [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_predictions, true_labels



    def compute_metrics():
        results = metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")  #在执行向后/更新传递之前要累积的更新步骤数。 defalut = 1
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # 每台机器上只显示一次进度条。
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps   # 在执行向后/更新传递之前要累积的更新步骤数。 defalut = 1
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                logger.info(f"complete train! completed_steps:{completed_steps}")
                break


        #  每args.save_epochs次epoch 保存一次模型   efault=1
        if args.num_train_epochs % args.save_epochs == 0:
            output_dir = os.path.join(args.output_dir, 'saved_epoch{}'.format(args.num_train_epochs))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model,
                                                    'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)

        #  每个epoch 模型评估
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            if not args.pad_to_max_length:  # 有必要填充预测和标签以供收集
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            predictions_gathered = accelerator.gather(predictions)
            labels_gathered = accelerator.gather(labels)
            preds, refs = get_labels(predictions_gathered, labels_gathered)
            metric.add_batch(
                predictions=preds,
                references=refs,
            )  # 预测和偏好应该是标签的嵌套列表，而不是 label_ids

        # eval_metric = metric.compute()
        eval_metric = compute_metrics()
        accelerator.print(f"epoch {epoch}:", eval_metric)
        logger.info(f" epoch {epoch}- result: = {eval_metric}")

    #if args.output_dir is not None:
    #    accelerator.wait_for_everyone()
    #    unwrapped_model = accelerator.unwrap_model(model)
    #    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
