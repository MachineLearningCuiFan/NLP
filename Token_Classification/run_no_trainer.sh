# Copyright 2020 The HuggingFace Team. All rights reserved.
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

accelerate launch run_csc_no_trainer.py \
  --model_name_or_path "roberta_pretrained" \
  --train_file data/small.csv \
  --validation_file data/test15.csv \
  --label_column_name labels \
  --output_dir /tmp/test-ner \
  --pad_to_max_length \
  --task_name tag  \
  --return_entity_level_metrics  \
  --per_device_eval_batch_size 200 \
  --per_device_train_batch_size 2 \
  --logging_steps 1 \
  --num_warmup_steps 10000 \
