# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

accelerate launch --num_processes 8 --num_machines 1 --multi_gpu run_summarization_no_trainer.py \
  --model_name_or_path local_path/bart-base-xsum_from_Aktsvigun \
  --dataset_name xsum \
  --dataset_config_name 3.0.0 \
  --pred_distill \
  --hid_distill \
  --num_train_epochs 20 \
  --weight_bits $1 \
  --input_bits $2 \
  --do_train \
  --do_test \
  --distill_encoder 6 \
  --distill_decoder 6 \
  --learning_rate $3 \
