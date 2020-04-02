#!/usr/bin/env bash

#CUDA_VISIBLE_DEVICES=0,1,2,3
CUDA_VISIBLE_DEVICES=-1

LARGE_DIR=xlnet_cased_L-12_H-768_A-12
GLUE_DIR=/home/jiangwei/DataSet/GLUE

python3 run_classifier.py \
  --do_train=True \
  --do_eval=False \
  --task_name=sts-b \
  --data_dir=${GLUE_DIR}/STS-B \
  --output_dir=proc_data/sts-b \
  --model_dir=exp/sts-b \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${LARGE_DIR}/xlnet_config.json \
  --init_checkpoint=${LARGE_DIR}/xlnet_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --learning_rate=5e-5 \
  --train_steps=1200 \
  --warmup_steps=120 \
  --save_steps=600 \
  --is_regression=True

# --do_train=True --do_eval=False --task_name=sts-b --data_dir=C:\Works\DataSet\GLUE\STS-B --output_dir=proc_data/sts-b --model_dir=exp/sts-b --uncased=False --spiece_model_file=C:\Works\PretrainedModel\xlnet_cased_L-12_H-768_A-12\spiece.model --model_config_path=C:\Works\PretrainedModel\xlnet_cased_L-12_H-768_A-12\xlnet_config.json --init_checkpoint=C:\Works\PretrainedModel\xlnet_cased_L-12_H-768_A-12\xlnet_model.ckpt --max_seq_length=128 --train_batch_size=8 --num_hosts=1 --num_core_per_host=1 --learning_rate=5e-5 --train_steps=1200 --warmup_steps=120 --save_steps=600 --is_regression=True

python3 run_classifier.py \
  --do_train=False \
  --do_eval=True \
  --task_name=sts-b \
  --data_dir=${GLUE_DIR}/STS-B \
  --output_dir=proc_data/sts-b \
  --model_dir=exp/sts-b \
  --uncased=False \
  --spiece_model_file=${LARGE_DIR}/spiece.model \
  --model_config_path=${LARGE_DIR}/xlnet_config.json \
  --max_seq_length=128 \
  --eval_batch_size=32 \
  --num_hosts=1 \
  --num_core_per_host=1 \
  --eval_all_ckpt=True \
  --is_regression=True


# --do_train=False --do_eval=True --task_name=sts-b --data_dir=C:\Works\DataSet\GLUE\STS-B --output_dir=proc_data/sts-b --model_dir=exp/sts-b --uncased=False --spiece_model_file=C:\Works\PretrainedModel\xlnet_cased_L-12_H-768_A-12\spiece.model --model_config_path=C:\Works\PretrainedModel\xlnet_cased_L-12_H-768_A-12\xlnet_config.json --max_seq_length=128 --eval_batch_size=8 --num_hosts=1 --num_core_per_host=1 --eval_all_ckpt=True --is_regression=True
