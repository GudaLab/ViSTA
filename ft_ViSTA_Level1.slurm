#!/bin/bash
#SBATCH --job-name=fine_ViSTA
#SBATCH --output=fine_ViSTA.out
#SBATCH --error=fine_ViSTA.err
#SBATCH --time=6-23:00:00
#SBATCH --mem=1450G
#SBATCH --cpus-per-task=15       ## It will occupy 15*7 = 105cpu, decide accordingly
##SBATCH --gres=gpu:8
##SBATCH --ntasks=8                # Number of tasks = number of GPUs
##SBATCH --ntasks-per-node=8      # 8 GPUs on 1 node

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 0,1,2,3,4,5,6,7, mention this to defin which gpu want to use
## conda activate dnabert2
# Configuration
export num_gpu=8      # Please change the value based on your setup

VCWin=4 # size of Varient centered window
SplitSeqLength=1000 # length of input sequence
export MAX_LENGTH=512 
export LR=1e-4 # learning rate

# Base and derived paths
export BASE_DIR="/"
export DATA_PATH="${BASE_DIR}/input_data" # Input data for finetuning
export OUTPUT_DIR="${BASE_DIR}/ft_ViSTA" # output directory for finetuned model

##--  pretrained models ---##
export pretrained_ViSTA_MaxLen512="${BASE_DIR}/pt_ViSTA" # pretrained model

start_time=$(date +%s)
for seed in 2
do
  echo "Running on data from ${DATA_PATH} with seed: $seed"
  torchrun --nproc_per_node=${num_gpu} ViSTA.py \
      --model_name_or_path $pretrained_ViSTA_MaxLen512 \
      --tokenizer_name /pt_ViSTA \
      --data_path $DATA_PATH \
      --kmer -1 \
      --run_name ViSTA_${LR}_seed${seed} \
      --model_max_length ${MAX_LENGTH} \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 4 \
      --learning_rate ${LR} \
      --num_train_epochs 2000 \
      --fp16 \
      --save_steps 5 \
      --output_dir $OUTPUT_DIR \
      --save_strategy epoch \
      --eval_strategy epoch \
      --logging_strategy epoch \
      --warmup_steps 100 \
      --overwrite_output_dir True \
      --log_level info \
      --lr_scheduler_type "linear" \
      --find_unused_parameters False \
      --use_lora \
      --load_best_model_at_end True \
      --metric_for_best_model eval_accuracy \
      --greater_is_better True
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
elapsed_minutes=$((elapsed / 60))
elapsed_seconds=$((elapsed % 60))
echo "Time taken: ${elapsed_minutes} minutes and ${elapsed_seconds} seconds"