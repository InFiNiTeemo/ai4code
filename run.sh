# pretrain
#python kaggle_ELL_train.py --n_workers 2 --seed 43\
#      --batch_size 2 \
#      --n_folds 5 \
#      --epochs 10 \
#      --eval_times_per_epoch 3 \
#      --is_pretrain


# train from pretrained
#python kaggle_ELL_train.py --n_workers 2 --seed 43\
#      --model_name_or_path outputs/kaggle-ELL/exp139 \
#      --n_folds 5 \
#      --epochs 8 \
#      --eval_times_per_epoch 3 \
#      --is_train

#  train
#python kaggle_ELL_train.py \
#      --seed 42 \
#      --n_folds 4 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --is_train

## train with awp
#python kaggle_ELL_train.py \
#      --seed 42 \
#      --n_folds 4 \
#      --epochs 5 \
#      --is_train \
#      --attacker "awp"



## train with FGM
#python kaggle_ELL_train.py \
#      --seed 42 \
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --is_train \
#      --attacker "fgm"

## quick exp
#python kaggle_ELL_train.py \
#      --seed 42 \
#      --n_folds 5 \
#      --epochs 5 \
#      --is_exp \
#      --eval_times_per_epoch 3 \
#      --n_exp_stop_fold 1

# exp
#python kaggle_ELL_train.py \
#       --seed 42 \
#       --n_folds 4 \
#       --epochs 5 \
#       --n_exp_stop_fold 3 \
#       --is_exp



# oof
#python kaggle_ELL_train.py --n_workers 2 --seed 43\
#      --n_folds 5 \
#      --epochs 8 \
#      --eval_times_per_epoch 3 \
#      --parallel 0 \
#      --is_oof \
#      --test_model_path outputs/kaggle-ELL/exp65  \
#      --on_kaggle \
#      --parallel 1

#python kaggle_ELL_train.py --n_workers 2 --seed 42 \
#      --n_folds 4 \
#      --epochs 8 \
#      --eval_times_per_epoch 3 \
#      --is_oof \
#      --test_model_path outputs/kaggle-ELL/exp579


#python kaggle_ELL_train.py --seed 42 \
#      --n_folds 4 \
#      --epochs 5 \
#      --is_train \
#      --batch_size 4 \
#      --model_name_or_path "funnel-transformer/large"

# out of memory though
#python kaggle_ELL_train.py --seed 42 \
#      --n_folds 4 \
#      --epochs 5 \
#      --batch_size 1 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v2-xxlarge"

# seed 42
#python kaggle_ELL_train.py \
#      --seed 42 \
#      --n_folds 10 \
#      --epochs 5 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v3-large"


#python kaggle_ELL_train.py \
#      --seed 42 \
#      --n_folds 4 \
#      --epochs 5 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v3-large"

#python kaggle_ELL_train.py --seed 42 \
#      --n_folds 4 \
#      --epochs 5 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v2-xlarge-mnli"
#
#python kaggle_ELL_train.py --seed 42 \
#      --n_folds 4 \
#      --epochs 5 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v2-xlarge"
##
#
##
### seed 43
#python kaggle_ELL_train.py --seed 43 \
#      --n_folds 4 \
#      --epochs 5 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v2-xlarge"
#
#python kaggle_ELL_train.py \
#      --seed 43 \
#      --n_folds 4 \
#      --epochs 5 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v3-large"
#
#python kaggle_ELL_train.py --seed 43 \
#      --n_folds 4 \
#      --epochs 5 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v2-xlarge-mnli"





