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

# train
#python kaggle_ELL_train.py \
#      --seed 42 \
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --is_train

# train


#python kaggle_ELL_train.py --n_workers 2 --seed 42\
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --is_train \
#      --model_name_or_path "xlm-roberta-base"


## train with FGM
#python kaggle_ELL_train.py \
#      --seed 42 \
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --is_train \
#      --attacker "fgm"



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

#python kaggle_ELL_train.py --n_workers 2 --seed 42\
#      --n_folds 5 \
#      --epochs 8 \
#      --eval_times_per_epoch 3 \
#      --is_oof \
#      --test_model_path outputs/kaggle-ELL/exp206



# exp
python kaggle_ELL_train.py --n_workers 2 --seed 42\
      --n_folds 5 \
      --epochs 5 \
      --eval_times_per_epoch 3 \
      --is_exp
#
#python kaggle_ELL_train.py --n_workers 2 --seed 42\
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v3-large"
#
#python kaggle_ELL_train.py --n_workers 2 --seed 42\
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v2-xlarge"
#
#python kaggle_ELL_train.py --n_workers 2 --seed 42\
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v2-xlarge-mnli"
#
#python kaggle_ELL_train.py --n_workers 2 --seed 42\
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --batch_size 4 \
#      --is_train \
#      --model_name_or_path "microsoft/deberta-v2-xxlarge"
#
#python kaggle_ELL_train.py --n_workers 2 --seed 42\
#      --n_folds 5 \
#      --epochs 5 \
#      --eval_times_per_epoch 3 \
#      --is_train \
#      --batch_size 4 \
#      --model_name_or_path "funnel-transformer-xlarge"