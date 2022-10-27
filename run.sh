#python yb_train.py --total_max_len 168 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train_cut0.csv --val_path ./data/yb_test_cut0.csv --seed 0  --fold 0  --model_name_or_path "hfl/chinese-macbert-large"
#python cd_train.py --total_max_len 256 --batch_size 16 --accumulation_steps 4 --epochs 3 --n_workers 8 # --train_path ./data/yb_train0.csv --val_path ./data/yb_test0.csv --seed 0  --fold 0  --model_name_or_path "hfl/chinese-roberta-wwm-ext-large"
#python yb_train.py --total_max_len 256 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train0.csv --val_path ./data/yb_test0.csv --seed 0  --fold 0  --model_name_or_path "fnlp/bart-large-chinese"


#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train1.csv --val_path ./data/yb_test1.csv --seed 1  --fold 1  --model_name_or_path 'hfl/chinese-macbert-large'
#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train2.csv --val_path ./data/yb_test2.csv --seed 2  --fold 2  --model_name_or_path 'hfl/chinese-macbert-large'
#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train3.csv --val_path ./data/yb_test3.csv --seed 3  --fold 3  --model_name_or_path 'hfl/chinese-macbert-large'
#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train4.csv --val_path ./data/yb_test4.csv --seed 4  --fold 4  --model_name_or_path 'hfl/chinese-macbert-large'
#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train0.csv --val_path ./data/yb_test0.csv --seed 5  --fold 0  --model_name_or_path 'hfl/chinese-macbert-base'
#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train1.csv --val_path ./data/yb_test1.csv --seed 6  --fold 1  --model_name_or_path 'hfl/chinese-macbert-base'
#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train2.csv --val_path ./data/yb_test2.csv --seed 7  --fold 2  --model_name_or_path 'hfl/chinese-macbert-base'
#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train3.csv --val_path ./data/yb_test3.csv --seed 8  --fold 3  --model_name_or_path 'hfl/chinese-macbert-base'
#python yb_train.py --total_max_len 128 --batch_size 16 --accumulation_steps 4 --epochs 10 --n_workers 8 --train_path ./data/yb_train4.csv --val_path ./data/yb_test4.csv --seed 9  --fold 4  --model_name_or_path 'hfl/chinese-macbert-base'

# 13g ->
python kaggle_ELL_train.py --total_max_len 1200 --batch_size 2 --epochs 5 --n_workers 8 \
      --seed 43 --n_folds 10 \
      --eval_times_per_epoch 5 \
      --is_experiment_stage \
      --is_test # --model_name_or_path "hfl/chinese-macbert-large" # --train_path ./data/yb_train0.csv --val_path ./data/yb_test0.csv --seed 0  --fold 0  --model_name_or_path "hfl/chinese-roberta-wwm-ext-large"
