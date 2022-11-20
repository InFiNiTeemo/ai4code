### Overview
A project for nlp competition, contains:
1. basic pipeline
2. model tuning: various kinds of pooling..
3. LLRD, Gradient clippling


### TODO
1. pretrain pipeline
2. solve too many exp folder, (when test, no need to create new exp folder), move exp1, exp2 to exp 
3. print special tokens in input_ids


实验
目前 new_module_lr 1e-5 ~ 4e-5 已尝试
复现 exp107，  exp93

epoch的数量会影响学习率， 故不是说epoch越大越好

llrd_v2 + new_module_lr 1e-5 (exp) + layer_reinit





### backbone
1. xlm-roberta-base
2. microsoft/deberta-v3-base
3. microsoft/deberta-v3-large
4. microsoft/deberta-v2-xlarge
5. microsoft/deberta-v2-xlarge-mnli 
6. microsoft/deberta-v2-xxlarge
7. funnel-transformer-xlarge
8. funnel-transformer-large
9. bigbird-roberta-large
10. bigbird-roberta-base

longformer-large, roberta-large, deberta-xxlarge, distilbart_mnli_12_9, bart_large_finetuned_squadv1