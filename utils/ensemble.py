import pandas as pd

""""

test = pd.read_csv('../input/feedback-prize-english-language-learning/test.csv')
submission = pd.read_csv('../input/feedback-prize-english-language-learning/sample_submission.csv')

sub1 = pd.read_csv(f'submission_1.csv')[target_cols] * CFG1.weight
sub2 = pd.read_csv(f'submission_2.csv')[target_cols] * CFG2.weight
sub3 = pd.read_csv(f'submission_3.csv')[target_cols] * CFG3.weight
sub4 = pd.read_csv(f'submission_4.csv')[target_cols] * CFG4.weight
sub5 = pd.read_csv(f'submission_5.csv')[target_cols] * CFG5.weight
sub6 = pd.read_csv(f'submission_6.csv')[target_cols] * CFG6.weight
sub7 = pd.read_csv(f'submission_7.csv')[CFG7.target_cols] * CFG7.weight
sub8 = pd.read_csv(f'submission_8.csv')[CFG8.target_cols] * CFG8.weight
sub9 = pd.read_csv(f'submission_9.csv')[CFG9.target_cols] * CFG9.weight
sub10 = pd.read_csv(f'submission_10.csv')[CFG10.target_cols] * CFG10.weight

ens = ((sub1 + sub2 + sub3 + sub4 + sub5 + sub6 + sub7 + sub8 + sub9 + sub10)
       /(CFG1.weight + CFG2.weight + CFG3.weight + CFG4.weight + CFG5.weight 
         + CFG6.weight + CFG7.weight + CFG8.weight + CFG9.weight + CFG10.weight))

#ens = (sub1 + sub2)/(CFG1.weight + CFG2.weight)

submission[CFG1.target_cols] = ens
display(submission.head())
submission.to_csv('submission.csv', index=False)
"""


## william
## https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/363773
"""
class CFG:
    target_cols = [
        "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"
    ]

    model_file_pattern = "fold_{}_best.pth"
    model_dirs = [
        "../input/model-cls-4folds-4epochs",
        "../input/model-mp-4folds-3epochs",
        "../input/model-att-4folds-3epochs",
        "../input/model-max-4folds-3epochs",
    ]
    weights = [0.32605782, 0.33205295, 0.3259095, 0.01597973]
    tune_weights = True

oof_dfs = []
for model_dir in CFG.model_dirs:
    oof_df = pd.read_pickle(f"{model_dir}/oof_df.pkl")
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    oof_dfs.append(oof_df)
    logger.info(f"Score: {score:<.6f}  Scores: {scores}")

if CFG.tune_weights:
    from scipy import optimize

    def loss(weights):
        return get_score(labels, np.clip(np.average(oof_preds, weights=weights, axis=0), 1, 5))[0]

    opt_weights = optimize.minimize(
        loss,
        [1/len(CFG.model_dirs)] * len(CFG.model_dirs),
        constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
        method= "SLSQP", #'SLSQP',
        bounds=[(0.0, 1.0)] * len(CFG.model_dirs),
        options = {'ftol':1e-10},
    )["x"]

    opt_weights = np.array(opt_weights) / sum(opt_weights)
    logger.info("score: %s", loss(opt_weights))
    logger.info(opt_weights)
    
"""


"""
https://www.kaggle.com/code/illidan7/ensemble-oof-101
"""