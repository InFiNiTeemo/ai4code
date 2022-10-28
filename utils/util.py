class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def split_dataset(df, frac=0.9, logger=None):
    if logger is not None:
        logger.info("**split dataset**")
    if isinstance(df, dict):
        return dict(df.items()[:int(len(df) * frac)]), dict(df.items()[int(len(df) * frac):])
    elif isinstance(df, list):
        return df[:int(len(df) * frac)], df[int(len(df) * frac):]
    if logger is not None:
        logger.info("*split dataframe*")
    index = df.sample(frac=frac).index
    train_df = df[df.index.isin(index)]
    val_df = df[~df.index.isin(index)]
    return train_df, val_df

def get_optimizer_grouped_parameters(
        model, model_type,
        learning_rate, weight_decay,
        layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "fc" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters