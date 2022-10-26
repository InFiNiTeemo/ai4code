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

