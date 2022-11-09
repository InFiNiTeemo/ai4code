def get_optimizer_grouped_parameters(
        model,
        model_type,
        new_module_lr=1e-5,
        encoder_lr=1e-5,
        weight_decay=0.01,
        layerwise_learning_rate_decay=0.9,
        is_parallel = False
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "fc" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": new_module_lr,
        },
    ]
    # initialize lrs for every layer
    # num_layers = model.model.config.num_hidden_layers
    if is_parallel:
        layers = [model.module.model.embeddings] + list(model.module.model.encoder.layer)
    else:
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
    layers.reverse()
    lr = encoder_lr
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


# from: https://zhuanlan.zhihu.com/p/449676168
def get_optimizer_grouped_parameters_ner(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    group1=['layer.0.','layer.1.','layer.2.','layer.3.']
    group2=['layer.4.','layer.5.','layer.6.','layer.7.']
    group3=['layer.8.','layer.9.','layer.10.','layer.11.']
    group_all=['layer.0.','layer.1.','layer.2.','layer.3.','layer.4.','layer.5.','layer.6.','layer.7.','layer.8.','layer.9.','layer.10.','layer.11.']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': args.weight_decay, 'lr': args.learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': args.weight_decay, 'lr': args.learning_rate*2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],'weight_decay': 0.0},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],'weight_decay': 0.0, 'lr': args.learning_rate/2.6},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],'weight_decay': 0.0, 'lr': args.learning_rate},
        {'params': [p for n, p in model.roberta.named_parameters() if any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],'weight_decay': 0.0, 'lr': args.learning_rate*2.6},
        {'params': [p for n, p in model.named_parameters() if args.model_type not in n], 'lr':args.learning_rate*20, "weight_decay": 0.0},
    ]
    return optimizer_grouped_parameters