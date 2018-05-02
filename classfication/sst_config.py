class Config(object):
    vocab_size=15000
    max_grad_norm = 5
    init_scale = 0.05
    hidden_size = 300
    lr_decay = 0.95
    valid_portion=0.0
    batch_size=5
    keep_prob = 0.5
    #0.05
    learning_rate = 0.001
    max_epoch =2
    max_max_epoch =40
    num_label=5
    attention_iteration=3
    random_initialize=False
    embedding_trainable=True
    l2_beta=0.0