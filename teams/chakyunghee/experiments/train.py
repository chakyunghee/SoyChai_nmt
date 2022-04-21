# pip install torch-optimizer

import argparse
import pprint

import torch
#from torch import optim
import torch.nn as nn

import torch_optimizer as custom_optim

from simple_nmt.model.transformer import Transformer

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.trainer import SingleTrainer
from simple_nmt.trainer import MaximumLikelihoodEstimationEngine



def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument('--load_fn', required=True, help='Model file name to continue.')
    p.add_argument('--model_fn', required=not is_continue, help='Model file name to save. Additional information would be annotated to the file name.')

    p.add_argument('--train', required=not is_continue, help='Training set file name except the extention. (ex: train.en --> train)')
    p.add_argument('--valid', required=not is_continue, help='Validation set file name except the extention. (ex: valid.en --> valid)')

    p.add_argument('--lang', required=not is_continue, help='Set of extention represents language pair. (ex: en + ko --> enko)')
    p.add_argument('--gpu_id', type=int, default=-1, help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s')

    p.add_argument('--off_autocast', action='store_true', help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.')
    p.add_argument('--batch_size', type=int, default=32, help='Mini batch size for gradient descent. Default=%(default)s')

    p.add_argument('--n_epochs', type=int, default=20, help='Number of epochs to train. Default=%(default)s')
    p.add_argument('--verbose', type=int, default=2, help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s')

    p.add_argument('--init_epoch', required=is_continue, type=int, default=1, help='Set initial epoch number, which can be useful in continue training. Default=%(default)s')
    p.add_argument('--max_length', type=int, default=100, help='Maximum length of the training sequence. Default=%(default)s')

    p.add_argument('--dropout', type=float, default=.2, help='Dropout rate. Default=%(default)s')
    p.add_argument('--hidden_size', type=int, default=768, help='Hidden size of Model. Default=%(default)s')
    p.add_argument('--n_layers', type=int, default=4, help='Number of layers in Model. Default=%(default)s')    

    p.add_argument('--max_grad_norm', type=float, default=5., help='Threshold for gradient clipping in case of SGD, default 1e+8 using (r)adam meaning no need to use max_grad_norm. Default=%(default)s')
    p.add_argument('--iteration_per_update', type=int, default=1, help='Number of feed-forward iterations for one parameter update. Default=%(default)s')

    p.add_argument('--lr', type=float, default=1., help='Initial learning rate. Default=%(default)s')
    
#    p.add_argument('--lr_step', type=int, default=1, help='Number of epochs for each learning rate decay. Default=%(default)s')
#    p.add_argument('--lr_gamma', type=float, default=5., help='Learning rate decay rate. Default=%(default)s')
#    p.add_argument('--lr_decay_start', type=int, default=10, help='Learning rate decay start at. Default=%(default)s')


    p.add_argument('--use_radam', action='store_true', help='Use rectified Adam as optimizer. Other lr arguments should be changed.')
    p.add_argument('--use_transformer', action='store_true', help='Set model architecture as Transformer.')
    p.add_argument('--n_splits', type=int, default=8, help='Number of heads in multi-head attentionin Transformer. Default=%(default)s')

    config = p.parse_args()

    return config



def get_model(input_size, output_size, config):
    if config.use_transformer:
        model = Transformer(
            input_size,                     # src lang vocab size
            config.hidden_size,
            output_size,                    # tgt lang vocab size
            n_splits=config.n_splits,       # 8, config
            n_enc_blocks=config.n_layers,   # 4, config
            n_dec_blocks=config.n_layers,   # 4, config
            dropout_p=config.dropout,       # .2, config
        )
    else:
        raise Exception('Set model architecture as Transformer.')

    return model
        


def get_crit(output_size, pad_index):
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )
    return crit



def get_optimizer(model, config):
    optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr)    # lr = 1e-3

    return optimizer


# SGD쓰는 경우는 해주어야. (adam의 경우 안써도 되는 걸로)
#def get_scheduler(optimizer, config):       
#    if config.lr_step > 0:
#        lr_scheduler = optim.lr_scheduler.MultiStepLR(
#            optimizer,
#            milestones=[i for i in range(
#                max(0, config.lr_decay_start - 1),              # lr_decay_start 부터.
#                (config.init_epoch - 1) + config.n_epochs,
#                config.lr_step
#            )],
#            gamma=config.lr_gamma,                             # lr_gamma: decay ratio
#            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
#        )
#    else:
#        lr_scheduler = None
#
#    return lr_scheduler



def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,
        config.valid, 
        (config.lang[:2], config.lang[-2:]),
        batch_size=config.batch_size,
        device=-1,
        max_length=config.max_length
    )

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD)   # data_loader.PAD: pad에 weight주지 않기

    # model_weight=None, opt_weight=None
    if model_weight is not None:                    # 학습이 중단된 경우 재개할 때 사용 (이전 모델 weight 이어받아 학습 재개)
        model.load_state_dict(model_weight)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    optimizer = get_optimizer(model, config)

    if opt_weight is not None and config.use_radam:
        optimizer.load_state_dict(opt_weight)

    #lr_scheduler = get_scheduler(optimizer, config)  # adam은 안쓰고 SGD만 쓸.

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)


    mle_trainer = SingleTrainer(MaximumLikelihoodEstimationEngine, config)
    mle_trainer.train(
        model,
        crit,
        optimizer,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=config.n_epochs,
#        lr_scheduler=lr_scheduler,
    )


if __name__ == '__main__':      # 밖에서 console을 진행하면 여기서 진입.한다고..?
    config = define_argparser() # define_argparser()함수 거치면 config객체가 나오고
    main(config) 