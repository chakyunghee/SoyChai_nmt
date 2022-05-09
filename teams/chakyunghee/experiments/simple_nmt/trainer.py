import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MaximumLikelihoodEstimationEngine(Engine):    # teacher forcing으로 모델 학습

    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config =config

        super().__init__(func)

        self.best_loss = np.inf
        self.scaler = GradScaler()          ## amp. autocast. scaler

    @staticmethod                           # process function(ignite에 등록)
    def train(engine, mini_batch):          # gradient accumulation(마치 배치사이즈 큰 것처럼)
        engine.model.train()                # iteration_per_update == 32, 32로 나눈 나머지가 1일 때마다의 iteration에서 zero_grad()
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
            engine.config.iteration_per_update == 1:    # <-- 매번 업뎃
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()            

        device = next(engine.model.parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])  # torchtext로부터.
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]         # 정답인 애, <bos> 뺀 애, <eos> 포함

        with autocast(not engine.config.off_autocast):          ## amp적용 시작   
            # runs the forward pass with autocasting
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])  ## (fp16) 모델에 넣어줌, <eos> 뺌
            loss = engine.crit(                                 ## (fp16) 화면 출력 loss, 여기서 정답과 y_hat비교
                y_hat.contiguous().view(-1, y_hat.size(-1)),    ## output
                y.contiguous().view(-1)                         ## target
            )                                                   
            backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)   # gradient descent 수행 back porp할 애.
                                                                                            # train.py 에 crit의 reduction=sum이라 나눠주기                                  
        if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            engine.scaler.scale(backward_target).backward()     ## loss scaling (fp16범위내로) 후 backward 호출
        else:
            backward_target.backward()

        word_count = int(mini_batch.tgt[1].sum())               # sample별 length의 sum

        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
            engine.state.iteration > 0:                         # iteration_per_update==32 32의 배수마다 gradient update

            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm
            )
            if engine.config.gpu_id >= 0 and engine.config.off_autocast:
                engine.scaler.step(engine.optimizer)            ## optimizer가 step할 때에도 scaler가 step
                engine.scaler.update()                          ## 다음번에 어떻게 scaling할지 업뎃
            else:
                engine.optimizer.step()

#            if engine.config.use_noam_decay and engine.lr_scheduler is not None:    # transformer
#                engine.lr_scheduler.step()

        loss = float(loss/word_count)       # 단어당 loss
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl
            }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]

            with autocast(not engine.config.off_autocast):
                y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
                loss = engine.crit(
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1)
                )

        word_count =  int(mini_batch.tgt[1].sum())
        loss = float(loss/word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl
        }

    @staticmethod
    def attach(
        train_engine, validation_engine,
        training_metric_names = ['loss', 'ppl'],
        validation_metric_names = ['loss', 'ppl'],
        verbose=VERBOSE_BATCH_WISE
    ):
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )
        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_loss,
                    np.exp(avg_loss)
                ))

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss)
                ))
    
    @staticmethod
    def resume_training(engine, resume_epoch):          # 학습 끊기고 난 뒤 재개할 때
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)
    
    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab): # PPL 낮더라도 번역 품질이 보장되는건 아니므로 모든 epochs 결과물 다 저장해놓음
        avg_train_loss = train_engine.state.metrics['loss']             # 나중에 evaluation할 때 보려고.
        avg_valid_loss = engine.state.metrics['loss']

        model_fn = config.model_fn.split('.')
        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss))] + [model_fn[-1]]
        model_fn = '.'.join(model_fn)
        
        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab
            }, model_fn
        )

class SingleTrainer():

    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        train_engine = self.target_engine_class(
            self.target_engine_class.train,     # 등록
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.validate,  # 등록
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            config=self.config
        )
        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)       

            if engine.lr_scheduler is not None and not engine.config.use_noam_decay:
                engine.lr_scheduler.step()
        
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine,
            valid_loader
        )
        train_engine.add_event_handler(
            Events.STARTED,                             # 재개하는 애 등록
            self.target_engine_class.resume_training,
            self.config.init_epoch
        )

        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.target_engine_class.check_best
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.target_engine_class.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab,
        )
        
        train_engine.run(train_loader, max_epochs=n_epochs)

        return model