import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import ModelCheckpoint, EarlyStopping
from pytorch_pretrained_bert import BertAdam
from dataset import TreeDataSet, SeqDataSet, collate_fn
from torch.utils.data import DataLoader
from model.module import make_model, Train, GreedyEvaluate
from train_utils import LabelSmoothing, BLEU4, MyLoss, Rouge, Meteor
from ignite.contrib.handlers.tensorboard_logger import *


class Solver:
    def __init__(self, args, ast2id, code2id, nl2id, id2nl):
        self.args = args

        self.ast2id = ast2id
        self.code2id = code2id
        self.nl2id = nl2id
        self.id2nl = id2nl

        self.nl_vocab_size = len(self.nl2id)

        if self.args.model in ['ast-transformer', 'sbt-transformer']:
            code_len = len(self.ast2id)
        else:
            code_len = len(self.code2id)

        self.model = make_model(code_vocab=code_len,
                                nl_vocab=len(self.nl2id),
                                N=self.args.num_layers,
                                d_model=self.args.model_dim,
                                d_ff=self.args.ffn_dim,
                                k=self.args.k,
                                h=self.args.num_heads,
                                dropout=self.args.dropout)

    def train(self):
        if self.args.model in ['ast-transformer', 'sbt-transformer']:
            use_relative = True if self.args.model == 'ast-transformer' else False
            train_data_set = TreeDataSet(file_name=self.args.data_dir + '/train.json',
                                         ast_path=self.args.data_dir + '/tree/train/',
                                         ast2id=self.ast2id,
                                         nl2id=self.nl2id,
                                         max_ast_size=self.args.code_max_len,
                                         k=self.args.k,
                                         max_comment_size=self.args.comment_max_len,
                                         use_code=use_relative)
            valid_data_set = TreeDataSet(file_name=self.args.data_dir + '/valid.json',
                                         ast_path=self.args.data_dir + '/tree/valid/',
                                         ast2id=self.ast2id,
                                         nl2id=self.nl2id,
                                         max_ast_size=self.args.code_max_len,
                                         k=self.args.k,
                                         max_comment_size=self.args.comment_max_len,
                                         use_code=use_relative)
        elif self.args.model in ['transformer']:
            train_data_set = SeqDataSet(file_name=self.args.data_dir + '/train.json',
                                        code2id=self.code2id,
                                        nl2id=self.nl2id,
                                        max_code_size=self.args.code_max_len,
                                        max_comment_size=self.args.comment_max_len)
            valid_data_set = SeqDataSet(file_name=self.args.data_dir + '/valid.json',
                                        code2id=self.code2id,
                                        nl2id=self.nl2id,
                                        max_code_size=self.args.code_max_len,
                                        max_comment_size=self.args.comment_max_len)

        train_loader = DataLoader(dataset=train_data_set,
                                  batch_size=self.args.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=valid_data_set,
                                  batch_size=self.args.batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn)

        device = "cpu"

        if torch.cuda.is_available():
            device = "cuda"
            print('use gpu')

        model_opt = BertAdam(self.model.parameters(), lr=1e-4)
        criterion = LabelSmoothing(size=self.nl_vocab_size,
                                   padding_idx=0, smoothing=0.1)

        train_model = Train(self.model)
        greedy_evaluator = GreedyEvaluate(self.model, self.args.comment_max_len, self.nl2id['<s>'])

        trainer = create_supervised_trainer(train_model, model_opt, criterion, device)

        metric_train = {"loss": MyLoss(self.nl_vocab_size)}
        metric_valid = {"bleu": BLEU4(self.id2nl), "rouge": Rouge(self.id2nl), "meteor": Meteor(self.id2nl)}

        pbar = ProgressBar(persist=True)
        pbar.attach(trainer, metric_names="all")
        """
        train + generator
        validation + greedy_decode
        """
        train_evaluator = create_supervised_evaluator(train_model, metric_train, device)
        validation_evaluator = create_supervised_evaluator(greedy_evaluator, metric_valid, device)

        # save model
        save_handler = ModelCheckpoint('checkpoint/' + self.args.model, n_saved=5,
                                       filename_prefix='',
                                       create_dir=True,
                                       global_step_transform=lambda e, _: e.state.epoch)
        trainer.add_event_handler(Events.EPOCH_COMPLETED(every=2), save_handler, {self.args.model: self.model})

        # early stop
        early_stop_handler = EarlyStopping(patience=10, score_function=self.score_function, trainer=trainer)
        validation_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            train_evaluator.run(train_loader)

            train_metrics = train_evaluator.state.metrics
            loss = train_metrics["loss"]

            pbar.log_message(
                "Training Results - Epoch: {} Avg loss: {:.2f}".format(
                    engine.state.epoch, loss
                )
            )

            validation_evaluator.run(valid_loader)

            valid_metrics = validation_evaluator.state.metrics
            bleu = valid_metrics["bleu"]
            rouge = valid_metrics["rouge"]
            meteor = valid_metrics["meteor"]

            pbar.log_message(
                "Validation Results - Epoch: {} BLEU: {:.2f} ROUGE: {:.2f} METEOR: {:.2f}".format(
                    engine.state.epoch, bleu, rouge, meteor
                )
            )

            pbar.n = pbar.last_print_n = 0

        tb_logger = TensorboardLogger(self.args.log_dir + self.args.model + '/')

        tb_logger.attach(
            train_evaluator,
            log_handler=OutputHandler(
                tag="training", metric_names=["loss"], another_engine=trainer),
            event_name=Events.EPOCH_COMPLETED,
        )

        tb_logger.attach(
            validation_evaluator,
            log_handler=OutputHandler(tag="validation", metric_names=["bleu", "rouge", "meteor"], another_engine=trainer),
            event_name=Events.EPOCH_COMPLETED,
        )

        trainer.run(train_loader, max_epochs=self.args.num_step)
        tb_logger.close()

    @staticmethod
    def score_function(engine):
        bleu = engine.state.metrics['bleu']
        return bleu
