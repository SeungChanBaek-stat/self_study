import argparse
import datetime
import os
import sys
import random

import numpy as np

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from tqdm import tqdm

import sys, os
sys.path.append(os.getcwd())
from functions.preprocess import LunaDataset
from functions.util import logging
from functions.model import LunaModel

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)

# Used for computeBatchLoss and logMetrics to index into metrics_t/metrics_a
METRICS_LABEL_NDX = 0
METRICS_PRED_NDX = 1
METRICS_LOSS_NDX = 2
METRICS_SIZE = 3


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--num-workers",
            help="Number of worker processes for background data loading",
            default=8,
            type=int,
        )
        parser.add_argument(
            "--batch-size",
            help="Batch size to use for training",
            default=32,
            type=int,
        )
        parser.add_argument(
            "--epochs",
            help="Number of epochs to train for",
            default=1,
            type=int,
        )

        parser.add_argument(
            "--tb-prefix",
            default="p2ch13",
            help="Data prefix to use for Tensorboard run. Defaults to chapter.",
        )

        parser.add_argument(
            "comment",
            help="Comment suffix for Tensorboard run.",
            nargs="?",
            default="dwlpt",
        )
        
        parser.add_argument(
            "--save-path",
            default="luna_best.pt",
            help="Checkpoint filename for best model",
        )
        
        parser.add_argument(
            "--gpu-ids",
            default=None,
            help='Comma-separated GPU ids to use, e.g. "0" or "0,1,2,3". Default: all visible GPUs.',
        )
        
        self.cli_args = parser.parse_args(sys_argv)
        
        if self.cli_args.gpu_ids is None:
            self.gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        else:
            self.gpu_ids = [int(x.strip()) for x in self.cli_args.gpu_ids.split(",") if x.strip() != ""]

        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0
        self.best_val_loss = float("inf")

        self.use_cuda = torch.cuda.is_available() and len(self.gpu_ids) > 0
        self.device = torch.device(f"cuda:{self.gpu_ids[0]}" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()

        if self.use_cuda:
            log.info(f"Using CUDA; gpu_ids={self.gpu_ids}")
            model = model.to(self.device)

            if len(self.gpu_ids) > 1:
                model = nn.DataParallel(model, device_ids=self.gpu_ids, output_device=self.gpu_ids[0])

        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())

    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= len(self.gpu_ids)

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=(self.cli_args.num_workers > 0),
            prefetch_factor=2 if self.cli_args.num_workers > 0 else None,
        )

        return train_dl

    def get_dl_subset(self, dataloader, num_samples=100):
        dataset = dataloader.dataset
        subset_indices = random.sample(range(len(dataset)), num_samples)
        train_subset = Subset(dataset, subset_indices)
        dl_subset = torch.utils.data.DataLoader(
            train_subset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=dataloader.num_workers
        )
        return dl_subset

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= len(self.gpu_ids)

        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=(self.cli_args.num_workers > 0),
            prefetch_factor=2 if self.cli_args.num_workers > 0 else None,
        )

        return val_dl

    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join("runs", self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=f"{log_dir}-trn_cls-{self.cli_args.comment}"
            )
            self.val_writer = SummaryWriter(
                log_dir=f"{log_dir}-val_cls-{self.cli_args.comment}"
            )

    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        train_dl = self.initTrainDl()
        val_dl = self.initValDl()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "models")

        for epoch_ndx in range(1, self.cli_args.epochs + 1):

            log.info(
                "Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch_ndx,
                    self.cli_args.epochs,
                    len(train_dl),
                    len(val_dl),
                    self.cli_args.batch_size,
                    (len(self.gpu_ids) if self.use_cuda else 1),
                )
            )

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, "trn", trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, "val", valMetrics_t)
            
            val_loss = valMetrics_t[METRICS_LOSS_NDX].mean().item()
            if val_loss <= self.best_val_loss:
                self.best_val_loss = val_loss
                if isinstance(self.model, (nn.DataParallel, DDP)):
                    save_obj = self.model.module.state_dict()
                else:
                    save_obj = self.model.state_dict()
                

                os.makedirs(model_dir, exist_ok=True)
                model_save_path = os.path.join(model_dir, self.cli_args.save_path)                    

                torch.save(save_obj, model_save_path)
                log.info(f"Saved new best checkpoint to {self.cli_args.save_path}")
                
        model_save_path = os.path.join(model_dir, "luna_final")                    

        torch.save(save_obj, model_save_path)
        log.info(f"Saved final checkpoint to {model_save_path}")

        if hasattr(self, "trn_writer"):
            self.trn_writer.close()
            self.val_writer.close()

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        train_progress = tqdm(train_dl, desc="E{} Training".format(epoch_ndx), total=len(train_dl))
        for batch_ndx, batch_tup in enumerate(train_progress):
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(
                batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g
            )

            loss_var.backward()
            self.optimizer.step()

            # # This is for adding the model graph to TensorBoard.
            # if epoch_ndx == 1 and batch_ndx == 0:
            #     with torch.no_grad():
            #         model = LunaModel()
            #         self.trn_writer.add_graph(model, batch_tup[0], verbose=True)
            #         self.trn_writer.close()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return trnMetrics_g.to("cpu")

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            val_progress = tqdm(val_dl, desc="E{} Validation".format(epoch_ndx), total=len(val_dl))
            for batch_ndx, batch_tup in enumerate(val_progress):
                self.computeBatchLoss(
                    batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g
                )

        return valMetrics_g.to("cpu")

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        input_t : torch.Tensor
        label_t : torch.Tensor
        
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)
        logits_g : torch.Tensor
        probability_g : torch.Tensor

        loss_func = nn.CrossEntropyLoss(reduction="none")   # reduction = 'none' 으로 샘플별 손실값을 얻는다.
        loss_g : torch.Tensor = loss_func(
            logits_g,
            label_g[:, 1],  # 원핫 인코딩 클래스의 인덱스 :LunaDataset의 getitem의 2번째 리턴값인 
            # pos_t에서, 2번째 성분인 candidateInfo_tup.isNodule_bool 에 해당함. 즉 이 값이 1이면 결절, 0이면 노 결절
        )
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = label_g[:, 1].detach()
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = probability_g[:, 1].detach()
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g.detach()

        return loss_g.mean()

    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t,
        classificationThreshold=0.5,
    ):
        self.initTensorboardWriters()
        log.info(
            "E{} {}".format(
                epoch_ndx,
                type(self).__name__,
            )
        )

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold   # 결절이면 label 값 1이고, clsthres=0.5보다 커지므로 False
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict = {}
        metrics_dict["loss/all"] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict["loss/neg"] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict["loss/pos"] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict["correct/all"] = (
            (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
        )
        metrics_dict["correct/neg"] = neg_correct / np.float32(neg_count) * 100
        try:
            metrics_dict["correct/pos"] = pos_correct / np.float32(pos_count) * 100
        except ZeroDivisionError:
            metrics_dict["correct/pos"] = 0

        log.info(
            f"E{epoch_ndx} {mode_str:8} {metrics_dict['loss/all']:.4f} loss, "
            f"{metrics_dict['correct/all']:-5.1f}% correct"
        )
        log.info(
            f"E{epoch_ndx} {mode_str + '_neg':8} {metrics_dict['loss/neg']:.4f} loss, "
            f"{metrics_dict['correct/neg']:-5.1f}% correct ({neg_correct} of {neg_count})"
        )
        log.info(
            f"E{epoch_ndx} {mode_str + '_pos':8} {metrics_dict['loss/pos']:.4f} loss, "
            f"{metrics_dict['correct/pos']:-5.1f}% correct ({pos_correct} of {pos_count})"
        )

        writer = getattr(self, mode_str + "_writer")

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        writer.add_pr_curve(
            "pr",
            metrics_t[METRICS_LABEL_NDX],
            metrics_t[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        )

        bins = [x / 50.0 for x in range(51)]

        negHist_mask = negLabel_mask & (metrics_t[METRICS_PRED_NDX] > 0.01)
        posHist_mask = posLabel_mask & (metrics_t[METRICS_PRED_NDX] < 0.99)

        if negHist_mask.any():
            writer.add_histogram(
                "is_neg",
                metrics_t[METRICS_PRED_NDX, negHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if posHist_mask.any():
            writer.add_histogram(
                "is_pos",
                metrics_t[METRICS_PRED_NDX, posHist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )

if __name__ == "__main__":
    LunaTrainingApp().main()