import argparse
import datetime
import os
import sys
import random

import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append(os.getcwd())
from functions.preprocess import LunaDataset
from functions.util import logging
from functions.model import LunaModel


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

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
            help="Number of worker processes for background data loading (per process)",
            default=4,
            type=int,
        )
        parser.add_argument(
            "--batch-size",
            help="Batch size to use per GPU/process",
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
            "--seed",
            default=1234,
            type=int,
            help="Random seed",
        )
        parser.add_argument(
            "--save-path",
            default="luna_ddp_best.pt",
            help="Checkpoint filename for best model",
        )
        parser.add_argument(
            "comment",
            help="Comment suffix for Tensorboard run.",
            nargs="?",
            default="dwlpt_ddp",
        )

        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")

        self.trn_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0
        self.best_val_loss = float("inf")

        self._init_distributed()
        self._set_seed(self.cli_args.seed + self.rank)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    # -------------------------
    # Distributed setup
    # -------------------------
    def _init_distributed(self):
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])

            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)

            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1

    def cleanup(self):
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()

    @property
    def is_main_process(self):
        return self.rank == 0

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # -------------------------
    # Init model / optimizer
    # -------------------------
    def initModel(self):
        model = LunaModel()
        model = model.to(self.device)

        if self.distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank] if self.use_cuda else None,
                output_device=self.local_rank if self.use_cuda else None,
            )

        if self.is_main_process:
            log.info(
                f"Using {'DDP' if self.distributed else 'single process'}; "
                f"world_size={self.world_size}, local_rank={self.local_rank}, device={self.device}"
            )

        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())

    # -------------------------
    # Dataloaders
    # -------------------------
    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False,
        )

        train_sampler = None
        if self.distributed:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )

        train_dl = DataLoader(
            train_ds,
            batch_size=self.cli_args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=(self.cli_args.num_workers > 0),
            prefetch_factor=2 if self.cli_args.num_workers > 0 else None,
        )

        return train_dl, train_sampler

    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True,
        )

        val_sampler = None
        if self.distributed:
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

        val_dl = DataLoader(
            val_ds,
            batch_size=self.cli_args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=(self.cli_args.num_workers > 0),
            prefetch_factor=2 if self.cli_args.num_workers > 0 else None,
        )

        return val_dl, val_sampler

    def get_dl_subset(self, dataloader, num_samples=100):
        dataset = dataloader.dataset
        subset_indices = random.sample(range(len(dataset)), num_samples)
        subset_ds = Subset(dataset, subset_indices)

        subset_sampler = None
        if self.distributed:
            subset_sampler = DistributedSampler(
                subset_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

        dl_subset = DataLoader(
            subset_ds,
            batch_size=dataloader.batch_size,
            shuffle=(subset_sampler is None),
            sampler=subset_sampler,
            num_workers=dataloader.num_workers,
            pin_memory=self.use_cuda,
            persistent_workers=(dataloader.num_workers > 0),
        )
        return dl_subset

    # -------------------------
    # TensorBoard
    # -------------------------
    def initTensorboardWriters(self):
        if not self.is_main_process:
            return

        if self.trn_writer is None:
            log_dir = os.path.join("runs", self.cli_args.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=f"{log_dir}-trn_cls-{self.cli_args.comment}"
            )
            self.val_writer = SummaryWriter(
                log_dir=f"{log_dir}-val_cls-{self.cli_args.comment}"
            )

    # -------------------------
    # Main
    # -------------------------
    def main(self):
        if self.is_main_process:
            log.info(f"Starting {type(self).__name__}, {self.cli_args}")

        train_dl, train_sampler = self.initTrainDl()
        val_dl, val_sampler = self.initValDl()

        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch_ndx)

            if self.is_main_process:
                log.info(
                    "Epoch {} of {}, train batches: {}, val batches: {}, batch size per GPU: {}, world_size: {}".format(
                        epoch_ndx,
                        self.cli_args.epochs,
                        len(train_dl),
                        len(val_dl),
                        self.cli_args.batch_size,
                        self.world_size,
                    )
                )

            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, "trn", trnMetrics_t)

            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            self.logMetrics(epoch_ndx, "val", valMetrics_t)

            # rank 0 only checkpoint
            if self.is_main_process:
                val_loss = valMetrics_t[METRICS_LOSS_NDX].mean().item()
                if val_loss <= self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_obj = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
                    
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    model_dir = os.path.join(base_dir, "models")
                    os.makedirs(model_dir, exist_ok=True)
                    model_save_path = os.path.join(model_dir, self.cli_args.save_path)                    

                    torch.save(save_obj, model_save_path)
                    log.info(f"Saved new best checkpoint to {self.cli_args.save_path}")

        if self.is_main_process and self.trn_writer is not None:
            self.trn_writer.close()
            self.val_writer.close()
            
            
    # -------------------------
    # Train / Val
    # -------------------------
    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()

        local_metrics = []

        iterable = train_dl
        if self.is_main_process:
            iterable = tqdm(
                train_dl,
                desc=f"E{epoch_ndx} Training",
                total=len(train_dl),
            )

        for batch_ndx, batch_tup in enumerate(iterable):
            self.optimizer.zero_grad(set_to_none=True)

            metrics_batch_t, loss_var = self.computeBatchLoss(batch_tup)

            loss_var.backward()
            self.optimizer.step()

            local_metrics.append(metrics_batch_t)

        metrics_local_t = self._stack_local_metrics(local_metrics)
        metrics_t = self._gather_metrics_across_ranks(metrics_local_t)

        if self.is_main_process:
            self.totalTrainingSamples_count += metrics_t.shape[1]

        return metrics_t

    def doValidation(self, epoch_ndx, val_dl):
        self.model.eval()

        local_metrics = []

        iterable = val_dl
        if self.is_main_process:
            iterable = tqdm(
                val_dl,
                desc=f"E{epoch_ndx} Validation",
                total=len(val_dl),
            )

        with torch.no_grad():
            for batch_ndx, batch_tup in enumerate(iterable):
                metrics_batch_t, _loss_var = self.computeBatchLoss(batch_tup)
                local_metrics.append(metrics_batch_t)

        metrics_local_t = self._stack_local_metrics(local_metrics)
        metrics_t = self._gather_metrics_across_ranks(metrics_local_t)

        return metrics_t

    # -------------------------
    # Metric helpers
    # -------------------------
    def computeBatchLoss(self, batch_tup):
        input_t, label_t, _series_list, _center_list = batch_tup
        input_t: torch.Tensor
        label_t: torch.Tensor

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        logits_g, probability_g = self.model(input_g)
        logits_g: torch.Tensor
        probability_g: torch.Tensor

        loss_func = nn.CrossEntropyLoss(reduction="none")
        loss_g: torch.Tensor = loss_func(
            logits_g,
            label_g[:, 1],
        )

        batch_metrics_t = torch.stack(
            [
                label_g[:, 1].detach().to(torch.float32),
                probability_g[:, 1].detach().to(torch.float32),
                loss_g.detach().to(torch.float32),
            ],
            dim=0,
        )  # shape: [3, batch]

        return batch_metrics_t, loss_g.mean()

    def _stack_local_metrics(self, metrics_list):
        if len(metrics_list) == 0:
            return torch.empty(METRICS_SIZE, 0, device=self.device, dtype=torch.float32)

        return torch.cat(metrics_list, dim=1)

    def _gather_metrics_across_ranks(self, local_metrics_t):
        """
        local_metrics_t: [METRICS_SIZE, local_n]
        returns:
            rank0 -> [METRICS_SIZE, global_n]
            others -> empty cpu tensor [METRICS_SIZE, 0]
        """
        local_n = torch.tensor([local_metrics_t.shape[1]], device=self.device, dtype=torch.long)

        if self.distributed:
            gathered_sizes = [torch.zeros_like(local_n) for _ in range(self.world_size)]
            dist.all_gather(gathered_sizes, local_n)
            sizes = [int(x.item()) for x in gathered_sizes]
            max_n = max(sizes)

            if local_metrics_t.shape[1] < max_n:
                pad = torch.zeros(
                    METRICS_SIZE,
                    max_n - local_metrics_t.shape[1],
                    device=self.device,
                    dtype=local_metrics_t.dtype,
                )
                padded_local = torch.cat([local_metrics_t, pad], dim=1)
            else:
                padded_local = local_metrics_t

            gather_list = [torch.zeros_like(padded_local) for _ in range(self.world_size)]
            dist.all_gather(gather_list, padded_local)

            if self.is_main_process:
                trimmed = []
                for rank_t, n in zip(gather_list, sizes):
                    trimmed.append(rank_t[:, :n])
                return torch.cat(trimmed, dim=1).cpu()
            else:
                return torch.empty(METRICS_SIZE, 0, dtype=torch.float32)
        else:
            return local_metrics_t.cpu()

    # -------------------------
    # Logging
    # -------------------------
    def logMetrics(
        self,
        epoch_ndx,
        mode_str,
        metrics_t,
        classificationThreshold=0.5,
    ):
        if not self.is_main_process:
            return

        self.initTensorboardWriters()

        log.info(
            "E{} {}".format(
                epoch_ndx,
                type(self).__name__,
            )
        )

        negLabel_mask = metrics_t[METRICS_LABEL_NDX] <= classificationThreshold
        negPred_mask = metrics_t[METRICS_PRED_NDX] <= classificationThreshold

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        metrics_dict = {}
        metrics_dict["loss/all"] = metrics_t[METRICS_LOSS_NDX].mean().item()

        if neg_count > 0:
            metrics_dict["loss/neg"] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean().item()
            metrics_dict["correct/neg"] = neg_correct / np.float32(neg_count) * 100
        else:
            metrics_dict["loss/neg"] = 0.0
            metrics_dict["correct/neg"] = 0.0

        if pos_count > 0:
            metrics_dict["loss/pos"] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean().item()
            metrics_dict["correct/pos"] = pos_correct / np.float32(pos_count) * 100
        else:
            metrics_dict["loss/pos"] = 0.0
            metrics_dict["correct/pos"] = 0.0

        metrics_dict["correct/all"] = (
            (pos_correct + neg_correct) / np.float32(metrics_t.shape[1]) * 100
            if metrics_t.shape[1] > 0 else 0.0
        )

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
    app = LunaTrainingApp()
    try:
        app.main()
    finally:
        app.cleanup()