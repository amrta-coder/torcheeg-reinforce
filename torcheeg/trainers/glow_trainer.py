import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torchmetrics
from itertools import chain
from torch.utils.data import DataLoader

from .basic_trainer import BasicTrainer


class GlowTrainer(BasicTrainer):
    r'''
    A generic trainer class for EEG classification.

    .. code-block:: python

        trainer = GlowTrainer(generator, discriminator)
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    The class provides the following hook functions for inserting additional implementations in the training, validation and testing lifecycle:

    - :obj:`before_training_epoch`: executed before each epoch of training starts
    - :obj:`before_training_step`: executed before each batch of training starts
    - :obj:`on_training_step`: the training process for each batch
    - :obj:`after_training_step`: execute after the training of each batch
    - :obj:`after_training_epoch`: executed after each epoch of training
    - :obj:`before_validation_epoch`: executed before each round of validation starts
    - :obj:`before_validation_step`: executed before the validation of each batch
    - :obj:`on_validation_step`: validation process for each batch
    - :obj:`after_validation_step`: executed after the validation of each batch
    - :obj:`after_validation_epoch`: executed after each round of validation
    - :obj:`before_test_epoch`: executed before each round of test starts
    - :obj:`before_test_step`: executed before the test of each batch
    - :obj:`on_test_step`: test process for each batch
    - :obj:`after_test_step`: executed after the test of each batch
    - :obj:`after_test_epoch`: executed after each round of test

    If you want to customize some operations, you just need to inherit the class and override the hook function:

    .. code-block:: python

        class MyGlowTrainer(GlowTrainer):
            def before_training_epoch(self, epoch_id: int, num_epochs: int):
                # Do something here.
                super().before_training_epoch(epoch_id, num_epochs)
    
    If you want to use multiple GPUs for parallel computing, you need to specify the GPU indices you want to use in the python file:
    
    .. code-block:: python

        trainer = GlowTrainer(generator, discriminator, device_ids=[1, 2, 7])
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader)

    Then, you can use the :obj:`torch.distributed.launch` or :obj:`torchrun` to run your python file.

    .. code-block:: shell

        python -m torch.distributed.launch \
            --nproc_per_node=3 \
            --nnodes=1 \
            --node_rank=0 \
            --master_addr="localhost" \
            --master_port=2345 \
            your_python_file.py

    Here, :obj:`nproc_per_node` is the number of GPUs you specify.

    Args:
        encoder (nn.Module): The encoder, whose inputs are EEG signals, outputs are two batches of vectors of the same dimension, representing the mean and variance estimated in the reparameterization trick.
        decoder (nn.Module): The decoder generating EEG signals from hidden variables encoded by the encoder. The dimensions of the input vector should be defined on the :obj:`in_channel` attribute.
        lr (float): The learning rate. (defualt: :obj:`0.0001`)
        weight_decay: (float): The weight decay (L2 penalty). (defualt: :obj:`0.0`)
        beta: (float): The weight of the KL divergence in the loss function. Please refer to betaGlow. (defualt: :obj:`1.0`)
        device_ids (list): Use cpu if the list is empty. If the list contains indices of multiple GPUs, it needs to be launched with :obj:`torch.distributed.launch` or :obj:`torchrun`. (defualt: :obj:`[]`)
        ddp_sync_bn (bool): Whether to replace batch normalization in network structure with cross-GPU synchronized batch normalization. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_replace_sampler (bool): Whether to replace sampler in dataloader with :obj:`DistributedSampler`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_val (bool): Whether to use multi-GPU acceleration for the validation set. For experiments where data input order is sensitive, :obj:`ddp_val` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
        ddp_test (bool): Whether to use multi-GPU acceleration for the test set. For experiments where data input order is sensitive, :obj:`ddp_test` should be set to :obj:`False`. Only valid when the length of :obj:`device_ids` is greater than one. (defualt: :obj:`True`)
    
    .. automethod:: fit
    .. automethod:: test
    .. automethod:: sample
    '''
    def __init__(self,
                 glow: nn.Module,
                 lr: float = 1e-4,
                 device_ids: List[int] = [],
                 ddp_sync_bn: bool = True,
                 ddp_replace_sampler: bool = True,
                 ddp_val: bool = True,
                 ddp_test: bool = True):
        super(GlowTrainer,
              self).__init__(modules={'glow': glow},
                             device_ids=device_ids,
                             ddp_sync_bn=ddp_sync_bn,
                             ddp_replace_sampler=ddp_replace_sampler,
                             ddp_val=ddp_val,
                             ddp_test=ddp_test)
        self.lr = lr

        self.optimizer = torch.optim.Adam(glow.parameters(), lr=lr)

        # init metric
        self.train_loss = torchmetrics.MeanMetric().to(self.device)
        self.val_loss = torchmetrics.MeanMetric().to(self.device)
        self.test_loss = torchmetrics.MeanMetric().to(self.device)

    def before_training_epoch(self, epoch_id: int, num_epochs: int):
        self.log(f"Epoch {epoch_id}\n-------------------------------")

    def log_prob(self, value, loc=0.0, scale=1.0):
        var = (scale**2)
        log_scale = math.log(scale)
        return -((value - loc)**2) / (2 * var) - log_scale - math.log(
            math.sqrt(2 * math.pi))

    def on_training_step(self, train_batch: Tuple, batch_id: int,
                         num_batches: int):
        self.train_loss.reset()

        X = train_batch[0].to(self.device)
        y = train_batch[1].to(self.device)

        self.optimizer.zero_grad()

        zs, logdet = self.modules['glow'].forward(X)

        log_prob_list = []
        for z in zs:
            log_prob = self.log_prob(z)
            log_prob_list.append(log_prob.sum([1, 2, 3]))

        log_prob_loss = -(sum(log_prob_list) + logdet)
        log_prob_loss /= (math.log(2) * X[0].numel())
        loss = log_prob_loss.mean(0)

        loss.backward()
        self.optimizer.step()

        # log five times
        log_step = math.ceil(num_batches / 5)
        if batch_id % log_step == 0:
            self.train_loss.update(loss)

            train_loss = self.train_loss.compute()

            # if not distributed, world_size is 1
            batch_id = batch_id * self.world_size
            num_batches = num_batches * self.world_size
            if self.is_main:
                self.log(
                    f"loss: {train_loss:>8f} [{batch_id:>5d}/{num_batches:>5d}]"
                )

    def before_validation_epoch(self, epoch_id: int, num_epochs: int):
        self.val_loss.reset()

    def on_validation_step(self, val_batch: Tuple, batch_id: int,
                           num_batches: int):
        X = val_batch[0].to(self.device)
        y = val_batch[1].to(self.device)

        zs, logdet = self.modules['glow'].forward(X)

        log_prob_list = []
        for z in zs:
            log_prob = self.log_prob(z)
            log_prob_list.append(log_prob.sum([1, 2, 3]))

        log_prob_loss = -(sum(log_prob_list) + logdet)
        log_prob_loss /= (math.log(2) * X[0].numel())
        loss = log_prob_loss.mean(0)

        self.val_loss.update(loss)

    def after_validation_epoch(self, epoch_id: int, num_epochs: int):
        val_loss = self.val_loss.compute()
        self.log(f"\nloss: {val_loss:>8f}")

    def before_test_epoch(self):
        self.test_loss.reset()

    def on_test_step(self, test_batch: Tuple, batch_id: int, num_batches: int):
        X = test_batch[0].to(self.device)
        y = test_batch[1].to(self.device)

        zs, logdet = self.modules['glow'].forward(X)

        log_prob_list = []
        for z in zs:
            log_prob = self.log_prob(z)
            log_prob_list.append(log_prob.sum([1, 2, 3]))

        log_prob_loss = -(sum(log_prob_list) + logdet)
        log_prob_loss /= (math.log(2) * X[0].numel())
        loss = log_prob_loss.mean(0)

        self.test_loss.update(loss)

    def after_test_epoch(self):
        test_loss = self.test_loss.compute()
        self.log(f"\nloss: {test_loss:>8f}")

    def test(self, test_loader: DataLoader):
        r'''
        Validate the performance of the model on the test set.

        Args:
            test_loader (DataLoader): Iterable DataLoader for traversing the test data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
        '''
        super().test(test_loader=test_loader)

    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int = 1):
        r'''
        Train the model on the training set and use the validation set to validate the results of each round of training.

        Args:
            train_loader (DataLoader): Iterable DataLoader for traversing the training data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            val_loader (DataLoader): Iterable DataLoader for traversing the validation data batch (torch.utils.data.dataloader.DataLoader, torch_geometric.loader.DataLoader, etc).
            num_epochs (int): training epochs. (defualt: :obj:`1`)
        '''
        super().fit(train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs)

    def sample(self, num_samples: int, z_std: float = 0.6) -> torch.Tensor:
        """
        Samples from the latent space and return generated results.

        Args:
            num_samples (int): Number of samples.

        Returns:
            torch.Tensor: the generated samples.
        """
        self.modules['glow'].eval()
        with torch.no_grad():
            samples, _ = self.modules['glow'].inverse(batch_size=num_samples,
                                                      z_std=z_std)
            return samples