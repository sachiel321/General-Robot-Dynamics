"""
Simple training loop
Author @ YYM CASIA
"""

import logging
import math
import os

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 2e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    local_rank = 0
    train_type = 'Dynamic'
    # checkpoint settings
    D_model_path = None
    Di_model_path = None
    ckpt_path = None
    test_loss_name = 'loss_store/test_loss.txt'
    num_workers = 8 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model.cuda()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.loss_store = np.zeros(config.max_epochs)

        state_dict = torch.load(self.config.D_model_path,map_location="cuda:%d" % self.config.local_rank)  # load_model
        self.model.load_state_dict(state_dict) 
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[self.config.local_rank], output_device=self.config.local_rank, find_unused_parameters=True)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            train_dataset = self.train_dataset
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_dataset = self.test_dataset
            train_iter = DataLoader(
                train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                pin_memory=True, shuffle=(train_sampler is None),
                sampler=train_sampler
            )
            # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            test_iter = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                   pin_memory=True, shuffle=True)
            if is_train:
                loader = train_iter
            else:
                loader = test_iter
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:
                # place data on the correct device
                x = x.cuda()  #dtype=torch.float64
                y = y.cuda()  #dtype=torch.float64
                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}")#. lr {lr:e}

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                self.loss_store[epoch] = test_loss
                np.savetxt(self.config.test_loss_name,self.loss_store)
            #save best model
            if self.config.local_rank == 0:
                if test_loss < best_loss :
                    best_loss=test_loss
                    self.save_checkpoint()

class TrainerDDi_DiD:

    def __init__(self, model,model_p2t, train_dataset, test_dataset, config):
        self.model = model.cuda()
        self.model_p2t = model_p2t.cuda()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.test_loss1_store = np.zeros(config.max_epochs)
        self.test_loss2_store = np.zeros(config.max_epochs)
        self.loss_store = np.zeros(config.max_epochs)

        model_temp = torch.load(self.config.D_model_path, map_location="cuda:%d" % self.config.local_rank)
        self.model.load_state_dict(model_temp)
        model_p2t_temp = torch.load(self.config.Di_model_path, map_location="cuda:%d" % self.config.local_rank)
        self.model_p2t.load_state_dict(model_p2t_temp)
        
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.config.local_rank],output_device=self.config.local_rank,find_unused_parameters=True)
        self.model_p2t = torch.nn.parallel.DistributedDataParallel(self.model_p2t, device_ids=[self.config.local_rank],output_device=self.config.local_rank,find_unused_parameters=True)


    def save_checkpoint(self):
        ckpt_model = self.model_p2t.module if hasattr(self.model_p2t, "module") else self.model_p2t
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(ckpt_model.state_dict(),self.config.ckpt_path)

    def train(self):
        model,model_p2t,config = self.model,self.model_p2t,self.config
        raw_model = model_p2t.module if hasattr(self.model_p2t, "module") else model_p2t
        optimizer = raw_model.configure_optimizers(config)

        def run_epochDDi(split):
            is_train = split == 'train'
            model_p2t.train(is_train)
            model.eval()
            train_dataset = self.train_dataset
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_dataset = self.test_dataset
            train_iter = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                    pin_memory=True, shuffle=(train_sampler is None),
                                    sampler=train_sampler) 
            test_iter = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                   pin_memory=True, shuffle=True)

            if is_train:
                loader = train_iter
            else:
                loader = test_iter
            a = torch.tensor([0, 1, 3, 4, 6, 7]).cuda()
            b = torch.tensor([2, 5, 8]).cuda()
            losses_1 = []
            losses_2 = []
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:
                # place data on the correct device
                x = x.cuda()  #[b,49,9]
                y = y.cuda()  #[b,49,9]
                y_tor = torch.index_select(y[:,:-1,:], 2, b)
                # forward the model
                with torch.no_grad():
                    logits, loss_1 = model(x, y)
                with torch.set_grad_enabled(is_train):
                    model_p2t_input = torch.cat((torch.index_select(x, 2, a),torch.index_select(logits, 2, a)),2)
                    tor,loss_2 = model_p2t(model_p2t_input,torch.index_select(x, 2, b))   #[b,49,3]

                    loss = loss_2
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                    loss_1 = loss_1.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses_1.append(loss_1.item())
                    loss_2 = loss_2.mean() # collapse all losses if they are scattered on multiple gpus
                    losses_2.append(loss_2.item())

                if is_train:
                    # backprop and update the parameters
                    model_p2t.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_p2t.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f} model loss {loss_1.item():.5f} model_p2t loss {loss_2.item():.5f}")#. lr {lr:e}
            if not is_train:
                test_loss_1 = float(np.mean(losses_1))
                test_loss_2 = float(np.mean(losses_2))
                loss = float(np.mean(losses))
                logger.info("model test loss: %f  model_p2t test loss: %f  sum loss: %f", test_loss_1,test_loss_2,loss)

                return test_loss_1,test_loss_2,loss
        
        def run_epochDiD(split):
            is_train = split == 'train'
            model_p2t.train(is_train)
            model.eval()
            train_dataset = self.train_dataset
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_dataset = self.test_dataset
            train_iter = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                    pin_memory=True, shuffle=(train_sampler is None),
                                    sampler=train_sampler) 

            test_iter = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers,
                                   pin_memory=True, shuffle=True)

            if is_train:
                loader = train_iter
            else:
                loader = test_iter
            a = torch.tensor([0, 1, 3, 4, 6, 7]).cuda()
            b = torch.tensor([2, 5, 8]).cuda()
            c = torch.tensor([0, 1, 3, 4, 6, 7,9,10,12,13,15,16]).cuda()
            losses_1 = []
            losses_2 = []
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y) in pbar:
                # place data on the correct device
                x = x.cuda()  #[b,49,9]
                y = y.cuda()  #[b,49,9]
                # forward the model
                with torch.set_grad_enabled(is_train):
                    temp_x = torch.cat((x,y),2)
                    tor, loss_1 = model_p2t(torch.index_select(temp_x, 2, c), torch.index_select(x, 2, b))

                    model_input = torch.cat((x[:,:,0:2],tor[:,:,0:1],x[:,:,3:5],tor[:,:,1:2],x[:,:,6:8],tor[:,:,2:3]),2)
                    logits,loss_2 = model(model_input,y)

                    loss = loss_2
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                    loss_1 = loss_1.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses_1.append(loss_1.item())
                    loss_2 = loss_2.mean() # collapse all losses if they are scattered on multiple gpus
                    losses_2.append(loss_2.item())

                if is_train:
                    # backprop and update the parameters
                    model_p2t.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_p2t.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f} model_p2t loss {loss_1.item():.5f} model loss {loss_2.item():.5f}")#. lr {lr:e}


        best_loss = float('inf')
        self.tokens = 0 # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            if self.config.train_type == 'DDi':
                run_epochDDi('train')
            elif self.config.train_type == 'DiD':
                run_epochDiD('train')
            if self.test_dataset is not None:
                if self.config.train_type == 'DDi':
                    test_loss_1,test_loss_2,loss = run_epochDDi('test')
                elif self.config.train_type == 'DiD':
                    test_loss_1,test_loss_2,loss = run_epochDiD('test')
                self.test_loss1_store[epoch] = test_loss_1
                self.test_loss2_store[epoch] = test_loss_2
                self.loss_store[epoch] = loss
                np.savetxt('test_loss1.txt',self.test_loss1_store)
                np.savetxt('test_loss2.txt',self.test_loss2_store)
                np.savetxt(self.config.test_loss_name,self.loss_store)

            if self.config.local_rank == 0:
                if loss < best_loss :
                    best_loss=loss
                    self.save_checkpoint()
