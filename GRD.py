"""
Robot GPT config
Author @ YYM CASIA
"""
import argparse
import logging

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from core.model import GPT, GPT_p2t, GPTConfig
from core.trainer import Trainer,TrainerDDi_DiD, TrainerConfig  # train
from core.utils import set_seed

def get_parser():
    parser = argparse.ArgumentParser(description="Robot GPT Network")
    parser.add_argument('--local_rank', type=int, help='node rank for distributed training')
    parser.add_argument('--mode_select', choices=['train','test'],default='test')
    parser.add_argument('--model_select', choices=['Dynamic','Reverse Dynamic','DiD','DDi'],default='Reverse Dynamic')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--learning_decay', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=42)    
    parser.add_argument('--train_epochs', type=int, default=1000)
    return parser

class MyDataset_D(Dataset):
    def __init__(self,data):
        self.data = data

    def __getitem__(self, index):
        data_step = self.data[index,:]
        x = torch.tensor(data_step[:-1,:], dtype=torch.float)  
        y = torch.tensor(data_step[1:,:], dtype=torch.float)    #49,9
        return x,y

    def __len__(self):
        return self.data.shape[0]

class MyDataset_Di(Dataset):
    def __init__(self,data):
        self.data = data
        self.a = torch.tensor([0,1,3,4,6,7])
        self.b = torch.tensor([2,5,8])

    def __getitem__(self, index):
        data_step = self.data[index]
        temp_x = torch.tensor(data_step[:,:], dtype=torch.float)  
        temp_x = torch.index_select(temp_x, 1, self.a)  #torch.Size([50, 6])
        x = torch.cat((temp_x[:-1],temp_x[1:]),1)    #torch.Size([49, 12])
        y = torch.tensor(data_step[:-1,:], dtype=torch.float)    #49,9
        #y = torch.cat((torch.zeros((1,9)),y),0)
        y = torch.index_select(y, 1, self.b)  #torch.Size([49,3])
        #print(x,y)
        return x,y

    def __len__(self):
        return self.data.shape[0]

if __name__ == '__main__':
    np.set_printoptions(suppress=False,threshold=np.inf)
    torch.set_printoptions(threshold=np.inf)
    parser = get_parser()
    args = parser.parse_args()
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    global_rank = dist.get_rank()

    set_seed(args.seed)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.mode_select == 'train':
        only_last_layer = False
        data = np.loadtxt("data/data.txt").reshape(-1,50,9)
        divide = int(data.shape[0] * 0.2)
        train = data[divide:]
        test = data[0:divide]
    elif args.mode_select == 'test':
        only_last_layer = True
        data = np.load("data/real_data_ur.npy")
        data = data[0:4050].reshape(-1,50,9)
        train = data[0:70]
        test = data[70:]

    if args.model_select == 'Reverse Dynamic':
        train_dataset = MyDataset_Di(train)
        test_dataset = MyDataset_Di(test)
    else:
        train_dataset = MyDataset_D(train)
        test_dataset = MyDataset_D(test)
    
    tokens_per_epoch = len(train_dataset) * 49
    train_epochs = args.train_epochs

    if args.model_select == 'DDi' or args.model_select == 'DiD':
        mconf = GPTConfig(49,n_layer=64, n_head=8, n_embd=512,only_last_layer=only_last_layer,is_DDi_DiD=False)   # model configuration
        model = GPT(mconf)
        mconf_p2t = GPTConfig(49,n_layer=64, n_head=8, n_embd=512,only_last_layer=only_last_layer,is_DDi_DiD=True)  # model configuration
        model_p2t = GPT_p2t(mconf_p2t)
        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(
            max_epochs=train_epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate,
            lr_decay=args.learning_decay, 
            warmup_tokens=tokens_per_epoch, 
            final_tokens=train_epochs*tokens_per_epoch,
            local_rank=args.local_rank,
            train_type = args.model_select,
            D_model_path = 'Dynamic_' + args.mode_select + '.pkl',
            Di_model_path = 'Reverse_Dynamic_' + args.mode_select + '.pkl',
            ckpt_path = 'GPT_' + args.mode_select + '.pkl',
            test_loss_name = 'test_loss' + '_' + args.model_select + '_' + args.mode_select + '.txt'
        )
        trainer = TrainerDDi_DiD(model,model_p2t, train_dataset, test_dataset, tconf)
    elif args.model_select == 'Dynamic':
        mconf = GPTConfig(49,n_layer=64, n_head=8, n_embd=512,only_last_layer=only_last_layer,is_DDi_DiD=False)   # model configuration
        model = GPT(mconf)
        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(
            max_epochs=train_epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate,
            lr_decay=args.learning_decay, 
            warmup_tokens=tokens_per_epoch, 
            final_tokens=train_epochs*tokens_per_epoch,
            local_rank=args.local_rank,
            train_type = args.model_select,
            D_model_path = args.model_select + '_' + args.mode_select + '.pkl',
            Di_model_path = args.model_select + '_' + args.mode_select + '.pkl',
            ckpt_path = 'GPT_' + args.mode_select + '.pkl',
            test_loss_name = 'test_loss' + '_' + args.model_select + '_' + args.mode_select + '.txt'
        )
        trainer = Trainer(model,train_dataset, test_dataset, tconf)
    else:
        mconf_p2t = GPTConfig(49,n_layer=64, n_head=8, n_embd=512,only_last_layer=only_last_layer,is_DDi_DiD=True)  # model configuration
        model_p2t = GPT_p2t(mconf_p2t)
        # initialize a trainer instance and kick off training
        tconf = TrainerConfig(
            max_epochs=train_epochs, 
            batch_size=args.batch_size, 
            learning_rate=args.learning_rate,
            lr_decay=args.learning_decay, 
            warmup_tokens=tokens_per_epoch, 
            final_tokens=train_epochs*tokens_per_epoch,
            local_rank=args.local_rank,
            train_type = args.model_select,
            D_model_path = "model_backup/p2t_model_12.18.pkl",
            Di_model_path = args.mode_select + '_' + args.mode_select + '.pkl',
            ckpt_path = 'GRD_' + args.mode_select + '.pkl',
            test_loss_name = 'loss_store/test_loss' + '_' + args.mode_select + '_' + args.mode_select + '.txt'
        )
        trainer = Trainer(model_p2t,train_dataset, test_dataset, tconf)


    trainer.train()
