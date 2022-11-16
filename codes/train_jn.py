import numpy as np
import torch
import torch.nn as nn
import os
import glob
import argparse
import random
from torch.utils.data import Dataset
import sklearn.model_selection as sk
from io_utils import model_dict, parse_args, get_resume_file
from jn_maml import MAML

class InputDataset(Dataset):
    def __init__(self, filename):
        xy = np.loadtxt(filename, delimiter=",", dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.x= self.x.reshape(self.x.shape[0], 1, self.x.shape[1]) #x needs to be reshaped to three dimensions
        self.y = torch.from_numpy(xy[:, [0]])
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

def train(model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc=0
    filelist=["All_Sklar_input.csv","c1_trial2_input.csv","phg000013.filtered.GRU.plink.update1_input.csv","BDO_BARD_GRU_merge_input.csv","c2_asc_phs000298_exomes_input.csv","phg000014.GRU.geno.filtered.plink_input.csv","New_CMB-trios_input.csv","nonGAIN_Schizophrenia_consent_GRU_input.csv","puwma_dbgap_16feb11_curated_input.csv","swe_input.csv"]
    for epoch in range(start_epoch, stop_epoch):
        batch_size = params.train_n_support + params.train_n_query  # n_support=n_query
        if (epoch % 2) ==0:
            trainfile = params.data_dir+"c1_trial2_input.csv"
        else:
            trainfile = params.data_dir+random.choice(filelist)
        dataset_train = InputDataset(trainfile)
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, shuffle=True,num_workers=5)

        model.train()
        model.train_loop(epoch, train_loader, optimizer)
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        valfile = params.data_dir + "c2_asc_phs000298_exomes_input.csv"
        dataset_val = InputDataset(valfile)
        val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, drop_last=True, shuffle=True,num_workers=4)
        acc=model.test_loop(val_loader)
        if acc>max_acc:
            print("best model! save...")
            max_acc=acc
            outfile=os.path.join(params.checkpoint_dir,'best_model.tar')
            torch.save({'epoch':epoch,'state':model.state_dict()},outfile)
            for weight in enumerate(model.parameters()):
                print(weight)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile=os.path.join(params.checkpoint_dir,'{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch,'state':model.state_dict()},outfile)

    return model

if __name__ == '__main__':
    np.random.seed(10)
    params = parse_args('train')

    ##device=torch.device('cuda:1')
    optimization = 'Adam'

    model = MAML(model_func=model_dict[params.model], n_support=params.train_n_support, n_query=params.train_n_query, n_feature=params.n_feature, approx=(params.method=='maml_approx'))
    ##model = model.to(device)
    params.checkpoint_dir='%s/checkpoints/%s_%s' %(params.save_dir,params.train_n_support,params.train_n_query)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch=params.start_epoch
    stop_epoch=params.stop_epoch
    #print(start_epoch,stop_epoch,params.data_dir,params.save_dir)

    if params.resume:
        resume_file=get_resume_file(params.checkpoint_dir)
        if resume_file is not None:
            tmp=torch.load(resume_file)
            start_epoch=tmp['epoch']+1
            model.load_state_dict(tmp['state'])
    #print(start_epoch,stop_epoch,params.data_dir,params.save_dir)
    model=train(model, optimization, start_epoch, stop_epoch, params)
