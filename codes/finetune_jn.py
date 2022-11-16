import numpy as np
import torch
import torch.nn as nn
import os
import random
from torch.utils.data import Dataset
import sklearn.model_selection as sk
from io_utils import parse_args, get_resume_file, get_best_file, get_assigned_file
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

def finetune(model, optimization, start_epoch, stop_epoch, params):
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    else:
        raise ValueError('Unknown optimization, please define by yourself')

    max_acc=params.best_acc

    ## Split the training file into training and testing sets
    trainfile = params.data_dir+"c1_trial2_input.csv"
    batch_size =  params.train_n_support + params.train_n_query  # n_support=n_query
    dataset_train = InputDataset(trainfile)
    train_len = int(0.8*len(dataset_train))
    valid_len = len(dataset_train) - train_len
    train_db,val_db = torch.utils.data.random_split(dataset_train,[train_len,valid_len]);
    train_loader = torch.utils.data.DataLoader(dataset=train_db, batch_size=batch_size, drop_last=True, shuffle=True,num_workers=5)
    val_loader = torch.utils.data.DataLoader(dataset=val_db, batch_size=batch_size, drop_last=True, shuffle=True,num_workers=5)

    for epoch in range(start_epoch, stop_epoch):

        model.train()
        model.train_loop(epoch, train_loader, optimizer)
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

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
    ##device=torch.device('cuda:1')
    params = parse_args('finetune')
    optimization = 'Adam'
    model = MAML(model_func=model_dict[params.model], n_support=params.test_n_support, n_query=params.test_n_query, n_feature=params.n_feature, approx=(params.method=='maml_approx'))
    ##model=model.to(device)
    params.checkpoint_dir='%s/checkpoints/%s_%s' %(params.save_dir,params.train_n_support,params.train_n_query)
    if params.save_iter !=-1:
        modelfile=get_assigned_file(params.checkpoint_dir,params.save_iter) ## use the model from the save_iter epoch
    else:
        modelfile=get_best_file(params.checkpoint_dir) ## use the best model
    if modelfile is not None:
        tmp=torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    for weight in enumerate(model.parameters()):
        print(weight)

    start_epoch=params.start_ft_epoch
    stop_epoch=params.stop_ft_epoch
    model=finetune(model, optimization, start_epoch, stop_epoch, params)
