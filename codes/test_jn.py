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

if __name__ == '__main__':
    ##device=torch.device('cuda:1')
    params = parse_args('test')

    model = MAML(model_func=model_dict[params.model], n_support=params.test_n_support, n_query=params.test_n_query, n_feature=params.n_feature, approx=(params.method=='maml_approx'))
    ##model=model.to(device)
    checkpoint_dir='%s/checkpoints/%s_%s' %(params.save_dir,params.train_n_support,params.train_n_query)
    if params.save_iter !=-1:
        modelfile=get_assigned_file(checkpoint_dir,params.save_iter) ## use the model from the save_iter epoch
    else:
        modelfile=get_best_file(checkpoint_dir) ## use the best model
    if modelfile is not None:
        tmp=torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    batch_size = params.test_n_support + params.test_n_query  # n_support=n_query
    filename = params.data_dir + "c2_asc_phs000298_exomes_input.csv"
    dataset = InputDataset(filename)
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=True, shuffle=True,num_workers=4)

    if params.adaptation:
        model.task_update_num=100
    model.eval()
    acc_mean, acc_std = model.test_loop(test_loader,return_std=True)
    for weight in enumerate(model.parameters()):
        print(weight)
