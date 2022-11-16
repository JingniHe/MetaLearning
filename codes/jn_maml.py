import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import sklearn.model_selection as sk
import backbone
##torch.cuda.set_device(1)

class MAML(nn.Module):
    def __init__(self, model_func, n_support, n_query, n_feature, approx=True):
        super(MAML,self).__init__()

        ##device = torch.device('cuda:1')
        #self.loss_fn = nn.CrossEntropyLoss().to(device)
        #self.loss_fn=nn.MSELoss().to(device)
        ##self.loss_fn=nn.BCEWithLogitsLoss().to(device)
        self.feature = model_func(n_feature) ## set the model, default as Conv4
        self.feat_dim = self.feature.final_feat_dim ## set the feature dimension for the last inear layer
        self.loss_fn=nn.BCEWithLogitsLoss()
        #self.sig=nn.Sigmoid()
        #self.loss_fn=nn.BCELoss().to(device)
        ##self.classifier = Linear_fw(n_feature, 1).to(device) #
        self.classifier = backbone.Linear_fw(self.feat_dim, 1)
        self.n_task = 4
        self.task_update_num = 5
        self.train_lr = 0.01  # this is the alpha learning_rate
        self.approx = approx  # first order approx.
        self.n_support=n_support
        self.n_query=n_query
        self.n_feature=n_feature
    def forward(self, x): ## x needs to be three dimensions for Conv1d [1,1,number_features]
        out = self.feature.forward(x) ## out is two dimension [1,self.feat_dim]
        #print(out.shape,out.dim())
        scores = self.classifier.forward(out)
        #print(scores.shape,scores.dim())
        return scores

    def set_forward(self, x_train, x_test, y_train):
        x_a_i = Variable(x_train)  # support data， x_a_i needs to be reshaped to three dimensions
        x_b_i = Variable(x_test)  # query data
        y_a_i = Variable(y_train)  # label for support data
        fast_parameters = list(self.parameters())  # the first gradient calcuated in line 45 is based on original weight
        for weight in self.parameters():
            weight.fast = None  ## what is weight.fast
        self.zero_grad()
        for task_step in range(self.task_update_num):
            scores = self.forward(x_a_i)  ## generate logits, which is after x_a_i go through the linear models
            #scores_sig=self.sig(scores)
            #print(scores,scores_sig)
            set_loss = self.loss_fn(scores,y_a_i)  ## get the loss between logits and y_hat within support(training) dataset, is a single value or vector
            # build full graph support gradient of gradient; gradient L/gradient parameters(weights)
            grad = torch.autograd.grad(set_loss, fast_parameters, create_graph=True)
            if self.approx:
                grad = [g.detach() for g in grad]  # do not calculate gradient of gradient if using first order approximation
            fast_parameters = []  ## empty parameter list
            for k, weight in enumerate(self.parameters()):
            # for usage of weight.fast, please see Linear_fw, Conv_fw in backbone.py
            ## weight.fast is the temporary weight for the k_th tweight
                if weight.fast is None:
                    weight.fast = weight - self.train_lr * grad[k]
                    # create weight.fast , update new weights
                else:
                    weight.fast = weight.fast - self.train_lr * grad[k]
                    # create an updated weight.fast, note the '-' is not merely minus value, but to create a new weight.fast
                fast_parameters.append(weight.fast)
                # gradients calculated in line 45 are based on newest fast weight, but the graph will retain the link to old weight.fasts
                ## the new fast_parameters contain new updated weight.fast
        scores = self.forward(x_b_i)  ## this will will reture logits for validated dataset x_b_i using the updated weight_fast
        return scores
		
    def set_forward_loss(self, x_train, x_test, y_train, y_test):
        scores = self.set_forward(x_train, x_test, y_train)  ##logits of validated dataset x_b_i using updated weight_fast
        #scores_sig=self.sig(scores)
        #print(scores,scores_sig)
        loss = self.loss_fn(scores,y_test)  ## get the loss between logits and y_hat within query(validation) dataset,is a single value or vector
        return loss

    def correct(self, x_train, x_test, y_train, y_test):
        scores = self.set_forward(x_train, x_test, y_train)
        self.sig=nn.Sigmoid()
        scores_sig=self.sig(scores).detach().cpu().numpy() #scores_sig return the probility of 1 for each samples and change to numpy
        y_pred=np.where(scores_sig > 0.5, 1, 0)
        y_query = y_test.cpu().numpy() ##y_query is a numpy
        num_correct = np.sum(y_pred == y_query)
        return num_correct, len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):  # overwrite parrent function
        print_freq = 10
        avg_loss=0
        task_count = 0
        loss_all = []
        optimizer.zero_grad()
        ##device = torch.device('cuda:1')

        # train
        for i, (x, y) in enumerate(train_loader):
            ##x_var = x.to(device)
            ##y_var = y.to(device)
            x_var = x
            y_var = y
            #x_var = Variable(x_t)
            #y_var=Variable(y_t)
            x_train, x_test, y_train, y_test = sk.train_test_split(x_var, y_var, test_size=0.5, random_state=40)
            #print(x.shape, y.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape)
            loss = self.set_forward_loss(x_train, x_test, y_train, y_test)
            avg_loss = avg_loss + loss.item() ## avg_loss it's the sum of all the loss that we get from validation set.
            #3print(loss.item())
            ## For new version of pytorch, should use avg_loss = avg_loss+loss.item()
            loss_all.append(loss)  ## loss_all is a list contain all the loss that we get from query(validation) dataset
            #print(loss_all)
            task_count += 1
            if task_count == self.n_task:
                loss_q = torch.stack(loss_all).sum(0)  ##  gradient the sum of loss within the validation set
                #print(loss_q)
                # this step will generate gradient L
                loss_q.backward()
                # this step will autimatically generate new theia
                optimizer.step()  ## the learning_rate inside the optimizer is the meta-learning_rate(beta)
                task_count = 0
                loss_all = []
            optimizer.zero_grad()
            if i % print_freq == 0:
                print('Epoch {} | Loss {:f}'.format(epoch, avg_loss / float(i + 1)))
    def test_loop(self,test_loader,return_std=False):  # overwrite parrent function
        correct = 0
        count = 0
        acc_all = []
       ##device = torch.device('cuda:1')

        iter_num=len(test_loader)
        for i, (x, y) in enumerate(test_loader):
            ##x_var = x.to(device)
            ##y_var = y.to(device)
            x_var=x
            y_var=y
            x_train, x_test, y_train, y_test = sk.train_test_split(x_var, y_var, test_size=0.5, random_state=42)
            #print(x.shape, y.shape, x_train.shape, x_test.shape, y_train.shape, y_test.shape)
            correct_this,count_this = self.correct(x_train, x_test, y_train, y_test)
            acc_all.append(correct_this/count_this*100)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))
        if return_std:
            return acc_mean, acc_std
        else:
            return acc_mean
