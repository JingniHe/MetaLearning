import numpy as np
import os
import glob
import argparse
import backbone

model_dict = dict(
            Conv4 = backbone.Conv4,
            Conv4NN = backbone.Conv4NN,
            Conv6 = backbone.Conv6,
            Conv6NN = backbone.Conv6NN,
            ResNet10 = backbone.ResNet10,
            ResNet18 = backbone.ResNet18,
            ResNet34 = backbone.ResNet34,
            ResNet50 = backbone.ResNet50,
            ResNet101 = backbone.ResNet101) 

def parse_args(script):
    parser = argparse.ArgumentParser(description='few-shot script %s' % (script))
    parser.add_argument('--model', default='Conv4', help='model: Conv{4|6} / Conv{4|6}NN / ResNet{10|18|34|50|101}')
    parser.add_argument('--method', default='maml_approx',
                        help='maml/maml_approx')
    parser.add_argument('--n_feature', default=100, type=int,
                        help='number of features')
    parser.add_argument('--train_n_query', default=30, type=int,
                        help='number of samples for validation')
    parser.add_argument('--train_n_support', default=30, type=int,
                        help='number of samples for training')
    parser.add_argument('--test_n_query', default=20, type=int,
                        help='number of samples for validation')
    parser.add_argument('--test_n_support', default=20, type=int,
                        help='number of samples for training')
    parser.add_argument('--train_aug', action='store_true',
                        help='perform data augmentation or not during training ') 
    parser.add_argument('--save_dir', default='/export/qlong/jhe/metalearning/simulations/metalearn/working_dir/')
    parser.add_argument('--data_dir', default='/export/qlong/jhe/metalearning/simulations/datasets/controls/SimInput/')
    if script == 'train':
        parser.add_argument('--save_freq', default=50, type=int, help='Save frequency')
        parser.add_argument('--start_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_epoch', default=2500, type=int,
                            help='Stopping epoch')
        parser.add_argument('--resume', action='store_true',
                            help='continue from previous trained model with largest epoch')
    elif script == 'finetune':
        parser.add_argument('--save_iter', default=-1, type=int,
                            help='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--save_freq', default=50, type=int, help='Save frequency')
        parser.add_argument('--best_acc', default=0.0, type=int, help='Best test accuracy from train.txt')
        parser.add_argument('--start_ft_epoch', default=0, type=int, help='Starting epoch')
        parser.add_argument('--stop_ft_epoch', default=600, type=int,
                            help='Stopping epoch')
        parser.add_argument('--resume', action='store_true',
                            help='continue from previous trained model with largest epoch')

    elif script == 'test':
        parser.add_argument('--save_iter', default=-1, type=int,
                            help='saved feature from the model trained in x epoch, use the best model if x is -1')
        parser.add_argument('--adaptation', action='store_true', help='further adaptation in test time or not')
    else:
        raise ValueError('Unknown script')

    return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
    assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
    return assign_file

def get_resume_file(checkpoint_dir):
    filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar')) ## Reture a list contain all the "tar" files under checkpoint_dir
    if len(filelist) == 0:
        return None

    filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ] ## filelist contains all the "tar" files under checkpoint_dir except "best_model.tar"
    epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
    max_epoch = np.max(epochs)
    resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch)) ## checkpoint_dir/max_epoch.tar
    return resume_file

def get_best_file(checkpoint_dir):
    best_file = os.path.join(checkpoint_dir, 'best_model.tar')
    if os.path.isfile(best_file):
        return best_file
    else:
        return get_resume_file(checkpoint_dir)
