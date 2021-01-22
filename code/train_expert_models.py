import random
import argparse
import sys, os
import numpy as np
from tqdm import tqdm
from torchvision import models
import torch
import torch.nn as nn
import time
from datetime import timedelta
from utils.logger import mkdir_if_missing,Logger
from networks.rps_net_mlp import RPS_net_mlp
from networks.resnet18_for_cifer100 import resnet18
from utils.Utils import get_concat_dataloader,accuracy
from utils.AverageMeter import AverageMeter


_model_dict = {
    'rps_net_mlp': RPS_net_mlp,
    'resnet18': resnet18,
    'standard_resnet18': models.resnet18
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='../datasets_folder')
    parser.add_argument("--model_root", type=str, default='../model')
    parser.add_argument("--dataset_name", type=str, default='cifar100')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument('--phases',type=int, default=10)
    return parser


def train_one_epoch(cur_epoch, criterion, model, optim, train_loader, device, start_label, scheduler=None, print_interval=50):
    """Train and return epoch loss"""

    if scheduler is not None:
        scheduler.step()
    loss = None
    for cur_step, (images, labels) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        labels -= start_label
        optim.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()
    return loss


def validate(model, loader, device, top, start_label):
    """Do validation and return specified samples"""
    opts = get_parser().parse_args()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            labels -= start_label
            outputs = model(images)
            preds = outputs.detach()
            targets = labels

            if opts.dataset_name == 'ilsvrc2012':
                # compute top-5 classification accuracy for ilsvrc 2012
                _, prec = accuracy(preds.data, targets, topk=(1, 5))
            else:
                # compute top-1 classification accuracy for cifar100 and mnist
                prec, _ = accuracy(preds.data, targets, topk=(1,2))
            top.update(prec.item(), images.size(0))  # inputs.size(0) = batch_size
    return top


def main():
    opts = get_parser().parse_args()

    # dir and log
    mkdir_if_missing('../logs')
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up random seed to ensure the results remain the same at any running time
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    sys.stdout = Logger(os.path.join('../logs', opts.dataset_name, str(opts.phases)+'phases','train_expert_models_result.txt'))

    train_dataset_folder = os.path.join(opts.data_root, opts.dataset_name,str(opts.phases)+'phases',
                                        'separated/train/batch_train')
    train_dataset_filename_list = os.listdir(train_dataset_folder)
    for index, train_dataset_filename in enumerate(train_dataset_filename_list[opts.start - 1:opts.end]):
        save_path = os.path.join(opts.model_root, opts.dataset_name, str(opts.phases)+'phases','expert_models',
                                 'expert_model_for_batch' + train_dataset_filename[-2:])  # 在参数中指定模型存放文件夹路径
        print('save_path:', save_path)
        #  Set up dataloader
        train_dataset_file_path = os.path.join(train_dataset_folder, train_dataset_filename)
        train_dataset_file_path_list = [train_dataset_file_path]
        print('train_dataset_file_path_list:', train_dataset_file_path_list)
        val_dataset_folder = os.path.join(opts.data_root, opts.dataset_name,str(opts.phases)+'phases', 'separated/test/batch_test')
        val_dataset_file_path = os.path.join(val_dataset_folder, train_dataset_filename)
        val_dataset_file_path_list = [val_dataset_file_path]
        print('val_dataset_file_path_list:', val_dataset_file_path_list)

        train_loader, val_loader, num_classes, start_label = get_concat_dataloader(opts.dataset_name,train_dataset_file_path_list, val_dataset_file_path_list,
                                                                            opts.batch_size)
        model = _model_dict[opts.model]()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.to(device)

        params_10x = []
        for name, param in model.named_parameters():
                params_10x.append(param)
        optimizer = torch.optim.Adam(params=[{'params': params_10x, 'lr': opts.lr*10}],
                                    lr=opts.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.epochs/2, gamma=0.1)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        cur_epoch = 0
        # Train Loop
        start_time = time.time()
        while cur_epoch < opts.epochs:
            top = AverageMeter()
            model.train()
            epoch_loss = train_one_epoch(model=model,
                                            criterion=criterion,
                                            cur_epoch=cur_epoch,
                                            optim=optimizer,
                                            train_loader=train_loader,
                                            device=device,
                                            scheduler=scheduler,
                                            start_label = start_label)

            # =====  Validation  =====
            print("validate on val set...")
            model.eval()
            top= validate(model=model,
                                 loader=val_loader,
                                 device=device,
                                 top=top,
                                 start_label = start_label
                                  )
            end_time = time.time()
            current_elapsed_time = end_time - start_time
            print('{:03}/{:03} | {} | Train : loss = {:.4f} | Val : accuracy = {}%'.
                  format(cur_epoch + 1, opts.epochs, timedelta(seconds=round(current_elapsed_time)),
                         epoch_loss, top.avg))

            cur_epoch += 1

        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'num_classes': num_classes,
        }
        # store model
        torch.save(state, save_path)

if __name__ == '__main__':
    main()
