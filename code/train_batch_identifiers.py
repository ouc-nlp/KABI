import random
import argparse
import sys, os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.stream_metrics import StreamClsMetrics, AverageMeter
from utils.logger import mkdir_if_missing, Logger
from utils.Utils import get_concat_dataloader
from networks.rps_net_mlp import RPS_net_mlp
from networks.resnet18_for_cifer100 import resnet18
from torchvision import models

_model_dict = {
    'rps_net_mlp':RPS_net_mlp,
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--num_new_classes_single_status", type=int, default=10)
    parser.add_argument("--phases", type=int, default=10)
    return parser


def train_one_epoch(cur_epoch, criterion, model, optim, train_loader, device, num_batches, num_new_classes_single_status,scheduler=None, print_interval=30):
    """Train and return epoch loss"""

    if scheduler is not None:
        scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

    avgmeter = AverageMeter()
    for cur_step, (images, labels) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        for label_index in range(len(labels)):
            for index in range(num_batches):
                if num_new_classes_single_status*index<=labels[label_index]<num_new_classes_single_status*(index+1):
                    labels[label_index] = labels[label_index] - labels[label_index] + index
        optim.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optim.step()
        avgmeter.update('loss', loss.item())
        avgmeter.update('interval loss', loss.item())

        if (cur_step+1) % print_interval == 0:
            interval_loss = avgmeter.get_results('interval loss')
            print("Epoch %d, Batch %d/%d, Loss=%f" %
                  (cur_epoch, cur_step+1, len(train_loader), interval_loss))
            avgmeter.reset('interval loss')
    return avgmeter.get_results('loss') / len(train_loader) # epoch loss


def validate(model, loader, device, num_batches, num_new_classes_single_status, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            for label_index in range(len(labels)):
                for index in range(num_batches):
                    if num_new_classes_single_status * index <= labels[label_index] < num_new_classes_single_status * (index + 1):
                        labels[label_index] = labels[label_index] - labels[label_index] + index
            outputs = model(images)
            preds = outputs.detach()
            targets = labels

            metrics.update(preds, targets)
        score = metrics.get_results()
    return score


def main():
    opts = get_parser().parse_args()
    print(opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mkdir_if_missing('../logs')
    # Set up random seed to ensure the results remain the same at any running time
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    sys.stdout = Logger(os.path.join('../logs', opts.dataset_name,str(opts.phases)+'phases', 'train_batch_identifiers_result.txt'))
    train_dataset_folder = os.path.join(opts.data_root, opts.dataset_name, str(opts.phases)+'phases','accumulated/train_for_batch/mini_batch_train')
    train_dataset_filename_list = os.listdir(train_dataset_folder)
    for index, dataset_filename in enumerate(train_dataset_filename_list[opts.start - 2:opts.end - 1]):
        save_path = os.path.join(opts.model_root, opts.dataset_name,str(opts.phases)+'phases', 'batch_identifiers','M_status' + dataset_filename[-2:])  # 在参数中指定模型存放文件夹路径
        print('save_path:', save_path)
        #  Set up dataloader
        train_file_path = os.path.join(opts.data_root,opts.dataset_name,str(opts.phases)+'phases','accumulated/train_for_batch/mini_batch_train/mini_batch'+dataset_filename[-2:])
        train_file_path_list = [train_file_path]
        print('train_file_path_list:', train_file_path_list)
        val_file_path_list = []
        val_dataset_folder = os.path.join(opts.data_root, opts.dataset_name, str(opts.phases)+'phases','separated/test/batch_test')
        val_dataset_filename_list = os.listdir(val_dataset_folder)
        for val_dataset_filename in val_dataset_filename_list[:opts.start + index]:
            val_file_path = os.path.join(val_dataset_folder, val_dataset_filename)
            val_file_path_list.append(val_file_path)
        print('val_file_path_list:', val_file_path_list)
        train_loader, val_loader, num_classes, start_label = get_concat_dataloader(opts.dataset_name,
                train_file_path_list, val_file_path_list,
                opts.batch_size)

        if int(dataset_filename[-2:]) == 10:
            model_load_path = os.path.join(opts.model_root, opts.dataset_name, str(opts.phases) + 'phases',
                                           'amalgamation_models/M_status10')
        else:
            model_load_path = os.path.join(opts.model_root, opts.dataset_name, str(opts.phases) + 'phases',
                                           'amalgamation_models/M_status' + dataset_filename[-2:])

        print('model_load_path:', model_load_path)
        # map_location用于在cpu上加载预先训练的GPU模型
        state = torch.load(model_load_path, map_location=lambda storage, loc: storage)
        model = _model_dict[opts.model]()
        model.fc = nn.Linear(model.fc.in_features, state['num_classes'])
        model.load_state_dict(state['state_dict'])
        model.fc = nn.Linear(model.fc.in_features, len(val_file_path_list))
        model.to(device)
        metrics = StreamClsMetrics(len(val_file_path_list))

        params_1x = []
        params_10x = []
        for name, param in model.named_parameters():
            if 'fc' in name:
                params_10x.append(param)
            else:
                params_1x.append(param)
        optimizer = torch.optim.Adam(params=[{'params': params_1x,  'lr': opts.lr},
                                            {'params': params_10x, 'lr': opts.lr*10}],
                                    lr=opts.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=25, gamma=0.1)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        cur_epoch = 0
        # Train Loop
        while cur_epoch < opts.epochs:
            model.train()
            epoch_loss = train_one_epoch(model=model,
                                            criterion=criterion,
                                            cur_epoch=cur_epoch,
                                            optim=optimizer,
                                            train_loader=train_loader,
                                            device=device,
                                            num_batches=len(val_file_path_list),
                                            num_new_classes_single_status=opts.num_new_classes_single_status,
                                            scheduler=scheduler,
                                            )
            print("End of Epoch %d/%d, Average Loss=%f" % (cur_epoch, opts.epochs, epoch_loss))


            # =====  Validation  =====
            print("validate on val set...")
            model.eval()
            val_score = validate(model=model,
                                 loader=val_loader,
                                 device=device,
                                 num_batches=len(val_file_path_list),
                                 num_new_classes_single_status=opts.num_new_classes_single_status,
                                 metrics=metrics,
                                 )
            print(metrics.to_str(val_score))
            cur_epoch += 1
            # =====  Save Best Model  =====
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'num_batches': len(val_file_path_list),
        }

        torch.save(state, save_path)

if __name__ == '__main__':
    main()
