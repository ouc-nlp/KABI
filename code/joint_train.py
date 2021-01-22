import random
import argparse
import sys, os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.stream_metrics import StreamClsMetrics, AverageMeter
from utils.logger import mkdir_if_missing,Logger
from networks.rps_net_mlp import RPS_net_mlp
from utils.Utils import get_concat_dataloader

_model_dict = {
    'rps_net_mlp':RPS_net_mlp,
}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='../datasets_folder')
    parser.add_argument("--model_root", type=str, default='../model')
    parser.add_argument("--dataset_name", type=str, default='mnist')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default='rps_net_mlp')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--random_seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start", type=int, default=5)
    parser.add_argument("--end", type=int, default=5)
    parser.add_argument("--phases", type=int, default=5)
    return parser


def train_one_epoch(cur_epoch, criterion, model, optim, train_loader, device, scheduler=None, print_interval=50):
    """Train and return epoch loss"""

    if scheduler is not None:
        scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

    avgmeter = AverageMeter()
    for cur_step, (images, labels) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

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

def validate(model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            preds = outputs.detach()
            targets = labels
            metrics.update(preds, targets)
        score = metrics.get_results()
    return score


def main():
    opts = get_parser().parse_args()

    mkdir_if_missing('../logs')
    sys.stdout = Logger(os.path.join('../logs', opts.dataset_name, str(opts.phases)+'phases','joint_train_result.txt'))
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up random seed to ensure the results remain the same at any running time
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    train_dataset_folder = os.path.join(opts.data_root, opts.dataset_name,str(opts.phases)+'phases',
                                        'separated/train/batch_train')
    train_dataset_filename_list = os.listdir(train_dataset_folder)
    for index, train_dataset_filename in enumerate(train_dataset_filename_list[opts.start - 1:opts.end]):
        save_path = os.path.join(opts.model_root, opts.dataset_name,str(opts.phases)+'phases', 'joint_model',
                                 'joint_model_for_batch' + train_dataset_filename[-2:])  # 在参数中指定模型存放文件夹路径
        print('save_path:', save_path)
        train_dataset_file_path_list = []
        for i in range(index+opts.start):
            train_dataset_file_path = os.path.join(train_dataset_folder, train_dataset_filename_list[i])
            train_dataset_file_path_list.append(train_dataset_file_path)
        print('train_dataset_file_path_list:', train_dataset_file_path_list)
        val_dataset_file_path_list = []
        val_dataset_folder = os.path.join(opts.data_root, opts.dataset_name,str(opts.phases)+'phases', 'separated/test/batch_test')
        val_dataset_filename_list = os.listdir(val_dataset_folder)
        for i in range(index+opts.start):
            val_dataset_file_path = os.path.join(val_dataset_folder, val_dataset_filename_list[i])
            val_dataset_file_path_list.append(val_dataset_file_path)
        print('val_dataset_file_path_list:', val_dataset_file_path_list)
        train_loader, val_loader, num_classes,_ = get_concat_dataloader(opts.dataset_name, train_dataset_file_path_list, val_dataset_file_path_list,
                                                                            opts.batch_size)
        model = _model_dict[opts.model]()
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # change the number of output logits
        model.to(device)
        metrics = StreamClsMetrics(num_classes)

        params_10x = []
        for name, param in model.named_parameters():
            params_10x.append(param)

        optimizer = torch.optim.Adam(params=[{'params': params_10x, 'lr': opts.lr*10}],
                                    lr=opts.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=50, gamma=0.1)
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # Restore
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
                                            scheduler=scheduler)
            print("End of Epoch %d/%d, Average Loss=%f" % (cur_epoch, opts.epochs, epoch_loss))

            # =====  Validation  =====
            print("validate on val set...")
            model.eval()
            val_score = validate(model=model,
                                 loader=val_loader,
                                 device=device,
                                 metrics=metrics)
            print(metrics.to_str(val_score))
            cur_epoch += 1
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'num_classes': num_classes,
        }
        # store the model
        torch.save(state, save_path)

if __name__ == '__main__':
    main()
