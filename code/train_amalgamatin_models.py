import random
import os, sys
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
from loss import SoftCELoss
from utils.logger import mkdir_if_missing, Logger
from utils.Utils import get_concat_dataloader,accuracy
from networks.rps_net_mlp import RPS_net_mlp
from networks.resnet18_for_cifer100 import resnet18
from utils.AverageMeter import AverageMeter
from torchvision import models
import time
from datetime import timedelta
_model_dict = {
    'rps_net_mlp':RPS_net_mlp,
    'resnet18': resnet18,
    'standard_resnet18':models.resnet18

}

_dataset_class_number = {
    'mnist':10,
    'cifar100':100,
    'ilsvrc2012':1000
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--phases", type=int, default=10)
    return parser


def kd(cur_epoch, criterion_ce, model, teachers, optim, train_loader, device, scheduler=None, print_interval=50):
    """Train and return epoch loss"""
    t1, t2 = teachers

    if scheduler is not None:
        scheduler.step()

    print("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))
    loss = None
    for cur_step, (images, labels) in enumerate(train_loader):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # get soft-target
        optim.zero_grad()
        with torch.no_grad():
            t1_out = t1(images)
            t2_out = t2(images)

            t_outs = torch.cat((t1_out, t2_out), dim=1)
        # get student output
        s_outs = model(images)

        loss = criterion_ce(s_outs, t_outs, labels)

        loss.backward()
        optim.step()

    return loss


def validate(model, loader, device, top):
    """Do validation and return specified samples"""
    opts = get_parser().parse_args()
    class_num_each_phase = _dataset_class_number[opts.dataset_name] // opts.phases
    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(images)
            preds = outputs.detach()  #
            targets = labels  #
            if opts.dataset_name == 'ilsvrc2012':
                _, prec = accuracy(preds.data, targets, topk=(1, 5))  # 每组batch_size样本对应着一个prec1,一个prec5
            else:
                prec, _ = accuracy(preds.data, targets, topk=(1, 2))  # 每组batch_size样本对应着一个prec1,一个prec5
            top.update(prec.item(), images.size(0))  # inputs.size(0) = batch_size
        return top


def main():

    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mkdir_if_missing('../logs')
    print(opts)
    # Set up random seed to ensure the results remain the same at any running time
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)
    sys.stdout = Logger(os.path.join('../logs', opts.dataset_name, str(opts.phases)+'phases', 'train_amalgamation_models_result.txt'))

    first_teacher_path = os.path.join(opts.model_root, opts.dataset_name, str(opts.phases)+'phases','expert_models/expert_model_for_batch01')

    train_dataset_folder = os.path.join(opts.data_root, opts.dataset_name,str(opts.phases)+'phases','accumulated/train_for_cls/mini_batch_train')
    train_dataset_filename_list = os.listdir(train_dataset_folder)
    for index, dataset_filename in enumerate(train_dataset_filename_list[opts.start-2:opts.end-1]):
        save_path = os.path.join(opts.model_root,opts.dataset_name,str(opts.phases)+'phases','amalgamation_models', 'M_status'+dataset_filename[-2:])  # 在参数中指定模型存放文件夹路径
        print('save_path:', save_path)
        #  Set up dataloader
        train_file_path_of_store_exemplars = os.path.join(train_dataset_folder, dataset_filename)
        train_file_path_of_new_samples = os.path.join(opts.data_root,opts.dataset_name,str(opts.phases)+'phases','separated/train/batch_train', 'batch'+dataset_filename[-2:])
        train_file_path_list = [train_file_path_of_store_exemplars, train_file_path_of_new_samples]
        print('train_file_path_list:',train_file_path_list)

        val_file_path_list = []
        val_dataset_folder = os.path.join(opts.data_root,opts.dataset_name,str(opts.phases)+'phases','separated/test/batch_test')
        val_dataset_filename_list = os.listdir(val_dataset_folder)
        for val_dataset_filename in val_dataset_filename_list[:opts.start+index]:
            val_file_path = os.path.join(val_dataset_folder,val_dataset_filename)
            val_file_path_list.append(val_file_path)
        print('val_file_path_list:', val_file_path_list)
        train_loader, val_loader, num_classes, start_label = get_concat_dataloader(opts.dataset_name,
                train_file_path_list, val_file_path_list,
                opts.batch_size)
        # load teacher model from path on hard disk
        teacher_model1 = _model_dict[opts.model]()
        if opts.start + index == 2:
            model_load_path1 = first_teacher_path
        elif int(dataset_filename[-2:])-1<10:
            model_load_path1 = os.path.join(opts.model_root,opts.dataset_name,str(opts.phases)+'phases','amalgamation_models/M_status0'+str(int(dataset_filename[-2:])-1))
        elif int(dataset_filename[-2:]) - 1 >= 10:
            model_load_path1 = os.path.join(opts.model_root, opts.dataset_name, str(opts.phases) + 'phases',
                                            'amalgamation_models/M_status' + str(int(dataset_filename[-2:]) - 1))

        state1 = torch.load(model_load_path1, map_location=lambda storage, loc: storage)  # 加载上一个状态的模型，
        num_classes_of_teacher1 = state1['num_classes']
        teacher_model1.fc = nn.Linear(teacher_model1.fc.in_features, num_classes_of_teacher1)
        teacher_model1.load_state_dict(state1['state_dict'])
        print('model_load_path1:', model_load_path1)

        teacher_model2 = _model_dict[opts.model]()
        model_load_path2 = os.path.join(opts.model_root,opts.dataset_name,str(opts.phases)+'phases','expert_models/expert_model_for_batch'+dataset_filename[-2:])
        state2 = torch.load(model_load_path2, map_location=lambda storage, loc: storage)  # 加载上一个状态的模型，
        num_classes_of_teacher2 = state2['num_classes']
        teacher_model2.fc = nn.Linear(teacher_model2.fc.in_features, num_classes_of_teacher2)
        teacher_model2.load_state_dict(state2['state_dict'])
        print('model_load_path2:', model_load_path2)

        teacher_model1.to(device)
        teacher_model2.to(device)
        teacher_model1.eval()
        teacher_model2.eval()


        print("Target student: %s" % opts.model)
        stu = _model_dict[opts.model]()
        stu.fc = nn.Linear(teacher_model1.fc.in_features, num_classes_of_teacher1)
        stu.load_state_dict(state1['state_dict'])
        stu.fc = nn.Linear(teacher_model1.fc.in_features, num_classes_of_teacher1+num_classes_of_teacher2)
        stu.to(device)
        params_1x = []
        params_10x = []
        for name, param in stu.named_parameters():
            if 'fc' in name:
                params_10x.append(param)
            else:
                params_1x.append(param)
        optimizer = torch.optim.Adam([{'params': params_1x, 'lr': opts.lr},
                                      {'params': params_10x, 'lr': opts.lr * 10}, ],
                                     lr=opts.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opts.epochs/2, gamma=0.1)
        criterion_ce = SoftCELoss(T=3.0)

        print("Training ...")
        # ===== Train Loop =====#
        cur_epoch = 0
        start_time = time.time()
        while cur_epoch < opts.epochs:
            top= AverageMeter()
            stu.train()
            epoch_loss = kd(cur_epoch=cur_epoch,
                            criterion_ce=criterion_ce,
                            model=stu,
                            teachers=[teacher_model1, teacher_model2],
                            optim=optimizer,
                            train_loader=train_loader,
                            device=device,
                            scheduler=scheduler)
            print("End of Epoch %d/%d, Average Loss=%f" %
                  (cur_epoch, opts.epochs, epoch_loss))

            # =====  Validation  =====
            print("validate on val set...")
            stu.eval()
            top = validate(model=stu,
                                 loader=val_loader,
                                 device=device,
                                 top=top)


            end_time = time.time()
            current_elapsed_time = end_time - start_time
            print('{:03}/{:03} | {} | Train : loss = {:.4f} | Val : accuracy = {}%'.
                  format(cur_epoch + 1, opts.epochs, timedelta(seconds=round(current_elapsed_time)),
                         epoch_loss, top.avg))
            sys.stdout.flush()
            cur_epoch += 1
            # =====  Save Best Model  =====
        state = {
            'state_dict': stu.state_dict(),
            'optimizer': optimizer.state_dict(),
            'num_classes': num_classes,
        }

        torch.save(state, save_path)
if __name__ == '__main__':
    main()
