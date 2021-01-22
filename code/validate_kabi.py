import torch
import argparse
import torch.nn as nn
from networks.rps_net_mlp import RPS_net_mlp
import torch.nn.functional as F
import os
from utils.logger import mkdir_if_missing, Logger
from networks.resnet18_for_cifer100 import resnet18
from utils.Utils import accuracy, get_val_loader
from utils.AverageMeter import AverageMeter
import sys
from torchvision import models


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='../datasets_folder')
    parser.add_argument("--dataset_name", type=str, default='cifar100')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--model_root", type=str, default='../model')
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--num_of_sample_group", type=int, default=2)
    parser.add_argument("--phases", type=int, default=10)

    return parser


_model_dict = {  # model dict
    'rps_net_mlp':RPS_net_mlp,
    'resnet18': resnet18,
    'standard_resnet18': models.resnet18

}


# compute  probabilities the data contunnum belong to each batch
# for discriminating 'batch' in  mini-batch gradient descent and 'batch' in batch identifier we change the name of batch to task
def predict_task(distinguish_tasks_model, images):
    outputs = distinguish_tasks_model(images)
    outputs = outputs.sum(0)
    return F.softmax(outputs)


def predict_class(distinguish_classes_model, images, task_probability, labels):  # 计算属于每个类别的概率
    opts = get_parser().parse_args()
    num_batch = len(images)//opts.batch_size + 1  # 一组图片分多少个batch训练完
    outputs=None
    for i in range(num_batch):
        outputs_part = distinguish_classes_model(images[opts.batch_size * i:opts.batch_size*(i+1)])  # 列表截取时，右侧下标大于列表长度也可
        class_num_for_single_task = outputs_part.shape[1]//task_probability.shape[0]  # 总类别个数/任务数=每个任务包含的类别个数
        outputs_part = F.softmax(outputs_part)  # output经过softmax正则化后的结果
        for task_id, p in enumerate(task_probability):
            outputs_part[:, task_id * class_num_for_single_task:(task_id + 1) * class_num_for_single_task] = outputs_part[: ,task_id * class_num_for_single_task:(task_id + 1) * class_num_for_single_task] * p
        if i == 0:
            outputs=outputs_part
        else:
            outputs=torch.cat((outputs, outputs_part), dim=0)  # 按行拼接
    return outputs


def predict(val_dataset_path_list, distinguish_tasks_model, distinguish_classes_model, num_of_sample_group, dataset_name):  # 预测图片的类别
    opts = get_parser().parse_args()
    top5 = AverageMeter()
    top1 = AverageMeter()
    for val_dataset_path in val_dataset_path_list:  # 分批次验证验证集
        val_loader = get_val_loader(opts.dataset_name, val_dataset_path,batch_size=opts.num_of_sample_group)
        for cur_step, (images, labels) in enumerate(val_loader):
            images = images.to(0, dtype=torch.float32)
            labels = labels.to(0, dtype=torch.long)
            task_probability = predict_task(distinguish_tasks_model, images)
            outputs = predict_class(distinguish_classes_model, images, task_probability,  labels)

            # ilsvrc2012 ——top5   mnist,cifar100——top1
            if opts.dataset_name == 'ilsvrc2012':
                _, prec5 = accuracy(outputs, labels, topk=(1,5))
                top5.update(prec5.item(), images.size(0))
            elif opts.dataset_name in ['mnist','cifar100']:
                prec1 = accuracy(outputs,labels)[0]
                top1.update(prec1.item(), images.size(0))

    return top5.avg if opts.dataset_name == 'ilsvrc2012' else top1.avg


def main():  # 验证流程
    # 读取参数
    opts = get_parser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dir and log
    mkdir_if_missing('../logs')
    sys.stdout = Logger(os.path.join('../logs', opts.dataset_name,str(opts.phases)+'phases', 'validate_kabi_result.txt'))

    distinguish_task_model_dir = os.path.join(opts.model_root, opts.dataset_name,str(opts.phases)+'phases','batch_identifiers')
    distinguish_task_model_filename_list = os.listdir(distinguish_task_model_dir)
    distinguish_task_model_path_list = []  # 所有专家模型路径
    for distinguish_task_model_filename in distinguish_task_model_filename_list:
        distinguish_task_model_path = os.path.join(distinguish_task_model_dir, distinguish_task_model_filename)
        distinguish_task_model_path_list.append(distinguish_task_model_path)

    distinguish_class_model_dir = os.path.join(opts.model_root, opts.dataset_name, str(opts.phases)+'phases','amalgamation_models')
    distinguish_class_model_filename_list = os.listdir(distinguish_class_model_dir)
    distinguish_class_model_path_list = []  # 所有融合模型路径
    for distinguish_class_model_filename in distinguish_class_model_filename_list:
        distinguish_class_model_path = os.path.join(distinguish_class_model_dir, distinguish_class_model_filename)
        distinguish_class_model_path_list.append(distinguish_class_model_path)

    val_dataset_dir = os.path.join(opts.data_root, opts.dataset_name, str(opts.phases)+'phases','separated/test/batch_test')
    val_dataset_filename_list = os.listdir(val_dataset_dir)
    val_dataset_path_list = []  # 所有验证数据集路径
    for val_dataset_filename in val_dataset_filename_list:
        val_dataset_path = os.path.join(val_dataset_dir, val_dataset_filename)
        val_dataset_path_list.append(val_dataset_path)
    val_dataset_path_list_for_all_tasks = []  # 组装验证集
    for i in range(len(distinguish_class_model_path_list)):
        val_dataset_path_list_for_all_tasks.append(val_dataset_path_list[:i+2])



    task_and_class_model_and_val_dataset_path_list = zip(
                                    distinguish_task_model_path_list,
                                    distinguish_class_model_path_list,
                                    val_dataset_path_list_for_all_tasks)
    for task_and_class_model_path in task_and_class_model_and_val_dataset_path_list:
        model_for_task = _model_dict[opts.model]()
        print('batch identifier path：', task_and_class_model_path[0])
        task_model_state = torch.load(task_and_class_model_path[0], map_location=lambda storage, loc: storage)
        num_batches = task_model_state['num_batches']
        model_for_task.fc = nn.Linear(model_for_task.fc.in_features, num_batches)
        model_for_task.load_state_dict(task_model_state['state_dict'])
        model_for_task.to(device)
        model_for_task.eval()

        model_for_class = _model_dict[opts.model]()
        class_model_state = torch.load(task_and_class_model_path[1], map_location=lambda storage, loc: storage)
        num_classes = class_model_state['num_classes']
        model_for_class.fc = nn.Linear(model_for_class.fc.in_features, num_classes)
        model_for_class.load_state_dict(class_model_state['state_dict'])
        model_for_class.to(device)
        model_for_class.eval()
        print('batch identifier path', task_and_class_model_path[0])
        print('amalgamation model path', task_and_class_model_path[1])
        print('datasets list：', task_and_class_model_path[2])
        print('accurcy：', predict(task_and_class_model_path[2], model_for_task, model_for_class, opts.num_of_sample_group, opts.dataset_name))

if __name__ == '__main__':
    main()
