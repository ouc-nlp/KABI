import os
import torch
import torch.cuda as tc
from torchvision import transforms
from utils.MyImageFolder import ImagesListFileFolder
from networks.resnet18_for_cifer100 import resnet18
import torch.nn as nn
from utils.AverageMeter import AverageMeter
from networks.rps_net_mlp import RPS_net_mlp
from torchvision import models
import argparse
from utils.Utils import accuracy, get_concat_val_loader

_model_dict = {
    'rps_net_mlp': RPS_net_mlp,
    'resnet18': resnet18,
    'standard_resnet18': models.resnet18
}
_dataset_class_number = {
    'mnist':10,
    'cifar100':100,
    'ilsvrc2012':1000
}
"""
    the accurcy of amalgamation models with batch labels

"""
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default='../datasets_folder')
    parser.add_argument("--model_root", type=str, default='../model')
    parser.add_argument("--dataset_name", type=str, default='cifar100')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model", type=str, default='resnet18')
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10)
    parser.add_argument("--phases", type=int, default=10)
    return parser


def main():
    opts = get_parser().parse_args()
    models_folder=os.path.join(opts.model_root,opts.dataset_name,str(opts.phases)+'phases','amalgamation_models')
    model_file_name_list=os.listdir(models_folder)
    for model_file_name in model_file_name_list:
        model_path=os.path.join(models_folder,model_file_name)
        model = _model_dict[opts.model]()
        state = torch.load(model_path, map_location=lambda storage, loc: storage)
        num_cls = state['num_classes']
        model.fc = nn.Linear(model.fc.in_features, num_cls)
        model.load_state_dict(state['state_dict'])
        model.cuda()
        model.eval()
        val_file_path_list = []
        val_dataset_folder = os.path.join(opts.data_root,opts.dataset_name,str(opts.phases)+'phases','separated/test/batch_test')
        val_dataset_filename_list = os.listdir(val_dataset_folder)
        for val_dataset_filename in val_dataset_filename_list[:int(model_file_name[-2:])]:
            val_file_path = os.path.join(val_dataset_folder,val_dataset_filename)
            val_file_path_list.append(val_file_path)
            val_loader = get_concat_val_loader(val_file_path_list,opts.dataset_name,val_dataset_folder, opts.batch_size)
        top = AverageMeter()
        class_num_each_phase = _dataset_class_number[opts.dataset_name] // opts.phases
        for data in val_loader:
            inputs, labels = data
            if tc.is_available():
                inputs, labels = inputs.cuda(0), labels.cuda(0)
                outputs = model(inputs)
                output_output = outputs
                for label_index in range(len(outputs)):
                    min_value = torch.min(outputs[label_index])
                    for phase in range(opts.phases):
                        if class_num_each_phase * phase <= labels[label_index] < class_num_each_phase * (phase + 1):
                            outputs[label_index][:class_num_each_phase * phase] = min_value - 1
                            outputs[label_index][class_num_each_phase * (phase + 1):] = min_value - 1
                if opts.dataset_name == 'ilsvrc2012':
                    # compute top-5 classification accuracy for ilsvrc 2012
                    _, prec = accuracy(outputs.data, labels, topk=(1, 5))
                else:
                    # compute top-1 classification accuracy for cifar100 and mnist
                    prec, _ = accuracy(outputs.data, labels, topk=(1, 2))
                top.update(prec.item(), inputs.size(0))  # inputs.size(0) = batch_size
        print(top.avg)


if __name__ == '__main__':
    main()