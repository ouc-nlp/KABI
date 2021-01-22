import os
from .MyImageFolder import ImagesListFileFolder
from torchvision import transforms
import torch
_normalize_dict = {
    'mnist': transforms.Normalize([0.1307], std=[0.3081]),
    'cifar100': transforms.Normalize([0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    'ilsvrc2012':transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
}
_train_transforms_compose_dict={
'mnist':transforms.Compose([  # 用transforms.Compose把多个图像预处理步骤整合到一起
                transforms.ToTensor(),  # 将像素为0-255的图片转换为0-1的值组成的张量
                 _normalize_dict['mnist']]),
'cifar100':transforms.Compose([  # 用transforms.Compose把多个图像预处理步骤整合到一起
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                _normalize_dict['cifar100']]),
'ilsvrc2012': transforms.Compose([  # 用transforms.Compose把多个图像预处理步骤整合到一起
                transforms.RandomResizedCrop(224),  # 将给定的图片裁切成随机尺寸和长宽比。最终resize成给定尺寸。
                transforms.RandomHorizontalFlip(),  # 随机水平翻转，一般概率翻转，一半概率不翻转
                transforms.ToTensor(),  # 将像素为0-255的图片转换为0-1的值组成的张量
                _normalize_dict['ilsvrc2012']])
}
_test_transforms_compose_dict={
'mnist':_train_transforms_compose_dict['mnist'],
'cifar100':_train_transforms_compose_dict['mnist'],
'ilsvrc2012':transforms.Compose([  # 用transforms.Compose把多个图像预处理步骤整合到一起
                transforms.Resize(256),  # 长宽中小者为256个像素，长宽比保持不变，自适应另一个
                transforms.CenterCrop(224),  # 在图片中央剪裁一个图片
                transforms.ToTensor(),  # 将像素为0-255的图片转换为0-1的值组成的张量
                _normalize_dict['ilsvrc2012']])
}


def get_concat_dataloader(dataset_name,train_file_list, val_file_list, batch_size, num_workers=4):
    train_file_path_list = []
    num_classes = 0
    start_label = 0
    for i, train_file_path in enumerate(train_file_list):
        train_dataset = ImagesListFileFolder(
            train_file_path,
            _train_transforms_compose_dict[dataset_name]
            )
        if i == 0:
            start_label = train_dataset.start_label
        num_classes += len(train_dataset.classes)
        train_file_path_list.append(train_dataset)
    train_datasets = torch.utils.data.dataset.ConcatDataset(train_file_path_list)  # 将所有的训练集拼接在一起
# ----------------------整理验证集-----------------------------------------------------------
    val_file_path_list = []
    for val_file_path in val_file_list:
        val_dataset = ImagesListFileFolder(
            val_file_path,
            _test_transforms_compose_dict[dataset_name])
        val_file_path_list.append(val_dataset)

    val_datasets = torch.utils.data.dataset.ConcatDataset(val_file_path_list)  # 将验证集拼接起来
    # 训练数据集加载器
    train_loader = torch.utils.data.DataLoader(
        train_datasets, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)
    # 验证数据集加载器
    val_loader = torch.utils.data.DataLoader(
        val_datasets, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, num_classes, start_label


def get_val_loader(dataset_name, val_dataset_path, batch_size):
    val_dataset = ImagesListFileFolder(
        val_dataset_path,
        _test_transforms_compose_dict[dataset_name])
    val_loader = torch.utils.data.DataLoader(  # 验证集加载器
        val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    return val_loader


def get_concat_val_loader(val_file_path_list, dataset_name, val_dataset_dir, batch_size):
    val_dataset_list = []
    for val_file_path in val_file_path_list:
        val_dataset = ImagesListFileFolder(
            val_file_path,
            _test_transforms_compose_dict[dataset_name])
        val_dataset_list.append(val_dataset)
    val_datasets = torch.utils.data.dataset.ConcatDataset(val_dataset_list)  #
    val_loader = torch.utils.data.DataLoader(
        val_datasets, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    return val_loader


def accuracy(output, target, topk=(1,)):  # topk=(1, min(5, b * P))  b*P是已有的类别，如果类别超过五，就取5个
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)  # 取topk中的最大值，已出现的类别数或者5
    batch_size = target.size(0)  # batch_size的值
    _, pred = output.topk(maxk, 1, True, True)  # pred为每个样本output_score最大的前maxk个的索引
    pred = pred.t()  # 转置，将所有样本的top1放在一行，所有样本的top2放在一行...  样本个数列，maxk行
    # pred每个元素位置都有一个bool值，true or false
    correct = pred.eq(target.view(1, -1).expand_as(pred)) # 将target展成一行，batch_size列,拓展成k行，每行元素相同
    res = []
    for k in topk:  # k取1和（5或已出现类别的小者）
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)  # 一个batch_size中验证数据预测正确的样本，如果再除以batch_size就是预测准确率
        res.append(correct_k.mul_(100.0 / batch_size))  # 返回的是去掉%号后的值，如98%，返回为98，最大为[100,100]
    return res  # 在prec5计算准确率的时候，只要预测的五个标签中有正确标签则算预测正确，预测正确的样本数+1