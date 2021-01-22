import torch.utils.data as data
from torchvision.datasets.folder import accimage_loader
from PIL import Image
import imghdr

import os, sys
import os.path


class ImagesListFileFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, images_list_file, transform=None, target_transform=None, return_path=False, is_gray=False):

        self.return_path = return_path
        images_list_file = open(images_list_file, 'r').readlines()  # 打开存储数据集的文件，读取所有的行
        samples = []  # [(image_path, image_class)]
        for e in images_list_file:  # 每一行执行如下操作
            e = e.strip()  # 移除字符串头尾指定字符(默认为空格或换行符)或字符序列
            image_path = e.split()[0]  # 通过指定分隔符对字符串进行切片，分隔符默认为所有的空字符，包括空格、换行、制表符等， 此处获得图片路径
            try:
                assert(os.path.exists(image_path))  # 判断图片是否存在
            except AssertionError:
                print('Cant find '+image_path)
                sys.exit(-1)
            image_class = int(e.split()[-1])  # 图片所属类别标号
            samples.append((image_path, image_class))  # sample是图片路径和图片所属类别组成的元组

        if len(samples) == 0:  # 没有图片
            raise(RuntimeError("No image found"))

        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.classes = list(set([e[1] for e in samples]))
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.start_label = min(self.targets)



    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)  # 图片
        if self.transform is not None:
            sample = self.transform(sample)  # 处理图片
        if self.target_transform is not None:
            target = self.target_transform(target)  # 处理图片目标类别

        if self.return_path:  # 如果为true， 则迭代的时候返回：（处理过的图片， 目标类别）， 图片路径
            return (sample, target), self.samples[index][0]  #
        return sample, target  # 如果为false， 则迭代的时候返回：处理过的图片， 目标类别

    def __len__(self):
        return len(self.samples)

    # 对象的自我描述函数，无论是print（对象）， 还是在控制台上直接输入对象，都可以按如下格式输出信息，
    # 如果不重写，则打印对象的内存地址
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        # fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        # return img.convert('RGB')

        return img.convert('L')



def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)

    else:
        return pil_loader(path)


