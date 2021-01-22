class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):  # prec1 or prec5  和 batch_size,
        self.val = val
        self.sum += val * n  # 准确度乘以batch_size是什么鬼
        self.count += n  # batch_size
        self.avg = self.sum / self.count  # 根据每个batch_size的val和n，计算一个epoch的平均准确度
