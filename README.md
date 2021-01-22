
# Pytorch implementation of KABI: Class-Incremental Learning via knowledge Amalgamation and Batch Identification


## 实验结果复现：
### 一、图片数据集下载与整理： 
  1. 下载mnist，cifar 100数据集，并将每个类的图片整理成一个文件夹。文件夹以类名来命名。下载ilsvrc2012数据集。
  2. 将dataset_floder中整理的图片路径换成自己电脑上真实有效的路径。
### 二、代码运行顺序
  1. 运行train_expert_models.py训练expert models
  2. 运行train_amalgamatin_models.py训练amalgamatin_models
  3. 运行train_batch_identifiers.py训练batch_identifiers
  4. 运行validate_kabi.py求整个KABI方法的准确率
### 三、如果想要联合训练mnist数据集，可用joint_train.py
### 四、如果想验证最后一个消融实验中amalgamtion with batch label 的效果，可用amalgamtion_with_batch_label.py，注意这个需要在训练完amalgamatin_models后执行
### 五、参数设置：
- mnist:    --dataset_name mnist --model rps_net_mlp --end 5 --phases 5
- cifar100(5次增量):    --dataset_name cifar100 --model resnet18 --end 5 --phases 5
- cifar100(10次增量):    --dataset_name cifar100 --model resnet18 --end 10 --phases 10
- cifar100(20次增量):    --dataset_name cifar100 --model resnet18 --end 20 --phases 20
- ilsvrc2012:           --dataset_name ilsvrc2012 --model standard_resnet18 --end 10 --phases 10
- validate_kabi.py中的参数--num_of_sample_group代表data continuum size,可自行设置

### 六、项目中自带有mnist数据集训练得到了model，log等信息，可自行取用
