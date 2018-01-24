from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/home/yrinc/Desktop/data/VOC2012/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3 #　学习率


    # visualization
    env = 'faster-rcnn'  # 定义一个 visdom 的显示环境
    port = 8097 # visdom 的运行端口
    plot_every = 40  # vis every N iter; visualize prediction, loss etc every n batches.

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = False # Use Adam optimizer: use Adam instead of SGD, default SGD. (You need set a very low lr for Adam)
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead

    # debug
    debug_file = '/tmp/debugf'
    test_num = 10000

    # model
    load_path = None # Pretrained model path, default None, if it's specified, it would be loaded.
    caffe_pretrain = False # use caffe pretrained model instead of torchvision (Default: torchvison)
    caffe_pretrain_path = '' # 预训练模型的路径

    def _parse(self, kwargs): # 用来识别传递的参数并更新参数类
        state_dict = self._state_dict()
        for k, v in kwargs.items(): # 遍历传过来的词典的键值对
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict()) # pprint打印数据结构，格式比较漂亮
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}  # __dict__就是把类中的项作为词典输出出来，而且是不带_前缀的项


opt = Config() # 从类初始化一个实例
