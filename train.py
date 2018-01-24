import os

import ipdb # 用于交互式调试bug
import matplotlib
from tqdm import tqdm # 进度条

from utils.config import opt # 引入定义的config类实例opt
from data.dataset import Dataset, TestDataset, inverse_normalize # 作者自己定义的Dataset类
from model import FasterRCNNVGG16

from torch.autograd import Variable
from torch.utils import data as data_ # 防止重名
from trainer import FasterRCNNTrainer
from utils import array_tool as at # 作者定义的转换类型工具
from utils.vis_tool import visdom_bbox # 作者定义的可视化工具
from utils.eval_tool import eval_detection_voc # 作者定义的评价工具
from chainer import cuda # 神奇操作,使用cupy4.0替换了cupy2.x中的memorypool,速度提升5%


# --------用于修复bug----------
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
# --------修复bug结束----------

matplotlib.use('agg') # agg貌似不能绘图


# 指定所用显卡
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '1' # 使用第二张显卡


def eval(dataloader, faster_rcnn, test_num=10000): # 评价函数
    pred_bboxes, pred_labels, pred_scores = list(), list(), list() # 预测结果
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list() # 真实结果
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)): # ii 用来指示当前进度
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes]) # 预测图片
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy()) # 这哥们英语一般
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True) # 计算预测好坏情况
    return result

def train(**kwargs): # *变量名, 表示任何多个无名参数, 它是一个tuple；**变量名, 表示关键字参数, 它是一个dict
    opt._parse(kwargs) # 识别参数,传递过来的是一个字典,用parse来解析

    dataset = Dataset(opt) # 作者自定义的Dataset类
    print('读取数据中...')

    # Dataloader 定义了一次获取批次数据的方法
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers) # PyTorch自带的DataLoader类,生成一个多线程迭代器来迭代dataset, 以供读取一个batch的数据
    testset = TestDataset(opt, split='trainval')

    # 测试集loader
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16() # 网络定义
    print('模型构建完毕!')

    trainer = FasterRCNNTrainer(faster_rcnn).cuda() # 定义一个训练器,返回loss, .cuda()表示把返回的Tensor存入GPU

    if opt.load_path: # 如果要加载预训练模型
        trainer.load(opt.load_path)
        print('已加载预训练参数 %s' % opt.load_path)
    else:
        print("未引入预训练参数, 随机初始化网络参数")

    trainer.vis.text(dataset.db.label_names, win='labels') # 显示labels标题
    best_map = 0 # 定义一个best_map

    for epoch in range(opt.epoch): # 对于每一个epoch

        trainer.reset_meters() # 重置测各种测量仪

        # 对每一个数据
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale) # 转化为标量
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda() # 存入GPU
            img, bbox, label = Variable(img), Variable(bbox), Variable(label) # 转换成变量以供自动微分器使用
            # TODO
            trainer.train_step(img, bbox, label, scale) # 训练一步

            if (ii + 1) % opt.plot_every == 0: # 如果到达"每多少次显示"
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

        # 使用测试数据集来评价模型(此步里面包含预测信息)
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map) # 好到一定程度就存储模型, 存储在checkpoint文件夹内

        if epoch == 9: # 到第9轮的时候读取模型, 并调整学习率
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)

        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        # if epoch == 13:  # 到第14轮的时候停止训练
        #     break
        
    trainer.save(best_map=best_map)

if __name__ == '__main__':

    import fire # 在命令行直接传入参数
    fire.Fire() # 括号内不加内容表示命令行可以调用任何函数or类
