"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""
import os
import json
import numpy as np
import torch 
from torch import nn, optim
from torch.optim import lr_scheduler
import csv

from opts import parse_opts

from model import generate_model
from models import multimodalcnn
import transforms 
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch

import time

import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    opt = parse_opts()
    test_accuracies = []
    
    if opt.device != 'cpu':
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'  

    pretrained = opt.pretrain_path != 'None'
    
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        
    time_stamp = time.time()
    pre_result_path = opt.result_path

    opt.result_path = os.path.join( pre_result_path, 
                                    str(time.time())+ \
                                    'get_confusion_matrix'+ \
                                    'lr_'+str(opt.learning_rate)+ \
                                    'seed_'+str(opt.manual_seed)+ \
                                    'optimizer_'+str(opt.optimizer)+ \
                                    'weight_decay_'+str(opt.weight_decay)
                                    )
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)

    print(opt)
    with open(os.path.join(opt.result_path, 'opts'+'.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file, separators=(',\n', ':')) # , separators=(',\n', ':')
    
    
    # 加载模型
    pretrain_state_path = '/home/ubuntu/work_space/multimodal-emotion-recognition-experiment/best_results/1678717134.4151022lr_0.00017341600515462103seed_42optimizer_AdamWweight_decay_0.001_best_complete/RAVDESS_multimodalcnn_15_best0.pth'
    pretrain_path = '/home/ubuntu/work_space/EfficientFace-master/checkpoint/Pretrained_EfficientFace.tar'
    model = multimodalcnn.MultiModalCNN(fusion = 'iaLSTM', pretr_ef = pretrain_path)
    pretrained_state = torch.load(pretrain_state_path)
    pretrained_state_dict = pretrained_state['state_dict']
    # 这里要将一些字符串替换掉才能得到合适的字典
    pretrained_state_dict = {key.replace("module.", ""): value for key, value in pretrained_state_dict.items()}
    model.load_state_dict(pretrained_state_dict)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(opt.device)
    
    best_prec1 = 0
    best_loss = 1e10

    model.cuda(opt.device)
            
    if opt.test:

        test_logger = Logger(
                os.path.join(opt.result_path, 'test'+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

        video_transform = transforms.Compose([
            transforms.ToTensor(opt.video_norm_value)])
            
        test_data = get_test_set(opt, spatial_transform=video_transform) 
    
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        
        model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end_time = time.time()

        preds = []
        output_targets = []
        for i, (inputs_audio, inputs_visual, targets) in enumerate(test_loader):
            data_time.update(time.time() - end_time)
            inputs_visual = inputs_visual.permute(0,2,1,3,4)
            inputs_visual = inputs_visual.reshape(inputs_visual.shape[0]*inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
            
            targets = targets.to(opt.device)
            with torch.no_grad():
                inputs_visual = Variable(inputs_visual)
                inputs_visual = inputs_visual.to(opt.device)
                inputs_audio = Variable(inputs_audio)
                inputs_audio = inputs_audio.to(opt.device)
                targets = Variable(targets)
            outputs = model(inputs_audio, inputs_visual)
            print(outputs.shape)
            print(targets.shape)
            print(torch.argmax(outputs, dim=1).shape)

            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            output_targets.extend(targets.cpu().numpy())
            print("preds type", type(preds))
            print("output_targets type", type(output_targets))
            loss = criterion(outputs, targets)
            prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
            top1.update(prec1, inputs_audio.size(0))
            top5.update(prec5, inputs_audio.size(0))

            losses.update(loss.data, inputs_audio.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
                'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                    10000,
                    i + 1,
                    len(test_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5))
            
        cm = confusion_matrix(output_targets, preds)
        print(cm)


        test_logger.log({'epoch': 10000,
                    'loss': losses.avg.item(),
                    'prec1': top1.avg.item(),
                    'prec5': top5.avg.item()})

        test_loss, test_prec1 = losses.avg.item(), top1.avg.item()
        
        with open(os.path.join(opt.result_path, 'test_set_bestval'+'.txt'), 'a') as f:
                f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
        test_accuracies.append(test_prec1) 
                
            
    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n')
    with open(os.path.join(opt.result_path, 'confusion_matrix.txt'), 'a') as f:
        writer = csv.writer(f)
        # 将二维list中的每一行写入文件
        for row in cm:
            writer.writerow(row)

    end_time = time.time()
    run_time = end_time - time_stamp

    hours = int(run_time // 3600)
    minutes = int((run_time - hours * 3600) // 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)

    print("程序运行时间为：%02d:%02d:%02d" % (hours, minutes, seconds))

"""
[[32  0  0  8  0  0  0  0]
 [ 0 80  0  0  0  0  0  0]
 [ 0  0 80  0  0  0  0  0]
 [ 2  4  6 68  0  0  0  0]
 [ 2  0  4  4 54  8  0  8]
 [ 2  0  3  8  0 55  0 12]
 [ 0  0  0  2  0  0 76  2]
 [ 0  0 27  0  0  0  0 53]]
"""