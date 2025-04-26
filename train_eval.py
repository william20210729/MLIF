# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained.optimization import BertAdam

def append_to_log(filename, msg):
    file_object = open(filename, 'a', encoding='utf-8')  # 打开文件
    file_object.write(msg)  # 写入数据
    file_object.close()  # 关闭文件

# 
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter, log_file_withpath):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)
    total_batch = 0  # 
    dev_best_loss = float('inf')
    last_improve = 0  # 
    flag = False  #
    model.train()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        append_to_log(log_file_withpath, 'Epoch [{}/{}]'.format(epoch + 1, config.num_epochs)+"\n")
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                #
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}, {7}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve, time.strftime('%Y-%m-%d %H:%M:%S')))
                append_to_log(log_file_withpath, msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve, time.strftime('%Y-%m-%d %H:%M:%S')) +"\n" )

                model.train()
            total_batch += 1

            if total_batch % 3000 == 0: ### 
                GPU_status(log_file_withpath)

            if total_batch - last_improve > config.require_improvement:
                # 
                print("No optimization for a long time, auto-stopping...")
                append_to_log(log_file_withpath, "No optimization for a long time, auto-stopping..." +"\n" )
                GPU_status(log_file_withpath)
                flag = True
                break
        if flag:
            break

        GPU_status(log_file_withpath)

    test(config, model, test_iter, log_file_withpath)


def test(config, model, test_iter, log_file_withpath):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    append_to_log(log_file_withpath,msg.format(test_loss, test_acc)+"\n")
    append_to_log(log_file_withpath,"Precision, Recall and F1-Score..."+"\n")
    append_to_log(log_file_withpath,str(test_report)+"\n")
    append_to_log(log_file_withpath,"Confusion Matrix..."+"\n")
    append_to_log(log_file_withpath,str(test_confusion)+"\n")
    append_to_log(log_file_withpath,"Time usage:"+ str(time_dif)+"\n")


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


from pynvml import nvmlDeviceGetHandleByIndex, nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetMemoryInfo, nvmlDeviceGetName, \
    nvmlDeviceGetTemperature, nvmlShutdown

def GPU_status(log_file_withpath):
    # 初始化
    nvmlInit()
    # 获取GPU个数
    deviceCount = nvmlDeviceGetCount()
    # 总显存
    total_memory = 0
    # 未用总显存
    total_free = 0
    # 已用总显存
    total_used = 0
    # 遍历查看每一个GPU的情况
    for i in range(deviceCount):
        # 创建句柄
        handle = nvmlDeviceGetHandleByIndex(i)
        # 获取信息
        info = nvmlDeviceGetMemoryInfo(handle)
        # 获取gpu名称
        gpu_name = nvmlDeviceGetName(handle)
        # 查看型号、显存、温度、电源
        print("[ GPU{}: {}".format(i, gpu_name), end="    ")
        print("总共显存: {}G".format((info.total // 1048576) / 1024), end="    ")
        print("空余显存: {}G".format((info.free // 1048576) / 1024), end="    ")
        print("已用显存: {}G".format((info.used // 1048576) / 1024), end="    ")
        print("显存占用率: {:.2%}".format( info.used / info.total), end="    ")
        print("运行温度: {}摄氏度 ]".format(nvmlDeviceGetTemperature(handle, 0)))

        total_memory += (info.total // 1048576) / 1024
        total_free += (info.free // 1048576) / 1024
        total_used += (info.used // 1048576) / 1024
    # 打印所有GPU信息
    msg = "显卡名称：[{}]，显卡数量：[{}]，总共显存；[{}G]，空余显存：[{}G]，已用显存：[{}G]，显存占用率：[{:.2%}]。".format(gpu_name, deviceCount, total_memory,
                                                                                                             total_free, total_used,
                                                                                                             (total_used / total_memory))

    append_to_log(log_file_withpath, msg)
    print(msg)

    # 关闭管理工具
    nvmlShutdown()