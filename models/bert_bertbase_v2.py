# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel_newoutput, BertTokenizer
import time

def append_to_log(filename, msg):
    file_object = open(filename, 'a', encoding='utf-8')  # 打开文件
    file_object.write(msg)  # 写入数据
    file_object.close()  # 关闭文件

extcount = 0

class Config(object):

    """配置参数"""
    def __init__(self, dataset_folder,file_suffix, model_name, log_file_withpath \
            ,num_epochs, batch_size, pad_size , learning_rate, layercount ):

        self.log_file_withpath = log_file_withpath
        self.file_suffix   = file_suffix
        self.dataset_folder  =  dataset_folder
        self.model_name =  model_name
        self.train_path = dataset_folder + '/data/train' + file_suffix + '.txt'                                # 训练集
        self.dev_path = dataset_folder + '/data/dev' + file_suffix + '.txt'                                    # 验证集
        self.test_path = dataset_folder + '/data/test' + file_suffix + '.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset_folder + '/data/class.txt', encoding='utf-8').readlines()]                                # 类别名单

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                   # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)           # 类别数
        self.num_epochs = num_epochs  # epoch数
        self.batch_size = batch_size  # mini-batch大小
        self.pad_size = pad_size  # 每句话处理成的长度(短填长切)
        self.learning_rate = learning_rate  # 学习率
        self.layercount = layercount

        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.save_path = dataset_folder + '/saved_dict/' + self.model_name  \
                         + file_suffix +"_ep" + str(self.num_epochs)  +"_bs" + str(self.batch_size) \
                         +"_ps" + str(self.pad_size)  +"_lr" + str(self.learning_rate) + '.ckpt'  # 模型训练结果

        print(" \nself.num_epochs=" + str(self.num_epochs)+" \n ")
        print(" \nself.batch_size=" + str(self.batch_size)+" \n ")
        print(" \nself.pad_size=" + str(self.pad_size)+" \n ")
        print(" \nself.learning_rate=" + str(self.learning_rate)+" \n ")
        print(" \nself.train_path=", self.train_path)
        print(" \nself.save_path=", self.save_path)

        append_to_log(log_file_withpath, "\nself.save_path="+ self.save_path)
        append_to_log(log_file_withpath, "\nself.train_path="+ self.train_path)
        append_to_log(log_file_withpath, " \nself.num_epochs="+ str(self.num_epochs)+" \n ")
        append_to_log(log_file_withpath, " \nself.batch_size="+ str(self.batch_size)+" \n ")
        append_to_log(log_file_withpath, " \nself.pad_size="+ str(self.pad_size)+" \n ")
        append_to_log(log_file_withpath, " \nself.learning_rate="+ str(self.learning_rate)+" \n ")

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel_newoutput.from_pretrained(config.bert_path)
        self.pad_size=config.pad_size
        self.batchsize=config.batch_size
        self.hidden_size=config.hidden_size
        self.layercount=config.layercount
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备


        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)


    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        hidden_states, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=True) ## 改为True

        out = self.fc(pooled)
        return out


