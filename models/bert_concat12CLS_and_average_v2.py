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
        # self.attention = MultiHeadedAttention(12,  config.hidden_size).to(self.device) ## # 定义12个head，词向量维度为 768   
        self.fc = nn.Linear(config.hidden_size, config.num_classes) ### 768, 5


    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        hidden_states, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=True) ## 改为True
        ## returned, hidden_states  list(12) ; pooled, Tensor[2,768]

        # 取每一层encode出来的向量
        # outputs.pooler_output: [bs, hidden_size]

        # 12 CLSs
        cls_embeddings = hidden_states[0][:, 0, :].unsqueeze(1) # [bs, 1, hidden]  2,1,768
        # 将每一层的第一个token(cls向量)提取出来，拼在一起当作textcnn的输入
        xxx=hidden_states[12-self.layercount][:, 0, :].unsqueeze(1)
        for i in range(12-self.layercount+1, 11): ## pooled 单独处理
            cls_embeddings = torch.cat((cls_embeddings, hidden_states[i][:, 0, :].unsqueeze(1)), dim=1)
        # cls_embeddings: [bs, encode_layer=12, hidden]
        #print(cls_embeddings.shape) ## torch.Size([2, 12, 768])

        cls_embeddings = torch.cat((cls_embeddings, pooled.unsqueeze(1)), dim=1)

        ### lastlayer
        # lastlayer = hidden_states[-1][:, :, :]  # [bs, 1, hidden] ### 修改 hidden_states[0]  改为 hidden_states[-1]
        # last=lastlayer[0]
        # for i in range(1,self.pad_size):
        #     last = torch.cat((last, lastlayer[i].unsqueeze(1)), dim=1)


        # cat
        # d=torch.cat((cls_embeddings,lastlayer), dim=1) ### d shape: 2 26112
        # d=d.reshape(context.shape[0], (self.pad_size+self.layercount) , self.hidden_size)
        ### d shape: 2 26112
        #a = torch.rand(2,12,768)

        # print(d.shape)

        #out = self.fc(cls_embeddings)  ## out = self.fc(pooled)
        #out = self.fc(pooled) #
        #out = self.fc(avg_cls) # Tensor[2,768]
        #
        # att = self.attention(cls_embeddings.to(
        #     self.device))  ## RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x9216 and 768x768)



        results  =torch.mean(cls_embeddings,dim=1)

        out = self.fc(results)
        return out



