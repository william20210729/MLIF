# coding: UTF-8

###  
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

import time # 记录运行时间
print("#######################################################################################")
# 导入os模块，用于操作文件和路径
import os
# 初始化变量，用于存储文件名
filename = ''
# 获取当前运行文件的绝对路径
path = os.path.realpath(__file__)
# 使用os.path.basename()方法获取文件名
filename = os.path.basename(path)
print(filename)

start = time.time() # 记录运行时间

print("start  time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # 记录运行时间
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default="bert_concat12CLS_and_average_v2", help='choose a model: Bert') #####
parser.add_argument('--num_epochs', type=str, default="3", help='choose num_epochs') ##### 3
parser.add_argument('--batch_size', type=str, default="128", help='choose a batch_size') ##### 128
parser.add_argument('--pad_size', type=str, default="32", help='choose a pad_size') ##### 32
parser.add_argument('--learning_rate', type=str, default="5e-5", help='choose a learning_rate') #####
parser.add_argument('--dataset_folder', type=str, default='zongdiaosolution6', help='choose a dataset_folder') #####
parser.add_argument('--file_suffix', type=str, default="", help='运行指定日期时间点划分的数据集；不指定的话默认为空') ##### 
parser.add_argument('--layercount', type=str, default="12", help='使用顶上的几层CLS') #####


 
args = parser.parse_args()

layercount = int(args.layercount)
num_epochs = int(args.num_epochs)  # epoch数
batch_size = int(args.batch_size)  # mini-batch大小
pad_size = int(args.pad_size)  # 每句话处理成的长度(短填长切)
learning_rate = float(args.learning_rate) # 学习率  zongdiaosolution_midsize
dataset_folder = args.dataset_folder  # 数据集的文件夹    
model_name = args.model
file_suffix = args.file_suffix


original_data = 'original_data.txt' 
log_file = model_name + "_" + dataset_folder + file_suffix + '.log.txt'
file_suffix2 = "_runtime_" + time.strftime('%Y-%m-%d %H:%M:%S').replace("-","_").replace(":","_").replace(" ","_")
log_file = model_name + "_" + dataset_folder + file_suffix + file_suffix2    \
           +"_ep" + str(num_epochs)  +"_bs" + str(batch_size) \
                         +"_ps" + str(pad_size)  +"_lr" + str(learning_rate) +'.log.txt'
log_file_withpath =  r'./logs/'+ log_file
val_sample_count = 10000
test_sample_count = 10000

# 新建文件log

file_object = open(log_file_withpath, 'w', encoding='utf-8')  # 打开文件
file_object.close( )  # 关闭文件

def append_to_log(filename, msg):
    file_object = open(filename, 'a', encoding='utf-8')  # 打开文件
    file_object.write(msg+"\n")  # 写入数据
    file_object.close()  # 关闭文件


print("dataset_folder= ",dataset_folder)
print("data file_suffix= ",file_suffix)
print("file_suffix2= ",file_suffix2)
print("log_file_withpath= ",log_file_withpath)

append_to_log(log_file_withpath, " dataset_folder="+dataset_folder)
append_to_log(log_file_withpath, " data  file_suffix="+file_suffix)
append_to_log(log_file_withpath, " runtime file_suffix2="+file_suffix2)
append_to_log(log_file_withpath, " log_file_withpath="+log_file_withpath)
append_to_log(log_file_withpath, " runned filename="+filename)

append_to_log(log_file_withpath, "Loading data...")


if __name__ == '__main__':

    

    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset_folder,file_suffix, model_name, log_file_withpath \
            ,num_epochs, batch_size, pad_size , learning_rate, layercount )

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    append_to_log(log_file_withpath, "Loading data...")

    train_data, dev_data, test_data = build_dataset(config)

    # import random
    # random.shuffle(train_data)
    # random.shuffle(dev_data)
    # random.shuffle(test_data)
    # print("random.shuffle(train_data)")
    # append_to_log(log_file_withpath,"random.shuffle(train_data)" + '\n')

    print("build_dataset time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # 记录运行时间
    append_to_log(log_file_withpath, "build_dataset time: " + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    train_iter = build_iterator(train_data, config)
    print("build_iterator train_data time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # 记录运行时间
    append_to_log(log_file_withpath, "build_iterator train_data time: " + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    dev_iter = build_iterator(dev_data, config)
    print("build_iterator dev_data time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # 记录运行时间
    append_to_log(log_file_withpath, "build_iterator dev_data time: " +  time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    test_iter = build_iterator(test_data, config)
    print("build_iterator test_data time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # 记录运行时间
    append_to_log(log_file_withpath, "build_iterator test_data time: " + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter , dev_iter , test_iter, log_file_withpath )


end = time.time() # 记录运行时间
mystr="run time: %d seconds (=%f hours)" % ( end-start, (end-start)/3600) # 记录运行时间
print(mystr) # 记录运行时间
print("end  time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # 记录运行时间
append_to_log(log_file_withpath, mystr  )
append_to_log(log_file_withpath, "end  time: "+time.strftime('%Y-%m-%d %H:%M:%S') )
