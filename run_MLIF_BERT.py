# coding: UTF-8

###  
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import time 
print("#######################################################################################")
import os
filename = ''
path = os.path.realpath(__file__)
filename = os.path.basename(path)
print(filename)
start = time.time() 
print("start  time: ",time.strftime('%Y-%m-%d %H:%M:%S')) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### parameters settings
parser = argparse.ArgumentParser(description='the MLIF-BERT Model')
parser.add_argument('--model', type=str, default="MLIF_BERT", help='choose a model，models are put in the models folder') 
parser.add_argument('--num_epochs', type=str, default="3", help='set epoch') 
parser.add_argument('--batch_size', type=str, default="128", help='set batch size') 
parser.add_argument('--pad_size', type=str, default="32", help='set pad size') 
parser.add_argument('--learning_rate', type=str, default="5e-5", help='set learning rate') 
parser.add_argument('--dataset_folder', type=str, default='zongdiaosolution6', help='set the folder where data is（must have a data sub-folder）') 
parser.add_argument('--layercount', type=str, default="12", help='the number of top CLS') 


 

args = parser.parse_args()

layercount = int(args.layercount)
num_epochs = int(args.num_epochs)  # number of epoch 
batch_size = int(args.batch_size)  # mini-batch 
pad_size = int(args.pad_size)  # the new length of each text (cutting down if too long; pading if too short )
learning_rate = float(args.learning_rate) # learning_rate
dataset_folder = args.dataset_folder  # folder of data
model_name = args.model
file_suffix = ""

original_data = 'original_data.txt' 
log_file = model_name + "_" + dataset_folder + file_suffix + '.log.txt'
file_suffix2 = "_runtime_" + time.strftime('%Y-%m-%d %H:%M:%S').replace("-","_").replace(":","_").replace(" ","_")
log_file = model_name + "_" + dataset_folder + file_suffix + file_suffix2    \
           +"_ep" + str(num_epochs)  +"_bs" + str(batch_size) \
                         +"_ps" + str(pad_size)  +"_lr" + str(learning_rate) +'.log.txt'
log_file_withpath =  r'./logs/'+ log_file
val_sample_count = 10000
test_sample_count = 10000

# New a log file

file_object = open(log_file_withpath, 'w', encoding='utf-8')  # open the file
file_object.close( )  # close the file

def append_to_log(filename, msg):
    file_object = open(filename, 'a', encoding='utf-8')  # open the file
    file_object.write(msg+"\n")  # 
    file_object.close()  # close the file


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

    ##


    model_name = args.model  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset_folder,file_suffix, model_name, log_file_withpath \
            ,num_epochs, batch_size, pad_size , learning_rate, layercount )

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # to get same results for each run

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

    print("build_dataset time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # record the run-time
    append_to_log(log_file_withpath, "build_dataset time: " + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    train_iter = build_iterator(train_data, config)
    print("build_iterator train_data time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # record the run-time
    append_to_log(log_file_withpath, "build_iterator train_data time: " + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    dev_iter = build_iterator(dev_data, config)
    print("build_iterator dev_data time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # record the run-time
    append_to_log(log_file_withpath, "build_iterator dev_data time: " +  time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    test_iter = build_iterator(test_data, config)
    print("build_iterator test_data time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # record the run-time
    append_to_log(log_file_withpath, "build_iterator test_data time: " + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter , dev_iter , test_iter, log_file_withpath )


end = time.time() # record the run-time
mystr="run time: %d seconds (=%f hours)" % ( end-start, (end-start)/3600) # record the run-time
print(mystr) # record the run-time
print("end  time: ",time.strftime('%Y-%m-%d %H:%M:%S')) # record the run-time
append_to_log(log_file_withpath, mystr  )
append_to_log(log_file_withpath, "end  time: "+time.strftime('%Y-%m-%d %H:%M:%S') )
