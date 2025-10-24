
STEPS：
1. Install software or packages（on Windows, Mac , or Linux）
a) python 3.8.5  
b) pytorch 1.11.0  
c) tqdm 4.50.2 
d) sklearn 0.23.2
e) tensorboardX 2.6
f) numpy 1.19.2
g) boto3 1.26.0
h) pynvml 11.5.0

2. Prepare code and data
a) Download code from https://github.com/william20210729/MLIF. Unzip the files. 
b) Download the base model file https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz . Unzip it，and then put it under the folder bert_pretrain/.
c) Prepare data and put data under the folder zongdiaosolution6/data/. See Section 4 for details, please.
d) If you are about to test using our data (The work order text data contains sensitive user information, so it is not published on the network.), please send an email to contact us lihuisxxa@163.com.
　　
3. Run
python run_MLIF_BERT.py

Before running the model, you can update parameters in the file run_MLIF_BERT.py：
　　### parameters settings
　　parser = argparse.ArgumentParser(description='the MLIF-BERT Model')
　　parser.add_argument('--model', type=str, default="MLIF_BERT", help='choose a model，models are put in the models folder') 
　　parser.add_argument('--num_epochs', type=str, default="3", help='set epoch') 
　　parser.add_argument('--batch_size', type=str, default="128", help='set batch size') 
　　parser.add_argument('--pad_size', type=str, default="32", help='set pad size') 
　　parser.add_argument('--learning_rate', type=str, default="5e-5", help='set learning rate') 
　　parser.add_argument('--dataset_folder', type=str, default='zongdiaosolution6', help='set the folder where data is（must have a data sub-folder）') 
　　parser.add_argument('--layercount', type=str, default="12", help='the number of top CLS')

4. Data format
Users need to prepare four files.

In the class.txt file, one category name per line (demo in Chinese).

In the dev.txt/test.txt/train.txt file, each line is a sample. First is the text content, then the Tab separator, and then the category number.


-----------------------------------

Ablation studies
 
The following code files used for ablation studies:
run_bert_bertbase_v2.py
run_bert_concat12CLS_and_attention_v2.py
run_bert_concat12CLS_and_average_v2.py
run_bert_concat12CLS_and_lastlayer_v2.py
run_bert_concat12CLS_and_max_v2.py
run_bert_concat12CLS_and_min_v2.py
run_bert_concat12CLS_and_sum_v2.py
run_bert_concat12CLS_v2.py
run_bert_lastlayer_and_attention_v2.py
run_bert_lastlayer_v2.py

-----------------------------------

