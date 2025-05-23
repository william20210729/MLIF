# coding=utf-8
 

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
from io import open

import torch

import pytorch_pretrained_bert.tokenization_transfo_xl as data_utils
from pytorch_pretrained_bert.modeling_transfo_xl import (CONFIG_NAME,
                                                         WEIGHTS_NAME,
                                                         TransfoXLConfig,
                                                         TransfoXLLMHeadModel,
                                                         load_tf_weights_in_transfo_xl)
from pytorch_pretrained_bert.tokenization_transfo_xl import (CORPUS_NAME,
                                                             VOCAB_NAME)

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

# We do this to be able to load python 2 datasets pickles
# See e.g. https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory/2121918#2121918
data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules['data_utils'] = data_utils
sys.modules['vocabulary'] = data_utils

def convert_transfo_xl_checkpoint_to_pytorch(tf_checkpoint_path,
                                             transfo_xl_config_file,
                                             pytorch_dump_folder_path,
                                             transfo_xl_dataset_file):
    if transfo_xl_dataset_file:
        # Convert a pre-processed corpus (see original TensorFlow repo)
        with open(transfo_xl_dataset_file, "rb") as fp:
            corpus = pickle.load(fp, encoding="latin1")
        # Save vocabulary and dataset cache as Dictionaries (should be better than pickles for the long-term)
        pytorch_vocab_dump_path = pytorch_dump_folder_path + '/' + VOCAB_NAME
        print("Save vocabulary to {}".format(pytorch_vocab_dump_path))
        corpus_vocab_dict = corpus.vocab.__dict__
        torch.save(corpus_vocab_dict, pytorch_vocab_dump_path)

        corpus_dict_no_vocab = corpus.__dict__
        corpus_dict_no_vocab.pop('vocab', None)
        pytorch_dataset_dump_path = pytorch_dump_folder_path + '/' + CORPUS_NAME
        print("Save dataset to {}".format(pytorch_dataset_dump_path))
        torch.save(corpus_dict_no_vocab, pytorch_dataset_dump_path)

    if tf_checkpoint_path:
        # Convert a pre-trained TensorFlow model
        config_path = os.path.abspath(transfo_xl_config_file)
        tf_path = os.path.abspath(tf_checkpoint_path)

        print("Converting Transformer XL checkpoint from {} with config at {}".format(tf_path, config_path))
        # Initialise PyTorch model
        if transfo_xl_config_file == "":
            config = TransfoXLConfig()
        else:
            config = TransfoXLConfig(transfo_xl_config_file)
        print("Building PyTorch model from configuration: {}".format(str(config)))
        model = TransfoXLLMHeadModel(config)

        model = load_tf_weights_in_transfo_xl(model, config, tf_path)
        # Save pytorch-model
        pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME)
        pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path, CONFIG_NAME)
        print("Save PyTorch model to {}".format(os.path.abspath(pytorch_weights_dump_path)))
        torch.save(model.state_dict(), pytorch_weights_dump_path)
        print("Save configuration file to {}".format(os.path.abspath(pytorch_config_dump_path)))
        with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
            f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the folder to store the PyTorch model or dataset/vocab.")
    parser.add_argument("--tf_checkpoint_path",
                        default = "",
                        type = str,
                        help = "An optional path to a TensorFlow checkpoint path to be converted.")
    parser.add_argument("--transfo_xl_config_file",
                        default = "",
                        type = str,
                        help = "An optional config json file corresponding to the pre-trained BERT model. \n"
                            "This specifies the model architecture.")
    parser.add_argument("--transfo_xl_dataset_file",
                        default = "",
                        type = str,
                        help = "An optional dataset file to be converted in a vocabulary.")
    args = parser.parse_args()
    convert_transfo_xl_checkpoint_to_pytorch(args.tf_checkpoint_path,
                                     args.transfo_xl_config_file,
                                     args.pytorch_dump_folder_path,
                                     args.transfo_xl_dataset_file)
