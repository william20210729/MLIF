# coding=utf-8
 

from __future__ import absolute_import, division, print_function

import argparse
from io import open

import torch

from pytorch_pretrained_bert.modeling_openai import (CONFIG_NAME, WEIGHTS_NAME,
                                                     OpenAIGPTConfig,
                                                     OpenAIGPTModel,
                                                     load_tf_weights_in_openai_gpt)


def convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path):
    # Construct model
    if openai_config_file == "":
        config = OpenAIGPTConfig()
    else:
        config = OpenAIGPTConfig(openai_config_file)
    model = OpenAIGPTModel(config)

    # Load weights from numpy
    load_tf_weights_in_openai_gpt(model, openai_checkpoint_folder_path)

    # Save pytorch-model
    pytorch_weights_dump_path = pytorch_dump_folder_path + '/' + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + '/' + CONFIG_NAME
    print("Save PyTorch model to {}".format(pytorch_weights_dump_path))
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print("Save configuration file to {}".format(pytorch_config_dump_path))
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--openai_checkpoint_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path the TensorFlow checkpoint path.")
    parser.add_argument("--pytorch_dump_folder_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    parser.add_argument("--openai_config_file",
                        default = "",
                        type = str,
                        help = "An optional config json file corresponding to the pre-trained OpenAI model. \n"
                            "This specifies the model architecture.")
    args = parser.parse_args()
    convert_openai_checkpoint_to_pytorch(args.openai_checkpoint_folder_path,
                                         args.openai_config_file,
                                         args.pytorch_dump_folder_path)
