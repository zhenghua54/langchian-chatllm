import os

import torch

# device config
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

LLM_DEVICE = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
# LLM_DEVICE = "cpu"

MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'model_cache')

VS_ROOT_PATH = './'
num_gpus = torch.cuda.device_count()
init_base = "Graph"
#Graph database
host = '10.249.7.4' #http://10.90.1.19:22075/
port = '7687'
user= 'neo4j'
pwd = 'password'
# init model config
init_llm = "BaiChuan2-13B-Chat"
# init_llm = "BaiChuan2-13B-Chat-Int4"
# init_llm = "ChatGLM-6B-int8"
init_embedding_model = "text2vec-base"

# model config
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "text2vec-base": "/datasets/text2vec-base-chinese",#有
    'simbert-base-chinese': 'WangZeJun/simbert-base-chinese'
}
quant8_saved_dir = "baichuan-inc/Baichuan2-13B-Chat"
#quant8_saved_dir = "/datasets/baichuan2-13b-chat-int8"

llm_model_dict = {
    "chatglm": {
        "ChatGLM-6B": "THUDM/chatglm-6b",
        "ChatGLM-6B-int4": "THUDM/chatglm-6b-int4",
        "ChatGLM-6B-int8": "/datasets/chatglm-6b-int8",#有
        "ChatGLM-6b-int4-qe": "THUDM/chatglm-6b-int4-qe"
    },
    "belle": {
        "BELLE-LLaMA-Local": "/pretrainmodel/belle",
    },
    "vicuna": {
        "Vicuna-Local": "/pretrainmodel/vicuna",
    },
    "baichuan": {
        "BaiChuan2-7B": "baichuan-inc/Baichuan2-7B-Chat",
        "BaiChuan2-13B-Chat": "./path/to/my_local_model/Baichuan2-13B-Chat",
        #"BaiChuan2-13B-Chat": "/datasets/baichuan2-13B-chat",
        "BaiChuan2-13B-Chat-Int4": "/datasets/baichuan2-13B-chat-int4"#有
    }
}
