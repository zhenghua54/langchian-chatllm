import os
from typing import Dict, List, Optional, Tuple, Union

import torch
# from fastchat.conversation import (compute_skip_echo_len,
#                                    get_default_conv_template)
from fastchat.serve.inference import load_model as load_fastchat_model
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

from config import *
from path.to.my_local_model import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = LLM_DEVICE
DEVICE_ID = "1"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE

init_llm = init_llm
init_embedding_model = init_embedding_model




def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {
        'transformer.word_embeddings': 0,
        'transformer.final_layernorm': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


class ChatLLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []
    model_type: str = "chatglm"
    model_name_or_path: str = init_llm,
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        # if self.model_type == 'vicuna':
        #     # conv = get_default_conv_template(self.model_name_or_path).copy()
        #     # conv.append_message(conv.roles[0], prompt)
        #     # conv.append_message(conv.roles[1], None)
        #     # prompt = conv.get_prompt()
        #     # inputs = self.tokenizer([prompt])
        #     # output_ids = self.model.generate(
        #     #     torch.as_tensor(inputs.input_ids).cuda(),
        #     #     do_sample=True,
        #     #     temperature=self.temperature,
        #     #     max_new_tokens=self.max_token,
        #     # )
        #     # outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        #     # skip_echo_len = compute_skip_echo_len(self.model_name_or_path, conv, prompt)
        #     # response = outputs[skip_echo_len:]
        #     # torch_gc()
        #     # if stop is not None:
        #     #     response = enforce_stop_tokens(response, stop)
        #     pass
        #
        # elif self.model_type == 'belle':
        #     # prompt = "Human: " + prompt + " \n\nAssistant: "
        #     # input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        #     # generate_ids = self.model.generate(input_ids, max_new_tokens=self.max_token, do_sample=True, top_k=30,
        #     #                                    top_p=self.top_p, temperature=self.temperature, repetition_penalty=1.,
        #     #                                    eos_token_id=2, bos_token_id=1, pad_token_id=0)
        #     # output = \
        #     # self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #     # response = output[len(prompt) + 1:]
        #     # torch_gc()
        #     # if stop is not None:
        #     #     response = enforce_stop_tokens(response, stop)

        if self.model_type == 'chatglm':
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=self.history,
                max_length=self.max_token,
                temperature=self.temperature,
            )
            torch_gc()
            if stop is not None:
                response = enforce_stop_tokens(response, stop)
            # self.history = self.history + [[None, response]]
        elif self.model_type == 'baichuan':#未查询百川如何拼接聊天历史待实现

            messages = [{"role": "user", "content": prompt}]
            response = self.model.chat(self.tokenizer,
                                       messages)
            torch_gc()
            if stop is not None:
                response = enforce_stop_tokens(response, stop)
            # self.history = self.history + [[None,response]]
        return response

    def load_llm(self,
                 llm_device=DEVICE,
                 num_gpus='auto',
                 device_map: Optional[Dict[str, int]] = None,
                 **kwargs):
        if 'chatglm' in self.model_name_or_path.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True,
                                                           cache_dir=os.path.join(MODEL_CACHE_PATH,
                                                                                  self.model_name_or_path))
            print(f"模型'{self.model_type}'加载中...")
            print(f"llm_device:'{llm_device}'")
            if torch.cuda.is_available() and llm_device.lower().startswith("cuda"):
                # 根据当前设备GPU数量决定是否进行多卡部署
                if num_gpus <2 and device_map is None :
                    self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True,
                                                            cache_dir=os.path.join(MODEL_CACHE_PATH,
                                                                                   self.model_name_or_path),
                                                            **kwargs).half().cuda()
                else:
                    from accelerate import dispatch_model
                    model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True,
                                                      cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path),
                                                      **kwargs).half()
                    # 可传入device_map自定义每张卡的部署情况
                    if device_map is None:
                        device_map = auto_configure_device_map(num_gpus)

                    self.model = dispatch_model(model, device_map=device_map)
            else:
                self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True).float()
                #self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True).float()
            self.model = self.model.eval()
        elif 'baichuan' in self.model_name_or_path.lower():
            print(self.model_name_or_path)
            # print(local_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False,
                                                           trust_remote_code=True)
            #self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False,
                                                           # trust_remote_code=True)
            print(f"模型'{self.model_type}'加载中...")
            torch.cuda.set_device(0)  # 仅使用一块gpu
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path,
                                                              torch_dtype=torch.float16,
                                                              trust_remote_code=True)
            self.model.generation_config = GenerationConfig.from_pretrained(self.model_name_or_path)
            if torch.cuda.is_available():
                print(f"使用GPU-量化中...")
                self.model = self.model.quantize(4).cuda()

            self.model = self.model.eval()
        else:
            self.model, self.tokenizer = load_fastchat_model(
                model_path=self.model_name_or_path,
                device=llm_device,
                num_gpus=num_gpus
            )
            self.model = self.model.eval()

