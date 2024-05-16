from typing import Dict, List, Optional, Any, Iterator
from fastchat.serve.inference import load_model as load_fastchat_model
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
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
            # 可以清除这些未使用的内存，释放出你的GPU内存。但要注意的是，这个操作并不会将已经被张量占用的内存释放掉。
            torch.cuda.empty_cache()
            # 清理CUDA的进程间通信资源
            torch.cuda.ipc_collect()


# 根据指定的 GPU 数量 (num_gpus) 自动生成一个设备映射 (device_map) 字典，用于将模型的各个部分分配到不同的 GPU 上。
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
    model_type: str = "chatglm"  # 语言模型的类型
    model_name_or_path: str = init_llm,  # 语言模型的路径，会依据llm_model_dict中得到
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatLLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:  # prompt为提问文本
        if self.model_type == 'chatglm':
            response, _ = self.model.chat(
                self.tokenizer,
                prompt,
                history=self.history,
                max_length=self.max_token,
                temperature=self.temperature,
            )
            torch_gc()  # 释放资源
            if stop is not None:
                # 生成的文本中确保包含指定的停止标记
                # 1.遍历生成的文本（response），检查是否已经包含了指定的停止标记。如果已经包含，则表示文本已经结束，无需再添加停止标记。
                # 2.添加停止标记，如果生成的文本中未包含指定的停止标记，则在文本末尾添加停止标记，以确保生成的文本在语义上是完整的。
                # 3.返回处理后的文本
                response = enforce_stop_tokens(response, stop)
        elif self.model_type == 'baichuan': #未查询百川如何拼接聊天历史待实现

            messages = [{"role": "user", "content": prompt}]
            response = self.model.chat(self.tokenizer,
                                       messages)
            torch_gc()
            if stop is not None:
                response = enforce_stop_tokens(response, stop)
        return response

    def _stream(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        messages = [{"role": "user", "content": prompt}]
        response = self.model.chat(self.tokenizer,
                                   messages, stream=True)
        for i, r in enumerate(response):
            yield GenerationChunk(chunk_id=i, text=r)

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
                if num_gpus < 2 and device_map is None:
                    self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True,
                                                            cache_dir=os.path.join(MODEL_CACHE_PATH,
                                                                                   self.model_name_or_path),
                                                            **kwargs).half().cuda()
                else:  # 多卡情况
                    from accelerate import dispatch_model
                    model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True,
                                                      cache_dir=os.path.join(MODEL_CACHE_PATH, self.model_name_or_path),
                                                      **kwargs).half()
                    # 可传入device_map自定义每张卡的部署情况
                    if device_map is None:
                        device_map = auto_configure_device_map(num_gpus)

                    self.model = dispatch_model(model, device_map=device_map)
            else:  # 使用cpu
                self.model = AutoModel.from_pretrained(self.model_name_or_path, trust_remote_code=True).float()

            self.model = self.model.eval()
        elif 'baichuan' in self.model_name_or_path.lower():
            print(self.model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False,
                                                           trust_remote_code=True)
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

