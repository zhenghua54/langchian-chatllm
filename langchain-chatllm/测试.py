from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import json
from starlette.middleware.cors import CORSMiddleware
import asyncio
import os

# pretrained_model_name_or_path = "/root/code/langchain-chatllm/path/to/my_local_model/Baichuan2-13B-Chat"
# print(pretrained_model_name_or_path)
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False,
#                                           trust_remote_code=True)
# print(f"模型加载中...")
#
# model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,
#                                              torch_dtype=torch.float16,
#                                              trust_remote_code=True
#                                              )
# model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path)
# if torch.cuda.is_available():
#     print(f"使用GPU-量化中...")
#     model = model.quantize(4).cuda()
#
# model = model.eval()

from chatllm import ChatLLM
from config import *
llm = ChatLLM()
llm.model_type = 'baichuan'
llm.model_name_or_path = llm_model_dict['baichuan']["BaiChuan2-13B-Chat"]
llm.load_llm(llm_device=LLM_DEVICE, num_gpus=num_gpus)


app = FastAPI()
app.add_middleware(  # 添加跨资源共享中间件
    CORSMiddleware,
    allow_origins=["*"],  # 这里可以指定允许所有的前端域名
    allow_credentials=True,  # 允许接收包含凭据的请求，这包括cookies，授权标头或TLS客户端证书等
    allow_methods=["*"],  # 接收所有请求方法
    allow_headers=["*"],  # 接收所有请求头
)


@app.get("/chat_stream")
async def chat_stream():
    query: str = "如何实现大模型的微调，请详细说明。"
    messages = [
        {"role": "user", "content": query}
    ]
    # ret = model.chat(tokenizer, messages, stream=True)
    ret = llm.generate(query)
    # async def predict():
    #     for token in ret:
    #         yield token
    #         await asyncio.sleep(0.1)
    async def stream_answer():
        for token in ret:
            yield json.dumps({
                "answer": token,
                "context": [],
            })
            await asyncio.sleep(0.1)

    response = StreamingResponse(stream_answer(),
                                 media_type="text/event-stream")
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=22507, reload=False, workers=1)