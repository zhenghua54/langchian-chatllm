import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
from starlette.middleware.cors import CORSMiddleware
from time import sleep
import os
from typing import Generator
import asyncio
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

pretrained_model_name_or_path = "/root/code/langchain-chatllm/path/to/my_local_model/Baichuan2-13B-Chat"
print(pretrained_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False,
                                          trust_remote_code=True)
print(f"模型加载中...")

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path,
                                             torch_dtype=torch.float16,
                                             trust_remote_code=True
                                             )
model.generation_config = GenerationConfig.from_pretrained(pretrained_model_name_or_path)
if torch.cuda.is_available():
    print(f"使用GPU-量化中...")
    model = model.quantize(4).cuda()

model = model.eval()
app = FastAPI()
app.add_middleware(  # 添加跨资源共享中间件
    CORSMiddleware,
    allow_origins=["*"],  # 这里可以指定允许所有的前端域名
    allow_credentials=True,  # 允许接收包含凭据的请求，这包括cookies，授权标头或TLS客户端证书等
    allow_methods=["*"],  # 接收所有请求方法
    allow_headers=["*"],  # 接收所有请求头
)





app = FastAPI()

@app.get("/chat_stream")
async def chat_stream() -> StreamingResponse:
    query: str = "你是谁"
    messages = [
        {"role": "user", "content": query}
    ]
    ret = model.chat(tokenizer, messages, stream=True)

    async def predict() -> Generator:
        pre = 0
        try:
            while True:
                try:
                # 在独立线程中执行同步迭代器的 __next__ 方法
                    token = await asyncio.to_thread(next, ret)
                    new_data = token[pre:]
                    # 确保消息符合 SSE 格式
                    yield f"data: {new_data}\n\n"
                    pre += len(new_data)
                except StopIteration:
            # 迭代器耗尽，不需要特别处理
                    break
        except StopIteration:
            # 迭代器耗尽，不需要特别处理
            return

    # 返回 StreamingResponse 对象作为响应
    return StreamingResponse(predict(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    query: str = "你是谁"
    messages = [
        {"role": "user", "content": query}
    ]
    ret = model.chat(tokenizer, messages, stream=True)
    s = 0
    for i in ret:
        print(i[s:])
        s = len(i)


    uvicorn.run(app, host="0.0.0.0", port=22507, reload=False, workers=1)