from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile
from py2neo import Graph
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PyPDF2 import PdfReader
from io import BytesIO

from app import knowledge_based_chat_llm,host,port,user,pwd

import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 这里可以指定允许的前端域名，* 表示允许所有
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
knowledge_based_chat_llm

class Item(BaseModel):
    question:str
    top_k: Optional[int]=5
    history_len: Optional[int]=0
    temperature: Optional[float]=0.2
    top_p: Optional[float]=0.7
    knowledgebase: Optional[str]="Graph"
    chat_history: Optional[list]=None


@app.get("/")
async def index():
    g = Graph(
        host=host,
        port=port,
        user=user,
        password=pwd
    )
    query = """
    CALL db.schema.visualization()
    """
    result = g.query(query)
    return {"schema": result.data()[0]}


@app.post("/predict")
async def predict(item: Item):
    # item_dict = json.loads(item)
    print(item)
    item_dict = item.dict()
    print(f"请求处理中...问题：{item_dict['question']}")
    question:str = item_dict['question']
    top_k:int = item_dict['top_k']
    top_p:float = item_dict['top_p']
    temperature:float = item_dict['temperature']
    history_len:int = item_dict['history_len']
    knowledgebase:str = item_dict['knowledgebase']
    history:list = item_dict['chat_history']
    if history is None:
        history = []
    if knowledgebase == 'Graph':
        # answer, data = await knowledge_based_chat_llm.get_graph_based_answer(question, top_k, history_len, temperature, top_p, history)
        result = await knowledge_based_chat_llm.get_graph_based_answer(question, top_k, history_len, temperature, top_p, history)
        if isinstance(result, tuple):
            def g():
                yield json.dumps({
                    "answer": result[0],
                    "context": result[1]
                })
            return StreamingResponse(g(),
                                     media_type="text/event-stream")

        return StreamingResponse(result,
                                 media_type="text/event-stream")
    else:
        return {"answer": "目前暂无向量数据库！", "context": []}


@app.get("/search")
async def search_by_id(id:str):
    print("id:"+id+'\n');
    data = await knowledge_based_chat_llm.get_nodes_by_id(int(id))
    return {"context":data}


@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    # 获取文件名、大小和内容
    file_name = file.filename
    contents = await file.read()
    file_size = len(contents)

    # 创建一个可以从中读取PDF内容的文件对象
    file_object = BytesIO(contents)

    # 尝试提取文本内容
    try:
        pdf = PdfReader(file_object)
        text_contents = ''
        for page in pdf.pages:
            text_contents += page.extract_text()
        print("文件内容（文本）: ", text_contents)
    except Exception as e:
        print("提取PDF文本内容时出错: ", str(e))

    # 获取文件的内容类型
    content_type = file.content_type

    data = await knowledge_based_chat_llm.get_nodes_by_pdf(text_contents)


    # 返回接收到的PDF信息
    return {"context":data}


    return
@app.get("/api")
async def llm_api(id:str):
    return
if __name__ == "__main__":
    uvicorn.run(app='main:app', host="0.0.0.0", port=22507, reload=False, workers=1)
