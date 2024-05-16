from typing import Optional
import uvicorn
from fastapi import FastAPI
from py2neo import Graph
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from app_copy import knowledge_based_chat_llm,host,port,user,pwd

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
async def predict(item:Item):
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
        answer, data = await knowledge_based_chat_llm.get_graph_based_answer(question, top_k, history_len, temperature, top_p, history)
    else:
        answer = "目前暂无向量数据库！"
    print(answer)
    if data is None:
        data=[]
    return {"answer":answer,"context":data}
@app.get("/search")
async def search_by_id(id:str):
    data = await knowledge_based_chat_llm.get_nodes_by_id(int(id))
    return {"context":data}

@app.get("/api")
async def llm_api(id:str):
    return;
if __name__ == "__main__":
    uvicorn.run(app='main_copy:app', host="0.0.0.0", port=22507, reload=False, workers=1)
