import json
import os
import nltk
import sentence_transformers
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from chatllm_copy import ChatLLM
from config import *
from py2neo import Graph

nltk.data.path.append('./nltk_data')
embedding_model_dict = embedding_model_dict
llm_model_dict = llm_model_dict
EMBEDDING_DEVICE = EMBEDDING_DEVICE
LLM_DEVICE = LLM_DEVICE
num_gpus = num_gpus
init_llm = init_llm
init_embedding_model = init_embedding_model
llm_model_list = []
init_base = init_base
host = host
port = port
llm_model_dict = llm_model_dict
knowledge_base_list = ["Vector", "Graph"]
for i in llm_model_dict:
    for j in llm_model_dict[i]:
        llm_model_list.append(j)

class KnowledgeBasedChatLLM:

    llm: object = None
    embeddings: object = None
    vector_store: object = None
    def init_model_config(
        self,
        large_language_model: str = init_llm,
        embedding_model: str = init_embedding_model,
    ):
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name=embedding_model_dict[embedding_model], cache_folder='/datasets/text2vec-base-chinese')
        # self.embeddings.client = sentence_transformers.SentenceTransformer(
        #     self.embeddings.model_name,
        #     device=EMBEDDING_DEVICE,
        #     cache_folder=os.path.join(MODEL_CACHE_PATH,
        #                               self.embeddings.model_name))
        self.llm = ChatLLM()
        if 'chatglm' in large_language_model.lower():
            self.llm.model_type = 'chatglm'
            self.llm.model_name_or_path = llm_model_dict['chatglm'][
                large_language_model]
        elif 'belle' in large_language_model.lower():
            self.llm.model_type = 'belle'
            self.llm.model_name_or_path = llm_model_dict['belle'][
                large_language_model]
        elif 'vicuna' in large_language_model.lower():
            self.llm.model_type = 'vicuna'
            self.llm.model_name_or_path = llm_model_dict['vicuna'][
                large_language_model]
        elif 'baichuan' in large_language_model.lower():
            self.llm.model_type = 'baichuan'
            self.llm.model_name_or_path = llm_model_dict['baichuan'][
                large_language_model]
        self.llm.load_llm(llm_device=LLM_DEVICE, num_gpus=num_gpus)

    def init_knowledge_vector_store(self):
        client = QdrantClient(url="http://10.90.1.19:22079")
        collection_name = "default_db"
        vector_store = Qdrant(client, collection_name, self.embeddings.embed_query)
        return vector_store

    async def get_vector_based_answer(self,
                                   query,
                                   vector_store,
                                   top_k: int = 6,
                                   history_len: int = 3,
                                   temperature: float = 0.01,
                                   top_p: float = 0.1,
                                   history=[]):
        self.llm.temperature = temperature
        self.llm.top_p = top_p
        self.history_len = history_len
        self.top_k = top_k
        prompt_template ="""基于以下已知信息，请简洁并专业地回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"。不允许在答案中添加编造成分。另外，答案请使用中文。

已知内容:
{context}

问题:
{query}"""
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "query"])
        self.llm.history = []
        if self.history_len > 0:
            self.llm.history = history[-self.history_len:]
        knowledge_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=vector_store.as_retriever(
                search_kwargs={"k": self.top_k}),
            prompt=prompt)
        knowledge_chain.combine_documents_chain.document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}")
        knowledge_chain.return_source_documents = False

        result = knowledge_chain({"query": query})
        return result

    async def get_graph_based_answer(self,
                               query,
                               top_k: int = 6,
                               history_len: int = 0,
                               temperature: float = 0.2,
                               top_p: float = 0.1,
                               history: list=[]):
        self.llm.temperature = temperature
        self.history_len = history_len
        self.llm.history = []
        print("------get_graph_based_answer函数------")
        answer_generate_prompt_template = """你是一位医药知识问答助手，拥有一个庞大的医药知识数据库。
你的任务是根据用户的问题以及与问题相关的从你的数据库中取出的材料进行专业的回复。
如果材料部分显示'无相关材料'，你应当回答'抱歉，我的知识库中好像没有相关知识，请问问别的吧！'
你需要从医生的角度结合材料与问题给出专业礼貌且明了的回答，为用户进行帮助与建议。
请你合理组织答案，不允许在答案中添加编造的成分或其他与问题无关的内容。
注意，材料中的内容是从知识库中取出的，与问题高度相关，请充分利用！
------------
材料：
{context}
------------
问题:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
{query}
------------
"""
        answer_generate_prompt_template_nodatabase = """你是一位医药知识问答助手，拥有一个庞大的医药知识数据库。
你的任务是根据用户的问题进行专业的回复。
你需要从医生的角度结合材料与问题给出专业礼貌且明了的回答，为用户进行帮助与建议。
请你合理组织答案，不允许在答案中添加编造的成分或其他与问题无关的内容。
------------
问题:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
{query}
------------
"""
        judge_prompt_template="""你的任务是根据所给的问题，判断其内容是否在字面意义上包含某一疾病：
    如果包含，请你回答该疾病的名称；
    如果不包含，请你回答'不相关'。
    注意，你的回答应该仅为疾病名称或'不相关'，不要做任何解释或添加无关内容，若包含有多种疾病，以空格' '进行区分。
    以下是一些例子：
-----------------------
        1.
        问题：病毒性流感的易感群体的主要特征是什么？
        回答：病毒性流感
        2.
        问题：感冒吃哪些药好的快？
        回答：感冒
        3.
        问题：如果今天是2077年9月9日，那么今天是星期几？
        回答：不相关
        4.
        问题：'你有一点笨'用英语怎么说？
        回答：不相关
        5.
        问题：有没有治疗心脏病的特效药？
        回答：心脏病
-----------------------
    问题如下：
    {query}
"""
        name = self.llm(judge_prompt_template.format(query=query))
        # name疾病名称
        print(f"---------判定结果---------\n{name}\n---------判定结果---------")
        if '不相关' in name:
            return "不好意思！问题不在我的专业领域内！\n以下内容来自百川大模型:\n" + self.llm(answer_generate_prompt_template_nodatabase.format(query=query)), None
        print("正在查询图数据库...")
        try:
            data = await self.get_nodes_by_name(name)
            print(f"查询结果:{data}")
        except Exception as e:
            print(f"查询失败！错误信息：{str(e)}")
            return "出错了！数据查询失败...",None

        if len(data['nodes'])==0:
            return "不好意思！数据库内无相关信息！\n以下内容来自百川大模型:\n" + self.llm(answer_generate_prompt_template_nodatabase.format(query=query)), None
        context = f"疾病'{name}'相关材料:\n"+self.context_generate(data)
        print(f"context：{context}")
        if self.history_len > 0:
            self.llm.history = history[-self.history_len:]
        t = answer_generate_prompt_template.format(context=context,query=query)
        print(f"prompt:{t}")
        result = self.llm(t)
        print(f"答案\n{result}")
        return result,data

    async def get_nodes_by_name(self,name):#根据name查询知识图谱
        cypher_statement = f"""MATCH (n)-[r]-(m)
WHERE n.name = '{name}'
WITH
  COLLECT(DISTINCT {{id: ID(n), name: n.name, category: labels(n)[0]}}) AS nodesa,
  COLLECT(DISTINCT {{id: ID(m), name: m.name, category: labels(m)[0]}}) AS nodesb,
  COLLECT(DISTINCT {{
    source: ID(n),
    target: ID(m),
    label: type(r),
    time: COALESCE(r.time, null)
  }}) AS links
WITH nodesa + nodesb AS nodes, links
RETURN nodes, links;
        """
        graph = Graph(
            host=host,
            port=port,
            user=user,
            password=pwd
        )
        res = graph.query(cypher_statement)
        return res.data()[0]
    async def get_nodes_by_id(self, id:int):#根据id查询知识图谱

        cypher_statement = f"""MATCH (n)-[r]-(m)
WHERE ID(n) = {id}
WITH
  COLLECT(DISTINCT {{id: ID(n), name: n.name, category: labels(n)[0]}}) AS nodesa,
  COLLECT(DISTINCT {{id: ID(m), name: m.name, category: labels(m)[0]}}) AS nodesb,
  COLLECT(DISTINCT {{
    source: ID(n),
    target: ID(m),
    label: type(r),
    time: COALESCE(r.time, null)
  }}) AS links
WITH nodesa + nodesb AS nodes, links
RETURN nodes, links;
        """
        graph = Graph(
            host=host,
            port=port,
            user=user,
            password=pwd
        )
        res = graph.query(cypher_statement)
        return res.data()[0]

    def context_generate(self, data):#整合知识图谱获得的材料
        nodes = {node['id']: node['name'] for node in data['nodes']}
        accompany_with = []
        common_drug = []
        do_eat = []
        has_symptom = []
        need_check = []
        no_eat = []
        recommend_drug = []
        recommend_eat = []
        print("正在整理...context整合")
        for link in data['links']:
            if link['label'] == 'accompany_with':
                accompany_with.append(nodes[link['target']])
            elif link['label'] == 'common_drug':
                common_drug.append(nodes[link['target']])
            elif link['label'] == 'do_eat':
                do_eat.append(nodes[link['target']])
            elif link['label'] == 'has_symptom':
                has_symptom.append(nodes[link['target']])
            elif link['label'] == 'need_check':
                need_check.append(nodes[link['target']])
            elif link['label'] == 'no_eat':
                no_eat.append(nodes[link['target']])
            elif link['label'] == 'recommend_drug':
                recommend_drug.append(nodes[link['target']])
            elif link['label'] == 'recommend_eat':
                recommend_eat.append(nodes[link['target']])
        common_drug_str = '、'.join(common_drug)
        recommend_drug_str = '、'.join(recommend_drug)
        no_eat_str = '、'.join(no_eat)
        do_eat_str = '、'.join(do_eat)
        recommend_eat_str = '、'.join(recommend_eat)
        has_symptom_str = '、'.join(has_symptom)
        accompany_with_str = '、'.join(accompany_with)
        need_check_str = '、'.join(need_check)
        context = f"""{{
    '治疗常用药物'：'{common_drug_str}',
    '治疗推荐药物'：'{recommend_drug_str}',
    '禁止食用的食物'：'{no_eat_str}',
    '可以食用的食物'：'{do_eat_str}',
    '推荐食用的食物'：'{recommend_eat_str}',
    '可能发病症状'：'{has_symptom_str}',
    '常见并发症'：'{accompany_with_str}',
    '常用检查检验措施'：'{need_check_str}',
}}"""
        return context


    def get_entities(self, input):#抽取实体并存入知识图谱
        schema = """
{
    "药物名称":"缺失",
    "药物成分":"缺失",
    "用法用量":"缺失",
    "适应症":"缺失"
}"""
        extract_prompt_template = """你的任务是根据药物说明书材料，补全SCHEMA中标记为'缺失'的属性，并以SCHEMA的原始格式(JSON)输出补全结果。
要求：
1）如果SCHEMA中的某一缺失属性在提供的材料中没有对应内容，则以'未提及'作为该属性的值。
2）以JSON格式输出结果。
注意，药物说明书材料的所有的内容都是在描述同一种药物。
注意，材料中描述了多种属性，你提取并补全在SCHEMA中出现并标记为'缺失'的属性，其他属性无需提取。
注意，‘药物成分’属性可能存在多个值，请以数组形式表示。其他属性仅单个值，请以字符串表示。

-------------
SCHEMA：
{schema}
-------------
药物说明书材料：
{input}
-------------
直接输出补全结果(JSON格式)，禁止做任何解释或添加其他无关内容!!!
"""
        print(f"SCHEMA:{schema}")
        extract_prompt = PromptTemplate(input_variables=["schema","input"],template=extract_prompt_template)
        llm_chain = LLMChain(prompt=extract_prompt, llm=self.llm)
        result = llm_chain.predict(schema=schema, input=input)
        print(f"result:{result}")
        try:
            jsonobj = json.loads(result)
            print(f"jsonobj:{jsonobj}")
            print("创建药物节点...")
            query = f"""
MERGE (a:testDrug {{name: '{jsonobj['药物名称']}'}})
ON CREATE SET a.name = '{jsonobj['药物名称']}', a.test_usage='{jsonobj['用法用量']}', a.test_indication='{jsonobj['适应症']}'
ON MATCH SET a.name = '{jsonobj['药物名称']}', a.test_usage='{jsonobj['用法用量']}', a.test_indication='{jsonobj['适应症']}'
"""
            self.graph.run(query)
            for name in jsonobj['药物成分']:
                print(f"创建药物成分节点:{name}...")
                query = f"""
MERGE (a:testPharmaceuticalIngredient {{name: '{name}'}})
ON CREATE SET a.name = '{name}'
ON MATCH SET a.name = '{name}'
"""
                self.graph.run(query)
                print(f"创建药物成分关系...")
                query = f"""
MATCH (a:testDrug {{name: '{jsonobj['药物名称']}'}})
MATCH (b:testPharmaceuticalIngredient {{name: '{name}'}})
MERGE (a)-[:test_hsa_ingredient {{name: '含药物成分'}}]->(b)
"""
                self.graph.run(query)
            print("数据导入成功")
        except Exception as e:
            print(f"数据导入失败！错误消息：{str(e)}")

    # def load_file(self, filepath):
    #     if filepath.lower().endswith(".md"):
    #         loader = UnstructuredFileLoader(filepath, mode="elements")
    #         docs = loader.load()
    #     elif filepath.lower().endswith(".pdf"):
    #         loader = UnstructuredFileLoader(filepath)
    #         textsplitter = ChineseTextSplitter(pdf=True)
    #         docs = loader.load_and_split(textsplitter)
    #     else:
    #         loader = UnstructuredFileLoader(filepath, mode="elements")
    #         textsplitter = ChineseTextSplitter(pdf=False)
    #         docs = loader.load_and_split(text_splitter=textsplitter)
    #     return docs

# def extract_entity(file_obj):
#     text = ""#从pdf提取文本
#     with open(file_obj.name, 'rb') as file:
#         pdf_reader = PyPDF2.PdfReader(file)
#         num_pages = len(pdf_reader.pages)
#         for page_num in range(num_pages):
#             page = pdf_reader.pages[page_num]
#             text += page.extract_text()
#     knowledge_based_chat_llm.get_entities(text)
#     # return ''

knowledge_based_chat_llm = KnowledgeBasedChatLLM()
knowledge_based_chat_llm.init_model_config()
print(knowledge_based_chat_llm.llm._call("你好!"))

