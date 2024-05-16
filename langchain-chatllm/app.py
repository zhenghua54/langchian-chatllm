import asyncio
import json
import os
import nltk
import sentence_transformers
from langchain.chains import RetrievalQA, LLMChain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from chatllm import ChatLLM
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
#         answer_generate_prompt_template = """你是一位医药知识问答助手，拥有一个庞大的医药知识数据库。
# 你的任务是根据用户的问题以及与问题相关的从你的数据库中取出的材料进行专业的回复。
# 如果材料部分显示'无相关材料'，你应当回答'抱歉，我的知识库中好像没有相关知识，请问问别的吧！'
# 你需要从医生的角度结合材料与问题给出专业礼貌且明了的回答，为用户进行帮助与建议。
# 请你合理组织答案，不允许在答案中添加编造的成分或其他与问题无关的内容。
# 注意，材料中的内容是从知识库中取出的，与问题高度相关，请充分利用！
# ------------
# 材料：
# {context}
# ------------
# 问题:
# {query}
# ------------
# """
#         answer_generate_prompt_template_nodatabase = """你是一位医药知识问答助手，拥有一个庞大的医药知识数据库。
# 你的任务是根据用户的问题进行专业的回复。
# 你需要从医生的角度结合材料与问题给出专业礼貌且明了的回答，为用户进行帮助与建议。
# 请你合理组织答案，不允许在答案中添加编造的成分或其他与问题无关的内容。
# ------------
# 问题:
# {query}
# ------------
# """
        answer_generate_prompt_template = """你是一位医药知识问答助手，拥有一个庞大的医药知识数据库。你的任务是根据用户的问题以及相关的材料给出精确的回答。当提供的材料包含用户问及的实体信息时，请直接使用这些信息来构建你的答案。如果材料中显示'无相关材料'，你应当回答'抱歉，目前我的知识库中没有关于此问题的信息。' 在回答时，请确保你的答案简洁明了，且仅根据材料提供的信息进行回复，不要依据自己的推测或添加无关内容。

注意：下面的材料是基于用户问题从知识库中检索到的，它们与用户的问题密切相关。请仔细阅读并且基于这些材料给出你的回答。

------------
材料：
{context}
------------

问题：
{query}

------------
根据上述材料，请给出你的回答：
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
#         judge_prompt_template="""你的任务是根据所给的问题，判断其内容是否在字面意义上包含某种或多种疾病或某个病人姓名：
#     如果包含，请你回答该疾病的名称或病人姓名；
#     如果不包含，请你回答'不相关'。
#     注意，你的回答应该仅为疾病名称或'不相关'，不要做任何解释或添加无关内容，若包含有多种疾病，以空格' '进行区分。
#     以下是一些例子：
# -----------------------
#         1.
#         问题：病毒性流感的易感群体的主要特征是什么？
#         回答：病毒性流感
#         2.
#         问题：感冒吃哪些药好的快？
#         回答：感冒
#         3.
#         问题：如果今天是2077年9月9日，那么今天是星期几？
#         回答：不相关
#         4.
#         问题：'你有一点笨'用英语怎么说？
#         回答：不相关
#         5.
#         问题：有没有治疗心脏病的特效药？
#         回答：心脏病
#         6.
#         问题：肺气肿和百日咳有什么相似的症状
#         回答：肺气肿 百日咳
#         7.
#         问题：张三患有什么病
#         回答：张三
#         8.
#         问题：李四有什么病
#         回答：李四
#         9.
#         问题：感冒了能吃榴莲吗
#         回答：感冒
# -----------------------
#     问题如下：
#     {query}
# """
        judge_prompt_template = """你的任务是仔细分析所给的问题，并提取问题中直接询问的所有实体。这些实体可能是疾病的名称、病人的姓名，或者和医疗、健康以及日常生活相关的物品，如食物。请直接给出实体名称，避免提供不必要的解释或额外信息。如果问题中的实体不在我们的知识库中，或者问题没有明确询问特定的实体，请回答'不相关'。
此任务要求你不仅能够识别人名和疾病等医学相关的实体，还要能准确识别日常生活中的物品，如食物。请特别注意问题的具体内容，确保识别和提取所有相关的实体。
请按照以下示例格式进行：
-----------------------
        示例1:
        问题：张三患有什么疾病？
        回答：张三
        示例2:
        问题：心脏病的病人能吃巧克力吗？
        回答：心脏病 巧克力
        示例3:
        问题：感冒了能吃海鲜吗？
        回答：感冒 海鲜
        示例4:
        问题：高血压病人有什么禁忌？
        回答：高血压
        示例5:
        问题：运动后应该补充哪种饮料？
        回答：不相关
-----------------------
现在，请依据以下问题给出你的回答：
{query}
        """

        name = self.llm(judge_prompt_template.format(query=query))
        # name疾病名称
        print(f"---------判定结果---------\n{name}\n---------判定结果---------")

        if '不相关' in name:
            async def stream_postprocess():
                for token in self.llm.stream(answer_generate_prompt_template_nodatabase.format(query=query)):
                    yield json.dumps({
                        "answer": "不好意思！问题不在我的专业领域内！\n以下内容来自百川大模型:\n" + token,
                        "context": [],
                    })
                    await asyncio.sleep(0.1)

            return stream_postprocess()
        print("正在查询图数据库...")
        try:
            data = await self.get_nodes_by_name(name)
            print(f"查询结果:{data}")
        except Exception as e:
            print(f"查询失败！错误信息：{str(e)}")
            return "出错了！数据查询失败...", None

        if len(data['nodes']) == 0:
            async def stream_postprocess0():
                for token in self.llm.stream(answer_generate_prompt_template_nodatabase.format(query=query)):
                    yield json.dumps({
                        "answer": "不好意思！数据库内无相关信息！\n以下内容来自百川大模型:\n" + token,
                        "context": [],
                    })
                    await asyncio.sleep(0.1)

            return stream_postprocess0()
        context = f"'{name}'相关材料:\n"+self.context_generate(data)
        print(f"context：{context}")
        if self.history_len > 0:
            self.llm.history = history[-self.history_len:]
        t = answer_generate_prompt_template.format(context=context, query=query)
        print(f"prompt:{t}")
        # 返回迭代器

        async def stream_answer():
            for token in self.llm.stream(t):
                yield json.dumps({
                    "answer": token,
                    "context": data,
                })
                await asyncio.sleep(0.1)

        return stream_answer()

#     async def get_nodes_by_name(self,name): #根据name查询知识图谱
#         cypher_statement = f"""MATCH (n)-[r]-(m)
# WHERE n.name = '{name}'
# WITH
#   COLLECT(DISTINCT {{id: ID(n), name: n.name, category: labels(n)[0]}}) AS nodesa,
#   COLLECT(DISTINCT {{id: ID(m), name: m.name, category: labels(m)[0]}}) AS nodesb,
#   COLLECT(DISTINCT {{
#     source: ID(n),
#     target: ID(m),
#     label: type(r),
#     time: COALESCE(r.time, null)
#   }}) AS links
# WITH nodesa + nodesb AS nodes, links
# RETURN nodes, links;
#         """
#         graph = Graph(
#             host=host,
#             port=port,
#             user=user,
#             password=pwd
#         )
#         res = graph.query(cypher_statement)
#         return res.data()[0]

    async def get_nodes_by_name(self, names):  # 根据names查询知识图谱，names是一个由空格分隔的字符串
        name_list = names.split(' ')  # 将输入的字符串拆分成列表
        cypher_statement = f"""
    MATCH (n)-[r]-(m)
    WHERE n.name IN {name_list!r} OR m.name IN {name_list!r}
    WITH n, m, r
    UNWIND [n, m] AS node
    WITH DISTINCT node, r
    WITH 
      COLLECT(DISTINCT {{id: ID(node), name: node.name, category: labels(node)[0]}}) AS nodes,
      COLLECT(DISTINCT {{
        source: ID(startNode(r)),
        target: ID(endNode(r)),
        label: type(r),
        time: COALESCE(r.time, null)
      }}) AS links
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

#     def context_generate(self, data):#整合知识图谱获得的材料
#         nodes = {node['id']: node['name'] for node in data['nodes']}
#         accompany_with = []
#         common_drug = []
#         do_eat = []
#         has_symptom = []
#         need_check = []
#         no_eat = []
#         recommend_drug = []
#         recommend_eat = []
#         SUFFERS_FROM = []
#         belongs_to =[]
#         print("正在整理...context整合")
#         for link in data['links']:
#             if link['label'] == 'accompany_with':
#                 accompany_with.append(nodes[link['target']])
#             elif link['label'] == 'common_drug':
#                 common_drug.append(nodes[link['target']])
#             elif link['label'] == 'do_eat':
#                 do_eat.append(nodes[link['target']])
#             elif link['label'] == 'has_symptom':
#                 has_symptom.append(nodes[link['target']])
#             elif link['label'] == 'need_check':
#                 need_check.append(nodes[link['target']])
#             elif link['label'] == 'no_eat':
#                 no_eat.append(nodes[link['target']])
#             elif link['label'] == 'recommend_drug':
#                 recommend_drug.append(nodes[link['target']])
#             elif link['label'] == 'recommend_eat':
#                 recommend_eat.append(nodes[link['target']])
#             elif link['label'] == 'SUFFERS_FROM':
#                 SUFFERS_FROM.append(nodes[link['target']])
#             elif link['label'] == 'belongs_to':
#                 belongs_to.append(nodes[link['target']])
#         common_drug_str = '、'.join(common_drug)
#         recommend_drug_str = '、'.join(recommend_drug)
#         no_eat_str = '、'.join(no_eat)
#         do_eat_str = '、'.join(do_eat)
#         recommend_eat_str = '、'.join(recommend_eat)
#         has_symptom_str = '、'.join(has_symptom)
#         accompany_with_str = '、'.join(accompany_with)
#         need_check_str = '、'.join(need_check)
#         SUFFERS_FROM_str = '、'.join(SUFFERS_FROM)
#         belongs_to_str = '、'.join(belongs_to)
#         context = f"""{{
#     '治疗常用药物'：'{common_drug_str}',
#     '治疗推荐药物'：'{recommend_drug_str}',
#     '禁止食用的食物'：'{no_eat_str}',
#     '可以食用的食物'：'{do_eat_str}',
#     '推荐食用的食物'：'{recommend_eat_str}',
#     '可能发病症状'：'{has_symptom_str}',
#     '常见并发症'：'{accompany_with_str}',
#     '常用检查检验措施'：'{need_check_str}',
#     '患有疾病'：'{SUFFERS_FROM_str}',
#     '所属科室'：'{belongs_to_str}',
# }}"""
#         return context

    def context_generate(self, data):  # 整合知识图谱获得的材料
        nodes = {node['id']: node for node in data['nodes']}
        link_attributes = {}
        print("正在整理...context整合")
        for link in data['links']:
            label = link['label']
            target_node = nodes[link['target']]
            # 如果link中有time属性，则获取，否则设为"未知"
            time = link.get('time', '未知')
            # 为每种关系类型的目标节点创建一个字典项
            if label not in link_attributes:
                link_attributes[label] = []
            # 添加目标节点的name和time属性
            link_attributes[label].append(f"{target_node['name']}（时间：{time}）")

        # 构建整合后的context字符串
        context_parts = []
        for label, nodes in link_attributes.items():
            nodes_str = '、'.join(nodes)
            context_parts.append(f"'{label}'：'{nodes_str}'")

        context = "{" + ",\n".join(context_parts) + "}"
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


    async def get_nodes_by_pdf(self, context:str):

        extract_prompt_template = f'''文本内容：
{context}

任务指南:
我们的目标是深入挖掘和提取文本中隐藏的实体关系三元组信息，同时减少重复内容，确保每个三元组都揭示了新的信息。以前的输出中已经包含了一些基础的三元组，现在请着重寻找其他尚未发现的实体间的关联和信息。在提取时，特别注意以下方向：

1. 药品与研究数据的关系，如疗效统计、试验结果等。
2. 与药品相关的历史信息，包括开发、注册和上市时间线。
3. 药品与患者治疗体验的关系，包括用药便利性、副作用管理等。
4. 关注药品在不同国家和地区的使用和认证情况，包括批准使用的地区、与其他药品的比较研究等。
5. 药品的创新点和它对目前医疗领域带来的影响，包括创新类型、对患者治疗方法的改变等。

请尝试从文本中提取上述内容相关的实体关系三元组，并避免重复已有的信息。期望输出如下所示：

示例三元组：
- 主体: 伊鲁阿克片
- 关系: 生产企业
- 客体: 齐鲁制药有限公司

- 主体: 伊鲁阿克片
- 关系: 上市许可持有人
- 客体: 齐鲁制药有限公司

- 主体: 伊鲁阿克片
- 关系: 首次上市时间
- 客体: 2023-06

[...]

请确保涵盖和提取尽可能多样化的三元组信息，以丰富我们对文本内容的理解和分析。'''


        extract_informations = self.llm(extract_prompt_template)
        print(extract_informations)
        return ''

knowledge_based_chat_llm = KnowledgeBasedChatLLM()
knowledge_based_chat_llm.init_model_config()
print(knowledge_based_chat_llm.llm._call("你好!"))

