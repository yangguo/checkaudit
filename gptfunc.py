import json
import os

import numpy as np
import openai
import pandas as pd

# import faiss
import yaml
from dotenv import load_dotenv
from langchain import LLMChain

# from gpt_index import GPTSimpleVectorIndex, LLMPredictor, SimpleDirectoryReader
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders import DirectoryLoader

# from langchain.chains.question_answering import load_qa_chain
# from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from langchain.output_parsers import (
    ResponseSchema,
    RetryWithErrorOutputParser,
    StructuredOutputParser,
)
from langchain.prompts import PromptTemplate

# from langchain.indexes import VectorstoreIndexCreator
# from langchain.llms import OpenAIChat
from langchain.prompts.chat import (  # AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, SupabaseVectorStore
from supabase import Client, create_client

from upload import add_upload_folder

# import streamlit as st
# import pinecone


# config = yaml.safe_load(open("config.yaml"))

# uploadfolder = config["filepath"]["uploadpath"]
# filerawfolder = config["filepath"]["filerawpath"]
# fileidxfolder = config["filepath"]["fileidxpath"]
# ruleidxfolder = config["filepath"]["ruleidxpath"]


load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

# supabase: Client = create_client(supabase_url, supabase_key)


AZURE_BASE_URL = os.environ.get("AZURE_BASE_URL")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_DEPLOYMENT_NAME_16K = os.environ.get("AZURE_DEPLOYMENT_NAME_16K")
AZURE_DEPLOYMENT_NAME_GPT4 = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4")
AZURE_DEPLOYMENT_NAME_GPT4_32K = os.environ.get("AZURE_DEPLOYMENT_NAME_GPT4_32K")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")


# initialize pinecone
# pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# model_name = "shibing624/text2vec-base-chinese"
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# embeddings = HuggingFaceEmbeddings(model_name=model_name)

embeddings = HuggingFaceHubEmbeddings(
    repo_id=model_name,
    task="feature-extraction",
    huggingfacehub_api_token=HF_API_TOKEN,
)

# embeddings = OpenAIEmbeddings()

# embeddings = OpenAIEmbeddings(
#     deployment="ada02",
#     model="text-embedding-ada-002",
#     openai_api_type="azure",
#     openai_api_base=AZURE_BASE_URL,
#     openai_api_version="2023-05-15",
#     openai_api_key=AZURE_API_KEY)


# openai.api_base="https://tiny-shadow-5144.vixt.workers.dev/v1"
# openai.api_base="https://super-heart-4116.vixt.workers.dev/v1"
# openai.api_base = "https://az.139105.xyz/v1"
# openai.api_base = "https://op.139105.xyz/v1"

# llm = ChatOpenAI(model_name="gpt-3.5-turbo",
#                  openai_api_base="https://op.139105.xyz/v1",
#                  openai_api_key=api_key)
# llm = ChatOpenAI(model_name="gpt-3.5-turbo",
#                  openai_api_base="https://az.139105.xyz/v1",
#                  openai_api_key=AZURE_API_KEY)


# use azure model
# llm = AzureChatOpenAI(
#     openai_api_base=AZURE_BASE_URL,
#     openai_api_version="2023-07-01-preview",
#     deployment_name=AZURE_DEPLOYMENT_NAME,
#     openai_api_key=AZURE_API_KEY,
#     openai_api_type="azure",
# )

# convert gpt model name to azure deployment name
gpt_to_deployment = {
    "gpt-35-turbo": AZURE_DEPLOYMENT_NAME,
    "gpt-35-turbo-16k": AZURE_DEPLOYMENT_NAME_16K,
    "gpt-4": AZURE_DEPLOYMENT_NAME_GPT4,
    "gpt-4-32k": AZURE_DEPLOYMENT_NAME_GPT4_32K,
}


# use azure llm based on model name
def get_azurellm(model_name):
    deployment_name = gpt_to_deployment[model_name]
    llm = AzureChatOpenAI(
        openai_api_base=AZURE_BASE_URL,
        openai_api_version="2023-07-01-preview",
        deployment_name=deployment_name,
        openai_api_key=AZURE_API_KEY,
        openai_api_type="azure",
    )
    return llm


# use cohere model
# llm = Cohere(model="command-xlarge-nightly",cohere_api_key=COHERE_API_KEY)


# @st.cache_resource
def init_supabase():
    supabase: Client = create_client(supabase_url, supabase_key)
    return supabase


supabase = init_supabase()


def get_chatbot_response(text, model_name="gpt-35-turbo"):
    template = "你是一位十分善于解决问题、按步骤思考的专业咨询顾问。"
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = f"""
    针对监管要求编写审计步骤及相关内容：
    - '步骤编号'：'每一项审计工作的编号'
    - '审计工作'：'针对每一项监管要求，需要进行的具体审计工作'
    - '访谈问题'：'针对每一项审计工作，需要向被审计方提出的访谈问题'
    - '资料清单'：'针对每一项审计工作，需要准备的审计资料'

    监管要求使用'''进行间隔。

    输出的格式应为JSON列表，每个列表项为一个对象，包含以上四个字段："步骤编号"，"审计工作"，"访谈问题"，"资料清单"。字段的值应为具体的信息。

    以下是一个包含一个对象的输出样例：
    [{{{{"步骤编号": "1",
    "审计工作": "确认证券期货机构是否有相关文档分类管理制度",
    "访谈问题": "请描述证券期货机构的文档分类管理制度",
    "资料清单": "证券期货机构的文档分类管理制度" }}}}]

    监管要求：'''{text}'''
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    #     messages = [
    #     SystemMessage(content= "你是一位十分善于解决问题、按步骤思考的专业咨询顾问。"),
    #     # combine the prompt and the instruction with the end-of-sequence token
    #     HumanMessage(content=  "针对以下监管要求编写详细的审计步骤："+prompt),
    # ]
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # chat = ChatOpenAI(model_name=model_name)  # , max_tokens=512)

    chain = LLMChain(llm=get_azurellm(model_name), prompt=chat_prompt)
    # response = chat(chat_prompt.format_prompt(text=text).to_messages())
    response = chain.run(text=text)
    return response


# convert document list to pandas dataframe
def docs_to_df(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc = metadata["监管要求"]
        sec = metadata["结构"]
        row = {"条款": page_content, "监管要求": plc, "结构": sec}
        data.append(row)
    df = pd.DataFrame(data)
    return df


# convert industry chinese name to english name
def industry_name_to_code(industry_name):
    """
    Converts an industry name to an industry code.
    """
    industry_name = industry_name.lower()
    if industry_name == "银行":
        return "bank"
    elif industry_name == "保险":
        return "insurance"
    elif industry_name == "证券":
        return "securities"
    elif industry_name == "基金":
        return "fund"
    elif industry_name == "期货":
        return "futures"
    elif industry_name == "投行":
        return "invbank"
    elif industry_name == "反洗钱":
        return "aml"
    elif industry_name == "医药":
        return "pharma"
    else:
        return "other"


def convert_list_to_filter(lst):
    if len(lst) >= 1:
        return {"监管要求": lst[0]}
    else:
        return {}
        # return {"监管要求": {"$in": [item for item in lst]}}


# convert document list to pandas dataframe
def docs_to_df_audit(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc = metadata["source"]
        row = {"内容": page_content, "来源": plc}
        data.append(row)
    df = pd.DataFrame(data)
    return df


# convert json response to dataframe
def convert_json_to_df(response):
    data_dict = json.loads(response)

    # Convert dictionary to DataFrame
    # df = pd.DataFrame.from_dict(data_dict, orient='index')
    df = pd.DataFrame.from_dict(data_dict)

    return df


def merge_items(data):
    data_dict = json.loads(data)
    merged = {"审计工作": [], "访谈问题": [], "资料清单": []}

    for item in data_dict:
        for key in merged:
            merged[key].append(item["步骤编号"] + "." + item[key])

    for key in merged:
        merged[key] = ";".join(merged[key])

    return merged


def separate_items(input_list):
    audit_work_list = []
    interview_question_list = []
    document_list = []

    for item in input_list:
        audit_work_list.append(item["审计工作"])
        interview_question_list.append(item["访谈问题"])
        document_list.append(item["资料清单"])

    return audit_work_list, interview_question_list, document_list


def get_audit_steps(text, model_name="gpt-35-turbo"):
    response_schemas = [
        ResponseSchema(name="审计步骤", description="针对监管要求，需要执行的多项具体审计工作步骤"),
        ResponseSchema(name="访谈问题", description="针对监管要求，需要向被审计方提出的多项访谈问题"),
        ResponseSchema(name="资料清单", description="针对监管要求，需要被审计方准备的多项审计资料"),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    template = """
    你是一位具有10年资深经验的内部审计师，你的任务是根据监管要求生成审计工作计划。

    我需要你根据以下监管要求分解成审计目标，并针对这个审计目标编写详细的审计工作计划，并提供相关内容。内容包括：审计工作步骤、访谈问题、资料清单。

    所有的审计步骤、访谈问题和资料清单应当在一个完整的回复中给出。

    {format_instructions}

    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = """    
    监管要求的内容如下:
    {text}
    """

    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate(
        messages=[system_message_prompt, human_message_prompt],
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions},
    )
    llm = get_azurellm(model_name)

    chain = LLMChain(llm=llm, prompt=chat_prompt)
    response = chain.run(text=text)

    retry_parser = RetryWithErrorOutputParser.from_llm(parser=output_parser, llm=llm)

    try:
        response_json = output_parser.parse(response)
    except Exception as e:
        print(e)
        prompt_value = chat_prompt.format_prompt(text=text)
        response_json = retry_parser.parse_with_prompt(response, prompt_value)

    return response_json


def upload_to_df(docs):
    """
    Converts a list of documents to a pandas dataframe.
    """
    data = []
    for document in docs:
        page_content = document.page_content
        metadata = document.metadata
        plc = metadata["source"]
        row = {"条款": page_content, "来源": plc}
        data.append(row)
    df = pd.DataFrame(data)
    return df
