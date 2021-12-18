import os, glob
import numpy as np
import pandas as pd
import streamlit as st
import torch
import asyncio

from textrank4zh import TextRank4Keyword, TextRank4Sentence
from sklearn.cluster import AgglomerativeClustering

from transformers import RoFormerModel, RoFormerTokenizer


modelfolder = 'junnyu/roformer_chinese_sim_char_ft_base'
auditfolder='audit'

tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
model = RoFormerModel.from_pretrained(modelfolder)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9)


# def async sent2emb(sentences):
def sent2emb_async(sentences):
    """
    run sent2emb in async mode
    """
    # create new loop
    loop = asyncio.new_event_loop()
    # run async code
    asyncio.set_event_loop(loop)
    # run code
    task = loop.run_until_complete(sent2emb(sentences))
    # close loop
    loop.close()
    return task


def roformer_encoder(sentences):
   # Tokenize sentences
    encoded_input = tokenizer(sentences,
                              max_length=512,
                              padding=True,
                              truncation=True,
                              return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(
        model_output, encoded_input['attention_mask']).numpy()
    return sentence_embeddings


@st.cache
def get_csvdf(rulefolder):
    files2 = glob.glob(rulefolder + '**/*.csv', recursive=True)
    dflist = []
    for filepath in files2:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        newdf = rule2df(filename, filepath)[['监管要求', '结构', '条款']]
        dflist.append(newdf)
    alldf = pd.concat(dflist, axis=0)
    return alldf


def rule2df(filename, filepath):

    docdf = pd.read_csv(filepath)
    docdf['监管要求'] = filename
    return docdf


def get_embedding(folder, emblist):

    dflist = []
    for file in emblist:
        filepath = os.path.join(folder, file + '.npy')
        embeddings = np.load(filepath)
        dflist.append(embeddings)
    alldf = np.concatenate(dflist)
    return alldf


async def sent2emb(sents):
    embls = []

    for sent in sents:
        # get summary of sent
        summarize = get_summary(sent)
        sentence_embedding = roformer_encoder(summarize)
        embls.append(sentence_embedding)
    # count += 1
    all_embeddings = np.concatenate(embls)
    return all_embeddings


# split string by space into words, add brackets before and after words, combine into text
def split_words(text):
    words = text.split()
    words = ['(?=.*' + word + ')' for word in words]
    new = ''.join(words)
    return new


# get summary of text
def get_summary(text):
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')

    sumls = []
    for item in tr4s.get_key_sentences(num=3):
        sumls.append(item.sentence)
    summary = ''.join(sumls)
    return summary


# get section list from df
def get_section_list(searchresult, make_choice):
    '''
    get section list from df
    
    args: searchresult, make_choice
    return: section_list
    '''
    df = searchresult[(searchresult['监管要求'].isin(make_choice))]
    conls = df['结构'].drop_duplicates().tolist()
    unils = []
    # print(conls)
    for con in conls:
        itemls = con.split('/')
        #     print(itemls[:-1])
        for item in itemls[:2]:
            unils.append(item)
    # drop duplicates and keep list order
    section_list = list(dict.fromkeys(unils))
    return section_list


# conver items to cluster
def items2cluster(df,threshold):
    corpus = df['条款'].tolist()
    # get embedding
    embeddings = sent2emb_async(corpus)
    # Normalize the embeddings to unit length
    corpus_embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               distance_threshold=threshold)
    #                                            affinity='cosine',
    #                                            linkage='complete',
    #                                            distance_threshold=0.5)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    clustered_idlist = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []
            clustered_idlist[cluster_id] = []
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        clustered_idlist[cluster_id].append(sentence_id)

    # reset index
    dfbefore=df.reset_index(drop=True)
    for key, value in clustered_idlist.items():
        dfbefore.loc[value, '分组'] = str(key)

    dfsort = dfbefore.sort_values(by='分组')
    clusternum=len(clustered_idlist.keys())
    return dfsort,clusternum


# get folder name list from path
def get_folder_list(path):
    folder_list = [
        folder for folder in os.listdir(path) if os.path.isdir(
            os.path.join(path, folder))
    ]
    return folder_list


def get_auditfolder(industry_choice):
    # join folder with industry_choice
    folder = os.path.join(auditfolder, industry_choice)
    return folder

