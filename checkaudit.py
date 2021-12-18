from posixpath import join
import re
import os
import glob
import pandas as pd
import streamlit as st
import scipy
import numpy as np

from utils import split_words, roformer_encoder,get_auditfolder


def audit2df(filename, filepath):
    docdf = pd.read_csv(filepath)
    docdf['监管要求'] = filename
    return docdf


@st.cache
def get_csvdf(rulefolder):
    files2 = glob.glob(rulefolder + '**/*.csv', recursive=True)
    dflist = []
    # filelist = []
    for filepath in files2:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        newdf = audit2df(filename, filepath)
        dflist.append(newdf)
        # filelist.append(filename)
    alldf = pd.concat(dflist, axis=0)
    return alldf


def get_auditcol(industry_choice):
    if industry_choice in ['证券', '期货', '基金']:
        col = ['监管要求', '结构', '条款', '审计子程序', '资料','判断条件']
    elif industry_choice == '等级保护':
        col = ['监管要求', '结构', '序号', '条款', '审计子程序', '资料', '判断条件']
    elif industry_choice == '内审协会':
        col = ['监管要求', '结构', '条款', '审计子程序', '资料', '判断条件']
    return col


def list2text(ls):
    return ' '.join(ls)


def get_sampleaudit(key_list, industry_choice):
    auditfolder = get_auditfolder(industry_choice)
    col = get_auditcol(industry_choice)
    plcdf = get_csvdf(auditfolder)
    selectdf = plcdf[plcdf['监管要求'].isin(key_list)]
    tb_sample = selectdf[col]
    return tb_sample.reset_index(drop=True)


def item2text(item):
    return "".join(item).replace('\n', ' ')


def getresultls(results):
    resultnum = 3
    resultls = []
    for i in range(resultnum):
        resultstr = results["labels"][i] + "{0:.2f}".format(
            results["scores"][i])
        resultls.append(resultstr)
    return '| '.join(resultls)


def searchauditByName(search_text, industry_choice):
    rulefolder = get_auditfolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    plc_list = plcdf['监管要求'].drop_duplicates().tolist()

    choicels = []
    for plc in plc_list:
        if search_text in plc:
            choicels.append(plc)

    plcsam = get_sampleaudit(choicels, industry_choice)

    return plcsam, choicels


def searchauditByItem(searchresult, make_choice, column_text, item_text,
                      proc_text, pbc_text):

    # split item_text into item_list
    item_list = split_words(item_text)
    # split proc_text into proc_list
    proc_list = split_words(proc_text)
    # split pbc_text into pbc_list
    pbc_list = split_words(pbc_text)

    plcsam = searchresult[(searchresult['监管要求'].isin(make_choice))
                          & (searchresult['结构'].str.contains(column_text)) &
                          (searchresult['条款'].str.contains(item_list)) &
                          (searchresult['审计子程序'].str.contains(proc_list)) &
                          (searchresult['资料'].str.contains(pbc_list))]

    total = len(plcsam)
    # resdf = plcsam.reset_index(drop=True)
    return plcsam, total


def get_embedding(rulefolder, emblist):
    dflist = []
    for file in emblist:
        filepath = os.path.join(rulefolder, file + '.npy')
        embeddings = np.load(filepath)
        dflist.append(embeddings)
    alldf = np.concatenate(dflist)
    return alldf


def get_proc_embedding(rulefolder, emblist):
    dflist = []
    for file in emblist:
        filepath = os.path.join(rulefolder, file + '程序.npy')
        embeddings = np.load(filepath)
        dflist.append(embeddings)
    alldf = np.concatenate(dflist)
    return alldf


def searchaudit(text, section_text, make_choice, industry_choice, top, flag):
    # model = SentenceTransformer(sentmodel)
    queries = [text]
    # query_embeddings = model.encode(queries)
    query_embeddings = roformer_encoder(queries)

    searchdf = get_sampleaudit(make_choice, industry_choice)

    ruledf, _ = searchauditByItem(searchdf, make_choice, section_text, '', '',
                                  '')

    rulefolder = get_auditfolder(industry_choice)
    emblist = ruledf['监管要求'].unique().tolist()
    subsearchdf = get_sampleaudit(emblist, industry_choice)
    # fix index
    fixruledf, _ = searchauditByItem(subsearchdf, emblist, section_text, '',
                                     '', '')
    # get index of the rule
    ruledf_index = fixruledf.index.tolist()

    if flag:
        sentence_embeddings = get_proc_embedding(rulefolder, emblist)
    else:
        sentence_embeddings = get_embedding(rulefolder, emblist)
    # get sub-embedding by index
    sub_embeddings = sentence_embeddings[ruledf_index]

    avglist = []
    idxlist = []
    number_top_matches = top
    for query, query_embedding in zip(queries, query_embeddings):
        # distances = scipy.spatial.distance.cdist(
        #     [query_embedding], sentence_embeddings, "cosine")[0]
        distances = scipy.spatial.distance.cdist([query_embedding],
                                                 sub_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in results[0:number_top_matches]:
            idxlist.append(idx)
            avglist.append(1 - distance)
    return fixruledf.iloc[idxlist]


def searchls2df(search_list, section_text, make_choice, choice, top, flag):
    dfls = []
    # condition_list = []
    for search in search_list:
        resuledf = searchaudit(search, section_text, make_choice, choice, top,
                               flag)
        newdf = resuledf.fillna('')[['结构', '条款', '审计子程序', '资料']]
        dfls.append(newdf)

    return dfls

