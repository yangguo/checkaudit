import glob
import os
import shutil

import docx

# import numpy as np
import pandas as pd
import pdfplumber

# import scipy
import streamlit as st

# import faiss


# from utils import get_csvdf, get_embedding, sent2emb_async,roformer_encoder


uploadfolder = "uploads"
filerawfolder = "fileraw"
fileidxfolder = "fileidx"


def get_uploadfiles(uploadfolder):
    fileslist = glob.glob(uploadfolder + "/*.*", recursive=True)
    filenamels = []
    for filepath in fileslist:
        filename = os.path.basename(filepath)
        # name = filename.split('.')[0]
        filenamels.append(filename)
    return filenamels


def remove_uploadfiles(uploadfolder):
    files = glob.glob(uploadfolder + "/*.*", recursive=True)

    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            st.error("Error: %s : %s" % (f, e.strerror))


def savedf(txtlist, filename):
    df = pd.DataFrame(txtlist)
    df.columns = ["条款"]
    df["制度"] = filename
    df["结构"] = df.index
    basename = filename.split(".")[0]
    savename = basename + ".csv"
    savepath = os.path.join(uploadfolder, savename)
    df.to_csv(savepath)


def txt2df(filename, filepath):
    with open(filepath) as f:
        contents = f.readlines()
    f.close()
    text = "".join(contents)
    itemlist = text.replace(" ", "").split("\n")
    dflist = [item for item in itemlist if len(item) > 0]
    savedf(dflist, filename)


def get_txtdf():
    fileslist = glob.glob(uploadfolder + "**/*.txt", recursive=True)
    # get csv file name list
    csvfiles = get_csvfilelist()
    for filepath in fileslist:
        # get file name
        name = getfilename(filepath)
        if name not in csvfiles:
            txt2df(name, filepath)


# convert pdf to dataframe usng pdfplumber
def pdf2df_plumber(filename, filepath):
    result = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt != "":
                result += txt
    dflist = result.replace("\x0c", "").replace("\n", "").split("。")
    savedf(dflist, filename)


def get_pdfdf():
    fileslist = glob.glob(uploadfolder + "**/*.pdf", recursive=True)
    # get csv file name list
    csvfiles = get_csvfilelist()

    for filepath in fileslist:
        # get file name
        name = getfilename(filepath)
        # if name not in csvfiles
        if name not in csvfiles:
            pdf2df_plumber(name, filepath)


def doc2df(filename, filepath):
    # open connection to Word Document
    doc = docx.Document(filepath)
    # get all paragraphs in the document
    dflist = []
    for para in doc.paragraphs:
        txt = para.text
        if txt != "":
            dflist.append(txt)
    savedf(dflist, filename)


def get_docdf():
    fileslist = glob.glob(uploadfolder + "**/*.docx", recursive=True)
    # get csv file name list
    csvfiles = get_csvfilelist()
    for filepath in fileslist:
        # get file name
        name = getfilename(filepath)
        # if name not in csvfiles
        if name not in csvfiles:
            doc2df(name, filepath)


# return corpus_embeddings
def getfilename(file):
    filename = os.path.basename(file)
    name = filename.split(".")[0]
    return name


# def file2embedding(file):
#     df = pd.read_csv(file)
#     sentences = df['条款'].tolist()
#     # all_embeddings = sent2emb(sentences)
#     # use async to get embeddings
#     all_embeddings = sent2emb_async(sentences)
#     name = getfilename(file)
#     savename = name + '.npy'
#     savepath = os.path.join(uploadfolder, savename)
#     np.save(savepath, all_embeddings)


# def encode_plclist():
#     files = glob.glob(uploadfolder + '**/*.csv', recursive=True)
#     # get npy file name list
#     npyfiles = get_npyfilelist()
#     for file in files:
#         # get file name
#         name = getfilename(file)
#         # check if file is not in npy file list
#         if name not in npyfiles:
#             try:
#                 file2embedding(file)
#             except Exception as e:
#                 st.error(str(e))


# get npy file name list
def get_npyfilelist():
    files2 = glob.glob(uploadfolder + "**/*.npy", recursive=True)
    filelist = []
    for file in files2:
        name = getfilename(file)
        filelist.append(name)
    return filelist


# get csv file name list
def get_csvfilelist():
    files2 = glob.glob(uploadfolder + "**/*.csv", recursive=True)
    filelist = []
    for file in files2:
        name = getfilename(file)
        filelist.append(name)
    return filelist


# def upload_data():
#     try:
#         get_docdf()
#         get_txtdf()
#         get_pdfdf()
#         encode_plclist()
#     except Exception as e:
#         st.error(str(e))


def save_uploadedfile(uploadedfile):
    with open(os.path.join(uploadfolder, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("上传文件:{} 成功。".format(uploadedfile.name))


# def searchupload(text, ruledf, sentence_embeddings, top):
#     queries = [text]
#     query_embeddings = roformer_encoder(queries)

#     # emblist = ruledf['监管要求'].drop_duplicates().tolist()
#     # fix index
#     # fixruledf, _ = searchByItem(ruledf, emblist, '', '')
#     # get index of rule
#     # rule_index = fixruledf.index.tolist()

#     # get sub embedding
#     sub_embedding = sentence_embeddings#[rule_index]

#     avglist = []
#     idxlist = []
#     number_top_matches = top
#     for query, query_embedding in zip(queries, query_embeddings):
#         distances = scipy.spatial.distance.cdist([query_embedding],
#                                                  sub_embedding, "cosine")[0]

#         results = zip(range(len(distances)), distances)
#         results = sorted(results, key=lambda x: x[1])

#         for idx, distance in results[0:number_top_matches]:
#             idxlist.append(idx)
#             avglist.append(1 - distance)

#     return ruledf.iloc[idxlist]


def copy_files(file_list, source_folder, target_folder):
    for file_name in file_list:
        print(file_name)
        source_file = os.path.join(source_folder, file_name)
        target_file = os.path.join(target_folder, file_name)
        print(source_file)
        print(target_file)
        shutil.copy2(source_file, target_file)


def add_upload_folder(rule_choice):
    resls = []
    for rule in rule_choice:
        source = filerawfolder + "/" + rule
        resls.append(source)
    return resls


def remove_file(file_list, folder):
    for file_name in file_list:
        file_path = os.path.join(folder, file_name)
        os.remove(file_path)
