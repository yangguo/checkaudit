import glob
import os

# import numpy as np
import pandas as pd

# import streamlit as st
# import torch
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

# from transformers import RoFormerModel, RoFormerTokenizer

# modelfolder = "junnyu/roformer_chinese_sim_char_ft_base"
auditfolder = "audit"
rulefolder = "rules"

# tokenizer = RoFormerTokenizer.from_pretrained(modelfolder)
# model = RoFormerModel.from_pretrained(modelfolder)


# Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     # First element of model_output contains all token embeddings
#     token_embeddings = model_output[0]
#     input_mask_expanded = (
#         attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     )
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
#         input_mask_expanded.sum(1), min=1e-9
#     )


# def roformer_encoder(sentences):
#     # Tokenize sentences
#     encoded_input = tokenizer(
#         sentences, max_length=512, padding=True, truncation=True, return_tensors="pt"
#     )

#     # Compute token embeddings
#     with torch.no_grad():
#         model_output = model(**encoded_input)

#     # Perform pooling. In this case, max pooling.
#     sentence_embeddings = mean_pooling(
#         model_output, encoded_input["attention_mask"]
#     ).numpy()
#     return sentence_embeddings


def audit2df(filename, filepath):
    docdf = pd.read_csv(filepath)
    docdf["监管要求"] = filename
    return docdf


# @st.cache
def get_csvdf(rulefolder):
    files2 = glob.glob(rulefolder + "**/*.csv", recursive=True)
    dflist = []
    for filepath in files2:
        basename = os.path.basename(filepath)
        filename = os.path.splitext(basename)[0]
        newdf = audit2df(filename, filepath)
        dflist.append(newdf)
    alldf = pd.concat(dflist, axis=0)
    # fillna
    alldf = alldf.fillna("")
    return alldf


# def get_embedding(folder, emblist):
#     dflist = []
#     for file in emblist:
#         filepath = os.path.join(folder, file + ".npy")
#         embeddings = np.load(filepath)
#         dflist.append(embeddings)
#     alldf = np.concatenate(dflist)
#     return alldf


# split string by space into words, add brackets before and after words, combine into text
def split_words(text):
    words = text.split()
    words = ["(?=.*" + word + ")" for word in words]
    new = "".join(words)
    return new


# get section list from df
def get_section_list(searchresult, make_choice):
    """
    get section list from df

    args: searchresult, make_choice
    return: section_list
    """
    df = searchresult[(searchresult["监管要求"].isin(make_choice))]
    conls = df["结构"].drop_duplicates().tolist()
    unils = []
    for con in conls:
        itemls = con.split("/")
        for item in itemls[:2]:
            unils.append(item)
    # drop duplicates and keep list order
    section_list = list(dict.fromkeys(unils))
    return section_list


# get folder name list from path
def get_folder_list(path):
    folder_list = [
        folder
        for folder in os.listdir(path)
        if os.path.isdir(os.path.join(path, folder))
    ]
    return folder_list


def get_auditfolder(industry_choice):
    # join folder with industry_choice
    folder = os.path.join(auditfolder, industry_choice)
    return folder


def get_rulefolder(industry_choice):
    # join folder with industry_choice
    folder = os.path.join(rulefolder, industry_choice)
    return folder


# use aggrid to display dataframe
def df2aggrid(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(enabled=True)
    # gb.configure_side_bar()
    # gb.configure_auto_height()
    gb.configure_default_column(
        # genablePivot=True,
        # enableValue=True,
        # enableRowGroup=True,
        # groupable=True,
        editable=True,
        # wrapText=True,
        # autoHeight=True,
    )
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    # configure column visibility
    # gb.configure_column(field="lawid", hide=True)
    # gb.configure_column(field="id", hide=True)
    # gb.configure_grid_options(rowHeight=60, headerHeight=30)
    gridOptions = gb.build()
    ag_grid = AgGrid(
        df,
        theme="material",
        # theme="streamlit",
        height=400,
        # height="100%",
        width="100%",
        # wrap_text=True,
        # editable=True,
        fit_columns_on_grid_load=True,  # fit columns to grid width
        gridOptions=gridOptions,  # grid options
        key="select_grid",  # key is used to identify the grid
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        # data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        #  update_mode=GridUpdateMode.NO_UPDATE,
        # enable_enterprise_modules=True
    )
    return ag_grid


def savedf(df, filename):
    basename = filename.split(".")[0]
    savename = basename + ".csv"
    savepath = os.path.join(auditfolder, savename)
    df.to_csv(savepath)
