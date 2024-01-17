# import scipy

import pandas as pd

from gptfunc import industry_name_to_code, init_supabase
from utils import (  # roformer_encoder; get_embedding,;
    get_csvdf,
    get_rulefolder,
    split_words,
)

supabase = init_supabase()


rulefolder = "rules"
secpath = "rules/sec1.csv"
plcpath = "rules/lawdfall0507.csv"
metapath = "rules/lawmeta0517.csv"
dtlpath = "rules/lawdtl0517.csv"
orgpath = "rules/org1.csv"


def get_samplerule(key_list, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    selectdf = plcdf[plcdf["监管要求"].isin(key_list)]
    tb_sample = selectdf[["监管要求", "结构", "条款"]]
    return tb_sample.reset_index(drop=True)


# @st.cache_data
def searchByNamesupa(search_text, industry_choice):
    table_name = industry_name_to_code(industry_choice)

    # print(table_name)
    # Get all records from table and cast 'metadata' to text type
    result = supabase.table(table_name).select("content, metadata").execute()

    # print(result.data)
    # Convert the results to a DataFrame
    df = pd.json_normalize(result.data)
    df.columns = ["条款", "结构", "监管要求"]
    # print(df)
    # Filter DataFrame based on conditions
    filtered_results = df[df["监管要求"].str.contains(f".*{search_text}.*")]

    choicels = filtered_results["监管要求"].unique().tolist()

    return filtered_results, choicels


def searchByName(search_text, industry_choice):
    rulefolder = get_rulefolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    plc_list = plcdf["监管要求"].drop_duplicates().tolist()

    choicels = []
    for plc in plc_list:
        if search_text in plc:
            choicels.append(plc)

    plcsam = get_samplerule(choicels, industry_choice)

    return plcsam, choicels


def searchByItem(searchresult, make_choice, column_text, item_text):
    # split words item_text
    item_text_list = split_words(item_text)
    column_text = fix_section_text(column_text)
    plcsam = searchresult[
        (searchresult["监管要求"].isin(make_choice))
        & (searchresult["结构"].str.contains(column_text))
        & (searchresult["条款"].str.contains(item_text_list))
    ]
    total = len(plcsam)
    return plcsam, total


# fix section text with +
def fix_section_text(section_text):
    if "+" in section_text:
        section_text = section_text.replace("+", "\\+")
    return section_text
