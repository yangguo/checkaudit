from utils import (  # get_embedding,; roformer_encoder,
    get_auditfolder,
    get_csvdf,
    split_words,
)


def get_auditcol(industry_choice):
    if industry_choice in ["证券", "期货", "基金"]:
        col = ["监管要求", "结构", "条款", "审计程序"]
    elif industry_choice == "等级保护":
        col = ["监管要求", "结构", "序号", "条款", "资料"]
    elif industry_choice == "内审协会":
        col = ["监管要求", "结构", "条款", "审计子程序", "资料", "判断条件"]
    else:
        col = ["监管要求", "结构", "条款", "审计子程序"]
    return col


def get_sampleaudit(key_list, industry_choice):
    auditfolder = get_auditfolder(industry_choice)
    col = get_auditcol(industry_choice)
    plcdf = get_csvdf(auditfolder)
    selectdf = plcdf[plcdf["监管要求"].isin(key_list)]
    tb_sample = selectdf[col]
    return tb_sample.reset_index(drop=True)


def searchauditByName(search_text, industry_choice):
    rulefolder = get_auditfolder(industry_choice)
    plcdf = get_csvdf(rulefolder)
    plc_list = plcdf["监管要求"].drop_duplicates().tolist()

    choicels = []
    for plc in plc_list:
        if search_text in plc:
            choicels.append(plc)
    plcsam = get_sampleaudit(choicels, industry_choice)
    return plcsam, choicels


def searchauditByItem(searchresult, make_choice, column_text, item_text):
    # split item_text into item_list
    item_list = split_words(item_text)
    # split proc_text into proc_list
    # proc_list = split_words(proc_text)
    # split pbc_text into pbc_list
    # pbc_list = split_words(pbc_text)

    plcsam = searchresult[
        (searchresult["监管要求"].isin(make_choice))
        & (searchresult["结构"].str.contains(column_text))
        & (searchresult["条款"].str.contains(item_list))
        # & (searchresult["审计程序"].str.contains(proc_list))
        # & (searchresult["资料"].str.contains(pbc_list))
    ]
    total = len(plcsam)
    # reset index
    plcsam = plcsam.reset_index(drop=True)
    return plcsam, total
