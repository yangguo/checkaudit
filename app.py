import os

import docx
import numpy as np
import pandas as pd
import streamlit as st
from st_excel_table import Table

from checkaudit import searchauditByItem, searchauditByName, searchls2df
from utils import df2aggrid, get_folder_list, get_section_list, savedf

# set page config
st.set_page_config(layout="wide")

auditfolder = "audit"


def main():

    menulist = ["审计模版上传", "审计程序搜索"]

    choice = st.sidebar.selectbox("选择功能", menulist)

    if choice == "审计模版上传":
        st.subheader("审计模版上传")

        # upload  file
        upload_file = st.file_uploader("上传审计底稿", type=["xlsx", "docx"])
        if upload_file is not None:
            # if upload file is xlsx
            if (
                upload_file.type
                == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            ):
                # get sheet names list from excel file
                xls = pd.ExcelFile(upload_file)
                sheets = xls.sheet_names
                # choose sheet name and click button
                sheet_name = st.selectbox("选择工作表", sheets)

                # choose header row
                header_row = st.number_input(
                    "选择表头行", min_value=0, max_value=10, value=0
                )
                df = pd.read_excel(
                    upload_file, header=header_row, sheet_name=sheet_name
                )
                # filllna
                df = df.fillna("")
                # display the first five rows
                st.write(df.astype(str))

                # get df columns
                cols = df.columns
                # choose proc_text and audit_text column
                sec_col = st.sidebar.selectbox("选择章节字段", cols)
                plc_col = st.sidebar.selectbox("选择条款字段", cols)
                proc_col = st.sidebar.selectbox("选择测试步骤字段", cols)
                audit_col = st.sidebar.selectbox("选择现状描述字段", cols)
                pbc_col = st.sidebar.selectbox("选择资料字段", cols)
                # get new df
                newdf = df[[sec_col, plc_col, proc_col, audit_col, pbc_col]]
                newdf.columns = ["结构", "条款", "审计子程序", "现状描述", "资料"]

            elif (
                upload_file.type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                # read docx file
                document = docx.Document(upload_file)
                # get table data
                tablels = []
                for table in document.tables:
                    tb = []
                    for row in table.rows:
                        rl = []
                        for cell in row.cells:
                            rl.append(cell.text)
                        tb.append(rl)
                    tablels.append(tb)

                # get tablels index list
                tablels_index = list(range(len(tablels)))
                tablels_no = st.selectbox("选择表格", tablels_index)
                # choose header row
                header_row = st.number_input(
                    "选择表头行", min_value=0, max_value=10, value=0
                )

                if tablels_no is not None:
                    # get tablels data
                    table = tablels[tablels_no]
                    dataarray = np.array(table)
                    dataarray2 = dataarray[header_row:, :]
                    df = pd.DataFrame(dataarray2)

                    st.write(df.astype(str))

                    # get df columns
                    cols = df.columns
                    # choose proc_text and audit_text column
                    sec_col = st.sidebar.selectbox("选择章节字段", cols)
                    plc_col = st.sidebar.selectbox("选择条款字段", cols)
                    proc_col = st.sidebar.selectbox("选择测试步骤字段", cols)
                    audit_col = st.sidebar.selectbox("选择现状描述字段", cols)
                    pbc_col = st.sidebar.selectbox("选择资料字段", cols)
                    # get newdf
                    newdf = df[[sec_col, plc_col, proc_col, audit_col, pbc_col]]
                    newdf.columns = ["章节", "条款", "测试步骤", "现状描述", "资料"]
                else:
                    st.error("请检查文件是否正确")
                    newdf = None

        else:
            st.error("请上传文件")
            newdf = None

        # get folder list
        folderlist = get_folder_list(auditfolder)
        # choose folder
        folder = st.selectbox("保存文件夹", folderlist)
        # get file name
        filename = st.text_input("输入文件名")

        if st.button("模版保存"):
            if newdf is not None and filename != "":
                newdf["监管要求"] = filename
                path = folder + "/" + filename
                savedf(newdf, path)
                st.success("保存成功")
            else:
                st.error("请检查文件或文件名是否正确")

    elif choice == "审计程序搜索":
        st.subheader("审计程序搜索")

        industry_list = get_folder_list(auditfolder)

        industry_choice = st.sidebar.selectbox("选择行业:", industry_list)

        if industry_choice != "":
            name_text = ""
            searchresult, choicels = searchauditByName(name_text, industry_choice)
            # st.write(searchresult)
            make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

            if make_choice == []:
                make_choice = choicels
            section_list = get_section_list(searchresult, make_choice)
            column_text = st.sidebar.multiselect("选择章节:", section_list)
            if column_text == []:
                column_text = ""
            else:
                column_text = "|".join(column_text)

            match = st.sidebar.radio("搜索方式", ("关键字搜索", "模糊搜索"))

            if match == "关键字搜索":

                item_text = st.sidebar.text_input("按条款关键字搜索")

                proc_text = st.sidebar.text_input("按审计程序关键字搜索")

                pbc_text = st.sidebar.text_input("按审计资料关键字搜索")

                # st.sidebar.subheader("搜索范围")
                # st.sidebar.write(make_choice)

                # if not all text is empty
                if (
                    (column_text != "")
                    or (item_text != "")
                    or (proc_text != "")
                    or (pbc_text != "")
                ):
                    plcsam, total = searchauditByItem(
                        searchresult,
                        make_choice,
                        column_text,
                        item_text,
                        proc_text,
                        pbc_text,
                    )

                    # st.title("Streamlit-Excel-Table")
                    # # convert dataframe to dict list
                    # df_dict = plcsam.to_dict("records")
                    # data = [
                    #     {"id": "hoge", "x": 5.77, "y": 8.85, "color": "red"},
                    #     {"id": "hogedb", "x": 15.77, "y": 18.85, "color": "red"},
                    #     {"id": "hogeba", "x": 25.77, "y": 28.85, "color": "red"},
                    #     {"id": "hogeas", "x": 35.77, "y": 38.85, "color": "red"},
                    # ]
                    # data=df_dict
                    # cols=plcsam.columns.tolist()
                    # columns=[
                    #     {"name": "id", "type": "text"},
                    #     {"name": "x", "type": "numeric"},
                    #     {"name": "y", "type": "numeric"},
                    #     {"name": "color", "type": "text"},
                    # ]
                    # columns=[
                    #     {"name": col, "type": "text"} for col in cols
                    # ]

                    # retable=Table(data, columns)
                    # st.write(retable)
                    # st.write(data)

                    # st.table(plcsam)
                    grid_table = df2aggrid(plcsam)

                    sel_row = grid_table["selected_rows"]

                    st.subheader("搜索结果")

                    df_sel_row = pd.DataFrame(sel_row)
                    # csv = df_sel_row.to_csv().encode("utf-8-sig")
                    if not df_sel_row.empty:
                        st.table(df_sel_row)
                        # convert dataframe to dict list
                        df_dict = df_sel_row.to_dict("records")
                        data = df_dict
                        cols = df_sel_row.columns.tolist()
                        # columns=[
                        #     {"name": col, "type": "text"} for col in cols
                        # ]
                        newdata = []
                        for i, row in enumerate(data):
                            st.markdown("#### 测试：" + str(i + 1))
                            newrow = {}
                            for col in cols:
                                # st.write(row[col])
                                # st.write(col)
                                # st.write(f"{col}{i}")
                                newval = st.text_area(
                                    label=col, value=row[col], key=f"{col}{i}"
                                )
                                newrow[col] = newval
                            testres = st.text_area(label="测试结果", key=f"testres{i}")
                            newrow["测试结果"] = testres
                            newdata.append(newrow)

                        newdf = pd.DataFrame(newdata)
                        st.table(newdf)

                        st.download_button(
                            label="下载底稿结果",
                            data=newdf.to_csv().encode("utf-8-sig"),
                            file_name="wpresults.csv",
                            mime="text/csv",
                        )
                    # search is done
                    st.sidebar.success("搜索完成")
                    st.sidebar.write("共搜索到" + str(total) + "条结果")
                    st.sidebar.download_button(
                        label="下载结果",
                        data=plcsam.to_csv(),
                        file_name="审计程序搜索结果.csv",
                        mime="text/csv",
                    )

            elif match == "模糊搜索":
                top = st.sidebar.slider("匹配数量选择", min_value=1, max_value=10, value=3)
                search_text = st.sidebar.text_area("按审计目标搜索")

                proc_text = st.sidebar.text_area("按审计程序搜索")

                search = st.sidebar.button("搜索审计程序")

                st.sidebar.subheader("搜索范围")
                st.sidebar.write(make_choice)

                if search:
                    # search by search_text
                    if search_text != "":
                        search_list = search_text.split()
                        search_list = list(filter(None, search_list))

                        procflag = False
                        with st.spinner("正在搜索中..."):
                            dfls = searchls2df(
                                search_list,
                                column_text,
                                make_choice,
                                industry_choice,
                                top,
                                procflag,
                            )
                            for search_obj, df in zip(search_list, dfls):
                                st.warning("审计目标:" + search_obj)
                                st.table(df)
                            # search is done
                            st.sidebar.success("审计目标搜索完成")
                            # download all dfls
                            st.sidebar.download_button(
                                label="下载审计目标结果",
                                data=pd.concat(dfls).to_csv(),
                                key="text",
                                file_name="审计目标匹配结果.csv",
                                mime="text/csv",
                            )

                    # seach by proc_text
                    if proc_text != "":
                        proc_list = proc_text.split()
                        proc_list = list(filter(None, proc_list))

                        procflag = True
                        with st.spinner("正在搜索中..."):
                            dfls = searchls2df(
                                proc_list,
                                column_text,
                                make_choice,
                                industry_choice,
                                top,
                                procflag,
                            )
                            for search_obj, df in zip(proc_list, dfls):
                                st.info("审计程序：" + search_obj)
                                st.table(df)
                            # search is done
                            st.sidebar.success("审计程序搜索完成")
                            # download all dfls
                            st.sidebar.download_button(
                                label="下载审计程序结果",
                                data=pd.concat(dfls).to_csv(),
                                key="proc",
                                file_name="审计程序匹配结果.csv",
                                mime="text/csv",
                            )


if __name__ == "__main__":
    main()
