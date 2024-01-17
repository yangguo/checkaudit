import docx
import numpy as np
import pandas as pd
import streamlit as st

from checkaudit import searchauditByItem, searchauditByName
from checkrule import searchByItem, searchByName
from gptfunc import get_audit_steps
from upload import get_uploadfiles, remove_file
from utils import get_folder_list, get_section_list, savedf

# from st_excel_table import Table


# set page config
st.set_page_config(layout="wide")

auditfolder = "audit"
rulefolder = "rules"


def main():
    menulist = ["审计程序搜索", "审计模版上传"]

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
                # df = df.fillna("")
                # display the first five rows
                st.write(df.astype(str))

                # get df columns
                cols = df.columns
                # choose proc_text and audit_text column
                sec_col = st.sidebar.selectbox("选择章节字段", cols)
                plc_col = st.sidebar.selectbox("选择条款字段", cols)
                proc_col = st.sidebar.selectbox("选择测试步骤字段", cols)
                # audit_col = st.sidebar.selectbox("选择现状描述字段", cols)
                # pbc_col = st.sidebar.multiselect("选择资料字段", cols)
                # get new df
                cols = [sec_col, plc_col, proc_col]
                newdf = df[cols]
                newdf.columns = ["结构", "条款", "审计程序"]

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
                    # audit_col = st.sidebar.selectbox("选择现状描述字段", cols)
                    # pbc_col = st.sidebar.selectbox("选择资料字段", cols)
                    # get newdf
                    cols = [sec_col, plc_col, proc_col]
                    newdf = df[cols]
                    newdf.columns = ["结构", "条款", "审计程序"]
                else:
                    st.error("请检查文件是否正确")
                    newdf = None

        else:
            st.error("请上传文件")
            newdf = None

        # display newdf
        if newdf is not None:
            st.markdown("#### 审计底稿")
            # fillna
            newdf["结构"].ffill(inplace=True)
            newdf["条款"].ffill(inplace=True)
            # newdf = newdf.fillna("")
            st.table(newdf)
        else:
            st.error("请检查文件是否正确")

        # get folder list
        folderlist = get_folder_list(auditfolder)
        # choose folder
        folder = st.sidebar.selectbox("保存文件夹", folderlist)
        # get file name
        filename = st.sidebar.text_input("输入文件名")

        if st.sidebar.button("模版保存"):
            if newdf is not None and filename != "":
                newdf["监管要求"] = filename
                path = folder + "/" + filename
                savedf(newdf, path)
                st.sidebar.success("保存成功")
            else:
                st.sidebar.error("请检查文件或文件名是否正确")

        path = auditfolder + "/" + folder
        upload_list = get_uploadfiles(path)
        upload_choice = st.sidebar.multiselect("选择待删除文件:", upload_list)

        if st.sidebar.button("删除文件"):
            if upload_choice != []:
                remove_file(upload_choice, path)
                st.sidebar.success("删除成功")
            else:
                st.sidebar.error("请选择要删除的文件")

    elif choice == "审计程序搜索":
        st.subheader("审计程序搜索")

        # choose between plc or audit proc
        plc_proc = st.sidebar.radio("选择搜索类型", ("审计程序", "监管制度"))

        if plc_proc == "审计程序":
            industry_list = get_folder_list(auditfolder)

            industry_choice = st.sidebar.selectbox("选择行业:", industry_list)

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

            # match = st.sidebar.radio("搜索方式", ("关键字搜索", "模糊搜索"))

            # if match == "关键字搜索":
            item_text = st.sidebar.text_input("按条款关键字搜索")

            proc_text = st.sidebar.text_input("按审计程序关键字搜索")

            # pbc_text = st.sidebar.text_input("按审计资料关键字搜索")

            # st.sidebar.subheader("搜索范围")
            # st.sidebar.write(make_choice)

            # if not all text is empty
            # if (
            #     (column_text != "")
            #     or (item_text != "")
            #     or (proc_text != "")
            #     or (pbc_text != "")
            # ):
            plcsam, total = searchauditByItem(
                searchresult,
                make_choice,
                column_text,
                item_text,
                proc_text,
                # pbc_text,
            )
            # proc_list = plcsam["审计子程序"].tolist()

        elif plc_proc == "监管制度":
            industry_list = get_folder_list(rulefolder)

            industry_choice = st.sidebar.selectbox("选择行业:", industry_list)

            name_text = ""
            searchresult, choicels = searchByName(name_text, industry_choice)

            make_choice = st.sidebar.multiselect("选择监管制度:", choicels)

            if make_choice == []:
                make_choice = choicels
            section_list = get_section_list(searchresult, make_choice)
            column_text = st.sidebar.multiselect("选择章节:", section_list)
            if column_text == []:
                column_text = ""
            else:
                column_text = "|".join(column_text)

            item_text = st.sidebar.text_input("按条文关键字搜索")

            plcsam, total = searchByItem(
                searchresult, make_choice, column_text, item_text
            )
        # search is done
        st.sidebar.success("搜索完成")
        st.sidebar.markdown("共搜索到" + str(total) + "条结果")

        proc_list = plcsam["条款"].tolist()

        newdf = st.data_editor(plcsam, num_rows="dynamic")
        # reset index
        newdf = newdf.reset_index(drop=True)
        # st.write(newdf)

        # choose model
        model_name = st.sidebar.selectbox(
            "选择模型",
            [
                "gpt-35-turbo",
                "gpt-35-turbo-16k",
                "gpt-4",
                "gpt-4-32k",
                "gpt-4-turbo",
                "tongyi",
                "ERNIE-Bot-4",
                "ERNIE-Bot-turbo",
                "ChatGLM2-6B-32K",
                "Yi-34B-Chat",
                "gemini-pro",
            ],
        )
        genauditproc = st.sidebar.button("审计方案生成")

        if genauditproc:
            # split list into batch of 5
            batch_num = 5
            proc_list_batch = [
                proc_list[i : i + batch_num]
                for i in range(0, len(proc_list), batch_num)
            ]

            auditls = []
            # get proc and audit batch
            for j, proc_batch in enumerate(proc_list_batch):
                with st.spinner("处理中..."):
                    # range of the batch
                    start = j * batch_num + 1
                    end = start + len(proc_batch) - 1

                    st.subheader("生成结果: " + f"{start}-{end}")

                    # get audit list
                    # audit_batch = predict(proc_batch,8, top, max_length)
                    # auditls = []
                    for i, proc in enumerate(proc_batch):
                        # audit range with stride x
                        # audit_start = i * top
                        # audit_end = audit_start + top
                        # get audit list
                        # audit_list = audit_batch[audit_start:audit_end]
                        # check if proc is blank, after remove space
                        if proc.replace(" ", "") == "":
                            st.error("审计程序为空")
                            audit_steps = ""
                            # merged_audit_list = ""
                        else:
                            audit_steps = get_audit_steps(proc, model_name)
                        # auditls.append(merged_audit_list)
                        # get count number from start
                        count = str(j * batch_num + i + 1)

                        # print proc and audit list
                        st.warning("审计要求 " + count + ":")
                        st.write(proc)
                        # print audit list
                        st.info("审计方案 " + count + ": ")
                        st.write(audit_steps)

                        auditls.append(audit_steps)

            # conversion is done
            st.sidebar.success("处理完成")
            # if dfls not empty
            if auditls:
                # combine with newdf
                newdf["审计方案"] = auditls

        st.markdown("#### 审计底稿处理结果")

        st.table(newdf)

        st.download_button(
            label="下载底稿结果",
            data=newdf.to_csv().encode("utf-8-sig"),
            file_name="wpresults.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
