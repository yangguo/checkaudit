import streamlit as st
import pandas as pd

from checkaudit import searchauditByName,searchauditByItem,searchls2df
from utils import get_folder_list, get_section_list

auditfolder = 'audit'


def main():

    st.subheader("审计程序搜索")
    industry_list = get_folder_list(auditfolder)

    industry_choice = st.sidebar.selectbox('选择行业:', industry_list)

    # industry_choice = '支付'
    if industry_choice != '':

        # name_text = st.sidebar.text_input('按制度名关键字搜索')
        name_text = ''

        searchresult, choicels = searchauditByName(name_text,
                                                    industry_choice)

        make_choice = st.sidebar.multiselect('选择监管制度:', choicels)

        if make_choice == []:
            make_choice = choicels

        # column_text = st.sidebar.text_input('按章节关键字搜索')
        section_list = get_section_list(searchresult, make_choice)
        column_text = st.sidebar.multiselect('选择章节:', section_list)
        if column_text == []:
            column_text = ''
        else:
            column_text = '|'.join(column_text)

        match = st.sidebar.radio('搜索方式', ('关键字搜索', '模糊搜索'))

        if match == '关键字搜索':

            item_text = st.sidebar.text_input('按条款关键字搜索')

            proc_text = st.sidebar.text_input('按审计程序关键字搜索')

            pbc_text = st.sidebar.text_input('按审计资料关键字搜索')

            st.sidebar.subheader("搜索范围")
            st.sidebar.write(make_choice)
            # st.sidebar.dataframe(choicels)

            # if not all text is empty
            if (column_text != '') or (item_text != '') or (
                    proc_text != '') or (pbc_text != ''):
                plcsam, total = searchauditByItem(searchresult,
                                                    make_choice, column_text,
                                                    item_text, proc_text,
                                                    pbc_text)

                st.table(plcsam)
                # search is done
                st.sidebar.success('搜索完成')
                st.sidebar.write('共搜索到' + str(total) + '条结果')
                st.sidebar.download_button(label='下载结果',
                                    data=plcsam.to_csv(),
                                    file_name='审计程序搜索结果.csv',
                                    mime='text/csv')

                # st.sidebar.write('总数:', total)

        elif match == '模糊搜索':
            top = st.sidebar.slider('匹配数量选择',
                                    min_value=1,
                                    max_value=10,
                                    value=3)
            search_text = st.sidebar.text_area('按审计目标搜索')

            proc_text = st.sidebar.text_area('按审计程序搜索')

            search = st.sidebar.button('搜索审计程序')

            # st.sidebar.warning("搜索范围:"+''.join(make_choice))
            st.sidebar.subheader("搜索范围")
            st.sidebar.write(make_choice)
            # st.sidebar.write(''.join(make_choice))

            if search:
                # search by search_text
                if search_text != '':
                    search_list = search_text.split()
                    search_list = list(filter(None, search_list))
                    # split by ""
                    # search_list = search_text.split()
                    # filter blank item
                    # search_list = list(
                    #     filter(lambda item: item.strip(), search_list))

                    procflag = False
                    with st.spinner('正在搜索中...'):
                        dfls = searchls2df(search_list, column_text,
                                            make_choice, industry_choice,
                                            top, procflag)
                        for search_obj, df in zip(search_list, dfls):
                            st.warning('审计目标：' + search_obj)
                            st.table(df)
                        # search is done
                        st.sidebar.success('审计目标搜索完成')                            
                        # download all dfls
                        st.sidebar.download_button(label='下载审计目标结果',
                                            data=pd.concat(dfls).to_csv(),
                                            key='text',
                                            file_name='审计目标匹配结果.csv',
                                            mime='text/csv')

                # seach by proc_text
                if proc_text != '':
                    proc_list = proc_text.split()
                    proc_list = list(filter(None, proc_list))
                    # proc_list = proc_text.split()
                    # filter blank item
                    # proc_list = list(
                    #     filter(lambda item: item.strip(), proc_list))

                    procflag = True
                    with st.spinner('正在搜索中...'):
                        dfls = searchls2df(proc_list, column_text,
                                            make_choice, industry_choice,
                                            top, procflag)
                        for search_obj, df in zip(proc_list, dfls):
                            st.info('审计程序：' + search_obj)
                            st.table(df)
                        # search is done
                        st.sidebar.success('审计程序搜索完成')
                        # download all dfls
                        st.sidebar.download_button(label='下载审计程序结果',
                                                data=pd.concat(dfls).to_csv(),
                                                key='proc',
                                                file_name='审计程序匹配结果.csv',
                                                mime='text/csv')


if __name__ == '__main__':
    main()