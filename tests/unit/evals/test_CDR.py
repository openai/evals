def str2list(input_str: str):
    # input_str = "1. [Doxorubicin, acute cardiomyopathy]\n2. [Doxorubicin, chronic cardiomyopathy]\n3. [Doxorubicin, cardiac damage]"
    # 使用分割和解析的方法将输入字符串转换成所需的列表结构
    # 首先，按换行符分割字符串，得到每一项
    items = input_str.split('\n')
    result = []
    for item in items:
       # 移除序号和点号，然后找到方括号内的内容
        item_clean = item[item.index('[') + 1:item.rindex(']')]
        # 将方括号内的内容按逗号分割，并去除两边的空白，然后转换为列表
        item_list = [x.strip() for x in item_clean.split(',')]
        # 处理每个元素，确保去除了引号
        item_list = [x.strip('\'"') for x in item_list]
        # 将处理后的列表添加到结果中
        result.append(item_list)

    return result

string =  "- [Z1n.SO [Z(4), hippocampal cognitiven dysfunction]\n- [BCNUSO,( hippoc4amp),al cognitive cognitive dysfunction dysfunction]\n]2. [BCNU, cognitive dysfunction]\n3. [BCNU, learning impairment]\n4. [BCNU, short-term memory impairment]\n5. [BCNU, decreased hippocampal glutathione reductase activity]\n6. [BCNU, reduced glutathione content]\n7. [BCNU, increased serum tumor necrosis factor-alpha]\n8. [BCNU, increased hippocampal malondialdehyde content]\n9. [BCNU, increased caspase-3 activity]\n10. [BCNU, hippocampal toxicity]\n11. [ZnSO(4), hippocampal toxicity]\n12. [ZnSO(4), preservation of cognition]\n"
strlist = str2list(string)
print(type(strlist))
print(strlist)