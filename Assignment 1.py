import os
import math
import jieba
import logging
import numpy as np
import re  # 引入正则表达式模块
from collections import Counter


def DFS_file_search(dict_name):
    # 初始化一个栈和一个列表用于存储结果
    stack = []  # 用于存储待访问的文件夹
    result_txt = []  # 用于存储找到的文件路径
    stack.append(dict_name)  # 将初始文件夹路径放入栈中
    # 不断循环直到栈为空，即所有文件夹路径都已放入栈中
    while len(stack) != 0:
        # 取出栈顶元素
        temp_name = stack.pop()
        try:
            # 获取当前文件夹下的所有文件和文件夹名字
            temp_name2 = os.listdir(temp_name)
            for eve in temp_name2:
                # 将文件夹下的子文件夹或文件路径添加到栈中
                stack.append(temp_name + "\\" + eve)
        except NotADirectoryError:
            # 如果是文件而不是文件夹，则将文件路径添加到结果列表中
            result_txt.append(temp_name)
    return result_txt

# 调用 DFS_file_search 函数，获取所有小说文件的路径列表
path_list = DFS_file_search(r"C:\Users\123\Downloads\jyxstxtqj_downcc.com")

# 初始化一个空列表，用于存储语料库中的文本
corpus = []

# 遍历每个小说文件的路径
for path in path_list:
    with open(path, "r" ,encoding="ANSI") as file:
        # 读取文件的每一行，去掉换行符和制表符，去掉前三行
        text = [line.strip("\n").replace("\u3000","").replace("\t","") for line in file][3:]
        # 将处理后的文本内容加入语料库列表中
        corpus += text

# 初始化正则表达式
regex_str = ".*?([^\u4E00-\u9FA5]).*?"  # 匹配非中文字符   
english = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;「<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 匹配英文字符/数字/符号
symbol = []  # 初始化符号列表

# 遍历语料库中的每一行文本
for j in range(len(corpus)):
    corpus[j] = re.sub(english,"",corpus[j])  # 去掉文本中的英文字符、数字、符号
    symbol += re.findall(regex_str,corpus[j])  # 超重非中文字符并添加到 symbol 列表中

count_ = Counter(symbol)  # 统计每个非中文字符出现的次数
count_symbol = count_.most_common()  # 按出现次数从高到低排序

# 初始化一个列表，用于存储出现次数小于 200 的噪声符号
noise_symbol = []
# 遍历出现次数统计结果
for eve_tuple in count_symbol:
     if eve_tuple[1] < 200:
         # 将出现次数小于 200 的噪声符号添加到列表中
         noise_symbol.append(eve_tuple[0])

# 初始化一个变量，记录替换的噪声数据点数
noise_number = 0

# 遍历语料库中的每一行文本
for line in corpus:
    # 遍历噪声符号列表
    for noise in noise_symbol:
        # 替换当前行中的噪声符号为空字符串
        line.replace(noise,"")
        # 更新替换的噪声数据点数
        noise_number += 1

print("完成的噪声数据替换点：",noise_number)
print("替换的噪声符号：")

# 遍历噪声符号列表，逐个打印出来
for i in range(len(noise_symbol)):
    print(noise_symbol[i],end = "")
    if i % 50 == 0:
        print()  # 每 50 个符号换行打印

# 将处理后的文本写入文件中
with open("预处理后的文本.txt","w",encoding="utf-8") as f:
    for line in corpus:
        if len(line) > 1:
            print(line,file=f)  # 将每一行非空文本写入文件中

# 打开预处理后的文本.txt，并将文件内容读取到名为 'corpus' 的列表中
with open("预处理后的文本.txt","r",encoding="utf-8") as f:
    corpus = [eve.strip("\n") for eve in f]

# 计算 1-gram 模型下的中文平均信息熵
token = [] # 创建一个空列表，用于存储语料库中的分词结果

# 遍历语料库中的每个段落，对其进行分词
for para in corpus:
    token += jieba.lcut(para)

token_num = len(token)  # 计算分词后的总词数
ct = Counter(token) # 统计每个词出现的频率

vocab1 = ct.most_common()  # 获取出现频率最高的 1-gram 词语
entropy_1gram = sum([-(eve[1]/token_num)*math.log((eve[1]/token_num),2) for eve in vocab1]) # 计算熵

print("1-gram:")
print("词库总词数：",token_num, "" , "不同词的个数:", len(vocab1))
print("出现频率前10的 1-gram 词语:", vocab1[:10])
print("词单位平均信息熵(1-gram):", entropy_1gram)

# 计算字单位平均信息熵
token_char = [char for word in token for char in word]
token_char_num = len(token_char)
ct_char = Counter(token_char)
vocab_char = ct_char.most_common()
entropy_char_1gram = sum([-(eve[1]/token_char_num)*math.log((eve[1]/token_char_num),2) for eve in vocab_char]) # 计算字单位熵

print("字单位平均信息熵(1-gram):", entropy_char_1gram)

# 计算 2-gram 模型下的中文平均信息熵
def combine2gram(cutword_list): # 定义一个函数，用于将分词列表转换为 2-gram 列表
    if len(cutword_list) == 1:  # 如果输入的分词长度为1，返回空列表
        return []
     # 否则，将相邻的两个词语组合成一个 2-gram,添加到结果列表中
    res = [] 
    for i in range(len(cutword_list)-1):
        res.append(cutword_list[i] + 's' + cutword_list[i+1])
    return res        
token_2gram = []
for para in corpus:
    cutword_list = jieba.lcut(para)
    token_2gram += combine2gram(cutword_list) # 调用函数 combine2gram 将分词列表转换为 2-gram 列表，并将结果添加到 token_2gram 列表中

# 计算 2-gram 的频率统计
token_2gram_num = len(token_2gram)
ct2 = Counter(token_2gram)
vocab2 = ct2.most_common()

# 计算 2-gram 相同句首的频率统计
same_1st_word = [eve.split("s")[0] for eve in token_2gram] # 取分割后的第一个部分即句首词 组成一个新列表
assert token_2gram_num == len(same_1st_word)
ct_1st = Counter(same_1st_word)
vocab_1st = dict(ct_1st.most_common()) # 将句首词及其出现次数转换为字典形式，并按照出现次数从高到低排序

entropy_2gram = 0
for eve in vocab2:
    p_xy = eve[1]/token_2gram_num
    first_word = eve[0].split("s")[0] # 获取句首词
    p_y = eve[1]/vocab_1st[first_word]
    entropy_2gram += -p_xy*math.log(p_y,2)
print("词库总词数：",token_2gram_num,"","不同词的个数：", len(vocab2))
print("出现频率前10的2-gram词语:",vocab2[:10])
print("词单位平均信息熵(2-gram):",entropy_2gram)

# 计算字单位平均信息熵
token_2gram_chars = [char for gram in token_2gram for char in gram]
token_2gram_chars_num = len(token_2gram_chars)
ct_2gram_char = Counter(token_2gram_chars)
vocab_2gram_char = ct_2gram_char.most_common()
entropy_char_2gram = sum([-(eve[1]/token_2gram_chars_num)*math.log((eve[1]/token_2gram_chars_num),2) for eve in vocab_2gram_char]) # 计算字单位熵

print("字单位平均信息熵(2-gram):", entropy_char_2gram)


#3-gram
import string


def combine3gram(cutword_list):
    if len(cutword_list) <= 2:
        return []
    res = []
    for i in range(len(cutword_list)-2):
        res.append(cutword_list[i] + cutword_list[i+1] + "s" +cutword_list[i+2] )

    return res

token_3gram = []
for para in corpus:
    cutword_list = jieba.lcut(para)
    #cutword_list = [removePunctuation(eve) for eve in cutword_list if removePunctuation(eve) != ""]
    token_3gram += combine3gram(cutword_list)

#3-gram的频率估计
token_3gram_num = len(token_3gram)
ct3 = Counter(token_3gram)
vocab3 = ct3.most_common()
#print(vocab3[:20])

#3-gram相同句首两个词语的频率估计
same_2st_word = [eve.split("s")[0] for eve in token_3gram]
assert token_3gram_num == len(same_2st_word)
ct_2st = Counter(same_2st_word)
vocab_2st = dict(ct_2st.most_common())
entropy_3gram = 0
for eve in vocab3:
    p_xyz = eve[1]/token_3gram_num
    first_2word = eve[0].split("s")[0]
    entropy_3gram += -p_xyz*math.log(eve[1]/vocab_2st[first_2word], 2)
print("词库总词数：",token_3gram_num,"","不同词的个数：", len(vocab3))
print("出现频率前10的3-gram词语:",vocab3[:10])
print("词单位平均信息熵(3-gram)",entropy_3gram)
# 计算字单位平均信息熵
token_3gram_chars = [char for gram in token_3gram for char in gram]
token_3gram_chars_num = len(token_3gram_chars)
ct_3gram_char = Counter(token_3gram_chars)
vocab_3gram_char = ct_3gram_char.most_common()
entropy_char_3gram = sum([-(eve[1]/token_3gram_chars_num)*math.log((eve[1]/token_3gram_chars_num),2) for eve in vocab_3gram_char]) # 计算字单位熵

print("字单位平均信息熵(3-gram):", entropy_char_3gram)
