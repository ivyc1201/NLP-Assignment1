import re
import jieba

DATA_PATH = 'C:/Users/123/Desktop/NLP/jyxstxtqj_downcc.com'


def get_single_corpus(file_path):
    """
    获取file_path文件对应的内容
    :return: file_path文件处理结果
    """
    corpus = ''
    # unuseful items filter
    r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~「」『』（）]+'
    with open('C:/Users/123/Desktop/NLP/cn_stopwords.txt', 'r', encoding='utf8') as f:
        stop_words = [word.strip('\n') for word in f.readlines()]
        f.close()
    # print(stop_words)
    with open(file_path, 'r', encoding='ANSI') as f:
        corpus = f.read()
        corpus = re.sub(r1, '', corpus)
        corpus = corpus.replace('\n', '')
        corpus = corpus.replace('\u3000', '')
        corpus = corpus.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
        f.close()
    words = list(jieba.cut(corpus))
    return [word for word in words if word not in stop_words]


if __name__ == '__main__':
    with open('C:/Users/123/Desktop/NLP/jyxstxtqj_downcc.com/inf.txt', 'r') as inf:
        txt_list = inf.readline().split(',')
        for name in txt_list:
            file_name = name + '.txt'
            file_content = get_single_corpus(DATA_PATH + '/' + file_name)
            temp = []
            count = 0
            lines = []
            for w in file_content:
                if count % 50 == 0:
                    lines.append(" ".join(temp))
                    temp = []
                    count = 0
                temp.append(w)
                count += 1
            with open('C:/Users/123/Desktop/NLP/dataset/' + 'train_' + file_name, 'w', encoding='utf8') as train:
                train.writelines(lines)

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence, PathLineSentences

DATA_PATH = 'C:/Users/123/Desktop/NLP/dataset/'

if __name__ == '__main__':
    test_name = ['郭靖', '杨过', '段誉', '令狐冲', '张无忌']
    model = Word2Vec(sentences=PathLineSentences(DATA_PATH), hs=1, min_count=10, window=5,
                     vector_size=200, sg=0, workers=16, epochs=200)
    for name in test_name:
        print(name)
        for result in model.wv.similar_by_word(name, topn=10):
            print(result[0], '{:.6f}'.format(result[1]))
        print('----------------------')
    model.save('model3.model')
    # with open('../inf.txt', 'r') as inf:
    #     txt_list = inf.readline().split(',')
    #     for idx, name in enumerate(txt_list):
    #         file_name = 'train_' + name + '.txt'
    #         model = Word2Vec(sentences=LineSentence(DATA_PATH + file_name), hs=1, min_count=10, window=5,
    #                          vector_size=200, sg=0, epochs=200)
    #         print(file_name, test_name[idx])
    #         for result in model.wv.similar_by_word(test_name[idx], topn=10):
    #             print(result[0], '{:.6f}'.format(result[1]))

from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('C:/Users/123/Desktop/NLP/name.txt', 'r', encoding='utf8') as f:
        names = f.readline().split(' ')
    model = Word2Vec.load('model3.model')
    names = [name for name in names if name in model.wv]
    name_vectors = [model.wv[name] for name in names]
    tsne = TSNE()
    embedding = tsne.fit_transform(name_vectors)
    n = 5
    label = KMeans(n).fit(embedding).labels_
    plt.title('kmeans聚类结果')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(label)):
        if label[i] == 0:
            plt.plot(embedding[i][0], embedding[i][1], 'ro', )
        if label[i] == 1:
            plt.plot(embedding[i][0], embedding[i][1], 'go', )
        if label[i] == 2:
            plt.plot(embedding[i][0], embedding[i][1], 'yo', )
        if label[i] == 3:
            plt.plot(embedding[i][0], embedding[i][1], 'co', )
        if label[i] == 4:
            plt.plot(embedding[i][0], embedding[i][1], 'bo', )
        if label[i] == 5:
            plt.plot(embedding[i][0], embedding[i][1], 'mo', )
        plt.annotate(names[i], xy=(embedding[i][0], embedding[i][1]), xytext=(embedding[i][0]+0.1, embedding[i][1]+0.1))
    plt.savefig('cluster3.png')
