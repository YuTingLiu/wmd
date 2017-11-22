#-*- encoding:utf-8 -*-

#start
import sys, numpy as np, pickle, multiprocessing as mp
from pyemd import emd
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cosine
import fasttext
import jieba
import collections
#text1 = sys.argv[0fasttext]
#text2 = sys.argv[1]
            
def tfcount(array,tokenizer=None):
    n = len(array)
    words_set = []
    if tokenizer is None:
        array = array
    else:
        array = map(tokenizer,array)
    for words in array:
        words_set.extend(words)
    words_count = np.zeros((n,),dtype=object)
    count=0
    for text in array:
        tmp=[x for x in words_set]
        tmp.extend(text)
        words_count[count]=[int(x)-1 for x in list(collections.Counter(tmp).values())]
        count+=1
    return list(set(words_set)),words_count

def main():
    
    with open('../TextRank4ZH/textrank4zh/stopwords.txt','r') as f:
        lines = f.readlines()
    stopwords = [x.strip() for x in lines]
    
    #load model
    ##############fasttext transform dataset
    model = fasttext.load_model(r'/media/ll/ufo/毕业论文/0research/fasttext/wiki.zh.bin')
    
    text1 = "2000年以来，先后两次在渝东南召开了民族地区经济社会发展现场会，并把培养选拔少数民族干部工作作为重要内容"
    text2 = "会后，我们将根据中组部和市委的要求，在认真调研的基础上，结合重庆民族地区和干部队伍的实际，制定贯彻实施意见，采取优惠政策措施，进一步把培养选拔少数民族干部工作抓出成效"
    text3 = "为统一规范和指导全省农村低保工作，进一步提高农村低保工作的管理水平，使这项工作向科学化、规范化、制度化方向发展，当务之急是要尽快制订出台《海南省农村居民最低生活保障办法》，对工作机构设置与人员编制，保障标准、补差标准，经费保障、资金管理，审批程序和奖罚责任等作出明确规定"
    #jieba cut
    text1 = [x for x in list(jieba.cut(text1)) if x not in stopwords]
    text2 = [x for x in list(jieba.cut(text2)) if x not in stopwords]
    text3 = [x for x in list(jieba.cut(text3)) if x not in stopwords]
    token = lambda x : x.split(' ')
    wordset, wordtf = tfcount([text1,text2])
    print("feature",' '.join(wordset))
    print("tf",' ' ,wordtf)
    
    v_1, v_2 = wordtf
    v_1 = np.array(v_1)
    v_2 = np.array(v_2)
    print(v_1, v_2)
    print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))
    
    W_ = [model[w] for w in wordset]
    D_ = euclidean_distances(W_)
    print(D_)
    print("d(科学化, 机构) = {:.2f}".format(D_[1, 2]))
    print("d(科学化, 培养) = {:.2f}".format(D_[1, 3]))
    # pyemd needs double precision input
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    D_ = D_.astype(np.double)
    D_ /= D_.max()  # just for comparison purposes
    print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))

if __name__ == "__main__":
    main()
    sys.exit()




