#"""""
import os
import re
from nltk.util import pr
import pandas as pd
import math
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from numpy import dot
from numpy.linalg import norm
from tabulate import tabulate



def tokenization(query):
    stopWords = set(stopwords.words('english'))
    stopWords.remove('in')
    stopWords.remove('to')
    stopWords.remove('where')
    tokens = query.split()
    tokensWithoutSW = [word.lower() for word in tokens if not word in stopWords]
    return  tokensWithoutSW

def positionalIndex (tokens):
    positionalindex = dict()
    for pos, term in enumerate(tokens):
        if term in positionalindex:
            positionalindex[term].append(pos + 1)
        else:
            positionalindex[term] = []
            positionalindex[term].append(pos + 1)
    return positionalindex

def search(inquery, paths):
    print('\nThe query appear in:',end=" ")
    for i in range(len(paths)):#############################11)
        with open(paths[i]) as f:
            data = f.read()
        if inquery in data:
            print("doc%d"%(i+1),end=" ")
    print(" ")


positions = dict()

def tokenAndPositionalndex(query):
    print("\n====================== Tokens in each document ======================\n")
    for i in range(1, 11):
        with open('%d.txt' % i, 'r', encoding="utf-8") as doc:
            read_string = doc.read()
            tokens = tokenization(read_string)
            positionalindex = positionalIndex(tokens)

            print(tokens)
            for word in query:
                for pos,term in enumerate(positionalindex):
                    if term == word:
                        count = len(positionalindex[word])
                        if word in positions:
                            positions[word][0] = positions[word][0] + count
                            positions[word].append([count,"Doc" + str(i), positionalindex[word]])
                        else:
                            positions[word] = []
                            positions[word].append(count)
                            positions[word].append([count,"Doc" + str(i), positionalindex[word]])
    df = pd.DataFrame([positions])

    print("\n \n")
    print("====================== Result of query ======================\n")
    print(tabulate(df, showindex=False, headers=df.columns, tablefmt = 'psql'))


def token_and_stop():
    paths = []
    stop_words = set(stopwords.words('english'))
    stop_words.remove('in')
    stop_words.remove('to')
    stop_words.remove('where')
    matrix = pd.DataFrame()
    for i in range(1,11):#############################11
        doc_path="E:\Facutly\Level 4\sem 1\IR\project\\new Project/"+str(i)+".txt"
        with open(doc_path,"r") as f:
            data = f.read()

        data = data.lower()
        word_tokens = word_tokenize(data)
        paths.append(doc_path)
        filtered_sentence = []
        for w in word_tokens:
            x=['.',',','-']#############'s
            if (w not in stop_words) and (w not in x):
                filtered_sentence.append(w)

        matrix = pd.concat([matrix,pd.Series(filtered_sentence)], ignore_index=True, axis=1)###append lists of different size into df
        doc_path=''
        filtered_sentence = []
        #paths.append(path)
        
    matrix.columns = [1,2,3,4,5,6,7,8,9,10]#################################10

    return matrix, paths

df,li = token_and_stop()

def isNaN(num):
    return num!= num

def make_dic(df):
    dic={}
    r,c = df.shape
    for x in range(1,c+1):
        for i in range(0,r):
            if isNaN(df[x][i]) != True:
                word = df[x][i]
                if word not in dic:
                    dic[word] = [[],[],[],[],[],[],[],[],[],[]]##############10[]
                    dic[word][x-1].append(i)
                else:
                    dic[word][x-1].append(i)
    return dic

def pos(df):
    dic = make_dic(df)
    dic2 = {}
    count = 0
    print("\n====================== Positional index for each term ======================")
    for key, value in dic.items():
        for i in range(1,11):#to get Docment freq (DF) and IDF
            e = key in df[i].values
            if e:
                count +=1
        print("\n<"+key+","+str(count)+";")
        dic2[key] = []
        dic2[key].append(count)#df
        dic2[key].append(math.log((10/count),10))#to get (IDF)
        for x in range(0,10):#####################0,10
            print("doc%d:"%(x+1),end=" ")
            print(*value[x],sep=', ')#iterate over a loop of numbers with "," between them as separator
        print(">")
        count = 0
    return dic2
    
def show_idf_df(dic2):
    print("------------------------------------------------------")
    print('\t\tDF\t|\tIDF')
    print("------------------------------------------------------")
    for key,value in dic2.items():
        if len(key)>6:
            print(key+"\t"+str(value[0])+"\t|\t"+str(value[1]))
        else:
            print(key+"\t\t"+str(value[0])+"\t|\t"+str(value[1]))


def tf(df):
    dic = make_dic(df)
    tf = pd.DataFrame(columns=['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10'])
    for key, value in dic.items():
        tf.loc[key] = [len(value[0]), len(value[1]), len(value[2]), len(value[3]), len(value[4]),
         len(value[5]), len(value[6]), len(value[7]), len(value[8]), len(value[9])]
        
    print(tf)
    return tf


def w_tf(tf):
    dic_3 = make_dic(tf)
    w_tf =  pd.DataFrame(columns=['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10'])
    for key, value in dic_3.items():
        list_of_tf=[]
        for i in range(10):
            if len(value[i])==0:
                list_of_tf.append(0.0)
            else :
                list_of_tf.append(math.log(len(value[i]) ,10)+ 1.0)
          
        w_tf.loc[key]=list_of_tf
    print(w_tf)
    return w_tf

def get_length(df,dic):
    #------(w_tf*idf)**2
    for key, value in dic.items():
        df.loc[key] = (value[1]*df.loc[key])**2 

    #------sum (w_tf*idf)**2 to each doc
    Total =df[['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']].sum().tolist()

    for i in range(10):
        doc_length = float(math.sqrt(Total[i]*1.0))
        length.append(doc_length)
        if i==9:
            print("d"+str(i+1)+" length:\t %f"%(doc_length))
        else:
            print("d"+str(i+1)+"  length:\t %f"%(doc_length))
        doc_length = 0

def Normalized_tf_idf(df,dic):
    #------(w_tf*idf)    wt
    for key, value in dic.items():
        df.loc[key] = (value[1]*df.loc[key]) / length[7]


    #  for key, value in wt.items():
        #print(key)
        #print(value[0])

       # for i in range(1,11):
      #      df.loc[key] = value[0] / length[i-1]

     #   df.loc[key]= value[0]
        #df.loc[key]=[list_of_normalized[0],list_of_normalized[1],list_of_normalized[2],list_of_normalized[3],list_of_normalized[4],
        #list_of_normalized[5],list_of_normalized[6],list_of_normalized[7],list_of_normalized[8],list_of_normalized[9]]
        #list_of_normalized = [0,0,0,0,0,0,0,0,0,0]
    #print(df)
    print(df)
        

  
    

def tf_idf(df, dic):
    for key, value in dic.items():
        df.loc[key] = value[1]*df.loc[key]
    print(df)
    return df

length =[
   # 1.373462315,1.279618468,0.498974257,0.782940962,0.582747258,0.674270197,1.223495757,1.223495757,1.106137281,1.106137281
    ] 
inquery = ""

def main():
    inquery = input("\nPlease enter query: ")
    query = tokenization(inquery)
    tokenAndPositionalndex(query)
    search(inquery,li)
    df_and_idf_dic = pos(df)
    print("\n=================== Term Frequency(TF) ===================\n")
    tf_dataframe = tf(df)
    print("\n=================== weight Term Frequency(w tf(1+ log tf)) ===================\n")
    wtf_dataframe = w_tf(df)
   
    print("\n\n====================== DF and IDF ======================\n")
    show_idf_df(df_and_idf_dic)
   
    print("\n\n======================== TF*IDF ========================\n")
    wt = tf_idf(tf_dataframe,df_and_idf_dic)

    print("\n\n======================== Length of documents ========================\n")
    get_length(wtf_dataframe,df_and_idf_dic)

    print("\n\n======================== Normalized of documents ========================\n")
    Normalized_tf_idf(tf_dataframe,df_and_idf_dic)

if __name__ == "__main__":
	main()


