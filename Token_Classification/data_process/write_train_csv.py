import logging
from bs4 import BeautifulSoup
import torch
import codecs
from transformers import BertTokenizer
import argparse
from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd

#  去除原文本中的 space
"""def read_file(sentences):
    data = []
    for line in sentences:
        line = line.strip()
        #line = re.sub("[a-zA-ZＡ-Ｚａ-ｚ]","",line)
        line = line.replace(" ","")
        data.append(line)
    return data"""



def read_train(file_name):
    logging.info(("Reading lines from {}".format(file_name)))
    total_data = {}
    src = []
    tgt = []
    labels = []
    with codecs.open(file_name, "r", "utf-8") as file:
        data = file.read()
        soup = BeautifulSoup(data, 'html.parser')
        results = soup.find_all('sentence')
        for item in results:

            text = item.find("text").text.strip()
            mistakes = item.find_all("mistake")
            sen_truth = list(text)
            locations = []           #存储错别字位置索引
            for mistake in mistakes:
                location = mistake.find("location").text.strip()      #获取错别字位置索引
                wrong =  mistake.find("wrong").text.strip()           #获取错别字字符
                truth = mistake.find("correction").text.strip()       #获取纠正字符
                locations.append(int(location))
                if text[int(location)-1] != wrong:
                    print("The character of the given location does not equal to the real character")
                sen_truth[int(location)-1] = truth              #获取句子内容
            sen_truth = "".join(sen_truth)

            ##Add By CuiFan # 这样做的话使每个bert分词时，每个字符都能作为一个token
            text = " ".join(text.replace(" ",""))
            sen_truth = " ".join(sen_truth.replace(" ",""))
            src.append(text)
            tgt.append(sen_truth)

    for error,truth in zip(src,tgt):
        label = []
        for e,t in zip(error,truth):
            if  e==" ":
                continue
            elif e==t:
                label.append("T")
            else:
                label.append("F")
        labels.append("".join(label))
    total_data["src"] = src
    total_data["tgt"] = tgt
    total_data["labels"] = labels
    return total_data

def write_csv(filename,save_path):
    total_data = read_train(filename)

    dataframe = pd.DataFrame({"src":total_data["src"],"tgt":total_data["tgt"],"labels":total_data["labels"]})
    dataframe.to_csv(save_path,index=False,sep="\t")



def write_txt(filename,save_path):
    total_data = read_train(filename)
    with open(save_path,"w",encoding="utf-8") as f:
        for src,tgt,label in zip(total_data["src"],total_data["tgt"],total_data["labels"]):
            f.write(src+","+tgt+","+label)




if __name__ == '__main__':
    train_filename = "../data/train.sgml"
    save_path = "../data/train.csv"
    write_csv(filename=train_filename,save_path=save_path)



















