import logging
from bs4 import BeautifulSoup
import torch
import codecs
from transformers import BertTokenizer
import argparse
from torch.utils.data import Dataset,DataLoader
import os


#定义一个gpu设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="微调robera，作序列标注任务"
    )

    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="训练或者测试的文件名",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="输入模型的句子的最大长度",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help= "微调的模型名称或者路径名"
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2,
        help= "train_batch_size"
    )

    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=100,
        help= "test_batch_size"
    )
    args = parser.parse_args()
    return args


#  去除原文本中的 space
def read_file(sentences):
    data = []
    for line in sentences:
        line = line.strip()
        #line = re.sub("[a-zA-ZＡ-Ｚａ-ｚ]","",line)
        line = line.replace(" ","")
        data.append(line)
    return data




def read_train(file_name):
    logging.info(("Reading lines from {}".format(file_name)))
    total_data = {}
    src = []
    tgt = []
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
            src.append(text)
            tgt.append("".join(sen_truth))
    src = read_file(src)
    tgt = read_file(tgt)

    total_data["src"] = src
    total_data["tgt"] = tgt
    return total_data


def read_test(filename_or_path):
    total_data = {}
    error = []
    truth = []
    error_path = "../data/"+filename_or_path+"_error.txt"
    print(error_path)
    truth_path = "../data/"+filename_or_path+"_correct.txt"
    with open(error_path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            # line = re.sub("[a-zA-ZＡ-Ｚａ-ｚ]","",line)
            line = line.replace(" ", "")
            error.append(line)

    with open(truth_path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            # line = re.sub("[a-zA-ZＡ-Ｚａ-ｚ]","",line)
            line = line.replace(" ", "")
            truth.append(line)
    total_data["src"] = error
    total_data["tgt"] = truth
    return total_data




class csc_dataset(Dataset):
    def __init__(self,filename_or_path,model_name_or_path,max_length,do_train):
        super(csc_dataset, self).__init__()
        self.max_length = max_length
        self.tokenlizer = BertTokenizer.from_pretrained(model_name_or_path)
        if do_train:
            self.total_data = read_train(filename_or_path)
        else:
            self.total_data = read_test(filename_or_path)


    def __getitem__(self, idx):
        src = self.total_data["src"][idx]
        tgt = self.total_data["tgt"][idx]
        tokenlizer_output = self.tokenlizer(" ".join(src),padding="max_length", truncation=True, max_length=self.max_length)
        src_id = tokenlizer_output["input_ids"]
        token_type_id = tokenlizer_output["token_type_ids"]
        attention_mask = tokenlizer_output["attention_mask"]
        tgt_id = self.tokenlizer(" ".join(tgt),padding="max_length", truncation=True, max_length=self.max_length)["input_ids"]
        label = self.get_label(src_id,tgt_id)

        return src,tgt,src_id,token_type_id,attention_mask,tgt_id,label

    def __len__(self):
        return len(self.total_data["src"])


    def get_label(self,error,truth):
        label = []
        for e, t in zip(error, truth):
            if e == t and e not in [0,101,102]:
                label.append(0)
            elif e == t and e in [0,101,102]:
                label.append(-100)
            else:
                label.append(1)

        return label


def get_dataloader(filename,model_name_or_path,max_length,do_train,batch_size):
    data_set = csc_dataset(filename,model_name_or_path,max_length,do_train)
    data_loader = DataLoader(data_set,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    return data_loader


def collate_fn(batch):
    src,tgt,src_id,token_type_id,attention_mask,tgt_id,label = list(zip(*batch))
    src_id = torch.tensor(src_id)
    token_type_id = torch.tensor(token_type_id)
    attention_mask = torch.tensor(attention_mask)
    tgt_id = torch.tensor(tgt_id)
    label = torch.tensor(label)
    return src,tgt,src_id,token_type_id,attention_mask,tgt_id,label


if __name__ == '__main__':
    filename = "../data/small_train.sgml"



    tokenizer_name_or_path = "../roberta_pretrained"
    max_length = 20

    dataset = csc_dataset(filename,tokenizer_name_or_path,max_length,True)
    print(dataset[0])
    dataloder = get_dataloader(filename,tokenizer_name_or_path,max_length,True,2)
    for batch in dataloder:
        print(batch)
