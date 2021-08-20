import pandas as pd


def read_test(filename_or_path):
    total_data = {}
    src = []
    tgt = []
    labels = []
    error_path = "../data/"+filename_or_path+"_error.txt"
    truth_path = "../data/"+filename_or_path+"_correct.txt"

    with open(error_path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace(" ", "")
            src.append(" ".join(line))

    with open(truth_path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace(" ", "")
            tgt.append(" ".join(line))

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

def write_csv(filename):
    total_data = read_test(filename)
    dataframe = pd.DataFrame({"src":total_data["src"],"tgt":total_data["tgt"],"labels":total_data["labels"]})
    dataframe.to_csv("../data/test15.csv",index=False,sep="\t")

if __name__ == '__main__':
    train_filename = "test15"
    write_csv(train_filename)












