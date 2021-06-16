import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

path = 'gpt2-500-0/'
task = path + "qnli/"

tokenizer = AutoTokenizer.from_pretrained("path_to_checkpoint")
model = AutoModelForSequenceClassification.from_pretrained("path_to_checkpoint")

model.to("cuda")
f = open('path_to_prediction', 'w')
with open('path_to_qnli_test.tsv') as fin:
    fin.readline()
    f.write('index\tprediction\n')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2 = tokens[1], tokens[2]

        token = tokenizer(sent1, sent2, return_tensors="pt").to("cuda")
        pcl = model(**token).logits
        result = torch.softmax(pcl, dim=1).tolist()[0]
        if (result[0] > result[1]): f.write(str(index) + '\t' + '0\n')
        else: f.write(str(index) + '\t' + '1\n')
        if ((index + 1) % 100 == 0):
            sys.stderr.write(str(index + 1) + '\n')
f.close()