import torch
import csv
from fairseq.models.bart import BARTModel
from examples.roberta import commonsense_qa  # load the Commonsense QA task
bart = BARTModel.from_pretrained('./roc_checkpoints_large', 'checkpoint_best.pt', '../data/roc-val')
bart.eval()  # disable dropout
bart.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
with open('./test_results/roc_bart_large.csv','w') as f:
    writer = csv.writer(f)
    with open('../data/roc-val/test.csv') as h:
        lines = list(csv.reader(h, delimiter=","))
        for id_, row in enumerate(lines[1:]):
            
            scores = []
            for choice in row[5:7]:
                input = bart.encode(
                    'Q: ' + row[1]+row[2]+row[3]+row[4],
                    'A: ' + choice,
                    no_separator=True,                    
                )
                score = bart.predict('sentence_classification_head', input, return_logits=True)
                scores.append(score)
            pred = torch.cat(scores).argmax()
            output_line = ['"'+str(row[1])+'"',str(int(pred))]
            writer.writerow(output_line)