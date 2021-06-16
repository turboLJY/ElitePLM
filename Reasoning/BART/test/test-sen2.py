import torch
import csv
from fairseq.models.bart import BARTModel
import fairseq_sen2
bart = BARTModel.from_pretrained('./sen2_checkpoints_large', 'checkpoint_best.pt', '../data/sen-random-large')
bart.eval()  # disable dropout
bart.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
with open('./test_results/sen2_bart_large.csv','w') as f:
    writer = csv.writer(f)
    with open('../data/sen-random-large/test.csv') as h:
        lines = list(csv.reader(h, delimiter=","))
        
        match = 0
        for id_, row in enumerate(lines[1:]):
            label = int(row[-2])
            scores = []
            question = row[0:2][1-label]
            for choice in row[2:5]:
                input = bart.encode(
                    'Q: ' + "'"+ question +"'"+ 'is against commonsense.',
                    'A: ' + choice,
                    no_separator=True,                    
                )
                score = bart.predict('sentence_classification_head', input, return_logits=True)
                scores.append(score)
            pred = torch.cat(scores).argmax()
            if int(pred)==int(label):
                match += 1
            output_line = ['"'+str(question)+'"',str(int(pred))]
            writer.writerow(output_line)
with open('test_results/sen2_bart_large.txt','w') as t:
    t.write('sen2_bart_large test accuracy:'+str(match/(id_+1)))