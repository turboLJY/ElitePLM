import torch
import csv
from fairseq.models.bart import BARTModel
import fairseq_swag
bart = BARTModel.from_pretrained('./swag_checkpoints_large', 'checkpoint_best.pt', '../data/swag')
bart.eval()  # disable dropout
bart.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
with open('./test_results/swag_bart_large.csv','w') as f:
    writer = csv.writer(f)
    with open('../data/swag/test.csv') as h:
        lines = list(csv.reader(h, delimiter=","))
        for id_, row in enumerate(lines[1:]):            
            scores = []
            for choice in row[7:11]:
                input = bart.encode(
                    'Q: ' + row[3],
                    'A: ' + choice,
                    no_separator=True,                    
                )
                score = bart.predict('sentence_classification_head', input, return_logits=True)
                scores.append(score)
            pred = torch.cat(scores).argmax()
            output_line = ['"'+str(row[3])+'"',str(int(pred))]
            writer.writerow(output_line)