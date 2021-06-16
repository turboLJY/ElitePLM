import torch
import json
import csv
from fairseq.models.bart import BARTModel
import fairseq_hellaswag
bart = BARTModel.from_pretrained('./hellaswag_checkpoints_large', 'checkpoint_best.pt', '../data/hellaswag')
bart.eval()  # disable dropout
bart.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0

with open('./test_results/hella_bart_large.csv','w') as f:
    writer = csv.writer(f)
    with open('../data/hellaswag/hellaswag_test.json') as h:
        for line in h:
            example = json.loads(line)
            scores = []
            for choice in example['endings']:
                input = bart.encode(
                    'Q: ' + example['ctx'],
                    'A: ' + choice,
                    no_separator=True,
                    
                )
                score = bart.predict('sentence_classification_head', input, return_logits=True)
                scores.append(score)
            pred = torch.cat(scores).argmax()
            output_line = [str(int(pred))]
            writer.writerow(output_line)
