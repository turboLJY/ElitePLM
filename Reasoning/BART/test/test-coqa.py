import torch
import json
import csv
from fairseq.models.bart import BARTModel
from fairseq.examples.roberta import commonsense_qa  # load the Commonsense QA task
bart = BARTModel.from_pretrained('./coqa_checkpoints_large', 'checkpoint_best.pt', '../data/CommonsenseQA')
bart.eval()  # disable dropout
bart.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
with open('./test_results/test-coqa_bart_large.csv','w') as f:
    writer = csv.writer(f)
    with open('../data/CommonsenseQA/test_rand_split_no_answers.json') as h:
        for line in h:
            example = json.loads(line)
            scores = []
            for choice in example['question']['choices']:
                input = bart.encode(
                    'Q: ' + example['question']['stem'],
                    'A: ' + choice['text'],
                    no_separator=True,                    
                )
                score = bart.predict('sentence_classification_head', input, return_logits=True)
                scores.append(score)
            pred = torch.cat(scores).argmax()
            output_line = ['"'+str(example['question']['stem'])+'"',str(int(pred))]
            writer.writerow(output_line)
