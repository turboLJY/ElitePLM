from fairseq.models.bart import BARTModel
import math

bart = BARTModel.from_pretrained(
    'bart_model_path',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../fairseq-master/STS-B-bin/'
)

label_fn = lambda label: bart.task.label_dictionary.string(
    [label + bart.task.label_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
x = []
y = []
sx, sy = 0.0, 0.0
with open('../GlueDataset/STS-B/test.tsv') as fin:
    fin.readline()
    print ('index\tprediction')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2 = tokens[7], tokens[8]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('sentence_classification_head', tokens, return_logits = True)
        prediction_rank = prediction[0][0].item() * 5.0
        print(str(index) + '\t' + str(prediction_rank))