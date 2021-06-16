from fairseq.models.roberta import RobertaModel
import sys

roberta = RobertaModel.from_pretrained(
    'roberta_model_path',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../fairseq-master/MNLI-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)   
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('../../gluedataset/MNLI/test_mismatched.tsv') as fin:
    fin.readline()
    print ('index\tprediction')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2 = tokens[8], tokens[9]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(str(index) + '\t' + prediction_label)