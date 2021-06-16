from fairseq.models.bart import BARTModel
import math

bart = BARTModel.from_pretrained(
    'bart_model_path',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../fairseq-master/CoLA-bin/'
)

label_fn = lambda label: bart.task.label_dictionary.string(
    [label + bart.task.label_dictionary.nspecial]
)   
nsamples = 0
tp, tn, fp, fn = 0, 0, 0, 0
bart.cuda()
bart.eval()
with open('../GlueDataset/CoLA/test.tsv') as fin:
    fin.readline()
    print ('index\tprediction')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent = tokens[1]
        # print("detail: ", sent)
        tokens = bart.encode(sent)
        prediction = bart.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(str(index) + '\t' + prediction_label)