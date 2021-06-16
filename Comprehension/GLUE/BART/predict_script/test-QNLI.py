from fairseq.models.bart import BARTModel

bart = BARTModel.from_pretrained(
    'bart_model_path',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../fairseq-master/QNLI-bin/'
)

label_fn = lambda label: bart.task.label_dictionary.string(
    [label + bart.task.label_dictionary.nspecial]
)   
ncorrect, nsamples = 0, 0
bart.cuda()
bart.eval()
with open('../GlueDataset/QNLI/test.tsv') as fin:
    fin.readline()
    print ('index\tprediction')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2 = tokens[1], tokens[2]
        tokens = bart.encode(sent1, sent2)
        prediction = bart.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(str(index) + '\t' + prediction_label)