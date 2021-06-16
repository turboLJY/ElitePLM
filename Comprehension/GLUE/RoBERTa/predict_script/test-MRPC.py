from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'roberta_model_path',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../fairseq-master-roberta/MRPC-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)   
nsamples = 0
tp, tn, fp, fn = 0, 0, 0, 0
roberta.cuda()
roberta.eval()
with open('../GlueDataset/MRPC/test.tsv') as fin:
    fin.readline()
    print ('index\tprediction')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2 = tokens[3], tokens[4]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(str(index) + '\t' + prediction_label)