from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'roberta_model_path',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../fairseq-master-roberta/CoLA-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)   
nsamples = 0
tp, tn, fp, fn = 0, 0, 0, 0
roberta.cuda()
roberta.eval()
with open('../GlueDataset/CoLA/test.tsv') as fin:
    fin.readline()
    print ('index\tprediction')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent = tokens[1]
        # print("detail: ", sent)
        tokens = roberta.encode(sent)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        print(str(index) + '\t' + prediction_label)