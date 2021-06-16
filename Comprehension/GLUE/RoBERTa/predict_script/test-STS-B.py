from fairseq.models.roberta import RobertaModel

roberta = RobertaModel.from_pretrained(
    'roberta_model_path',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='../fairseq-master/STS-B-bin/'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
)   
nsamples = 0
tp, tn, fp, fn = 0, 0, 0, 0
# roberta.cuda()
roberta.eval()
with open('../GlueDataset/STS-B/test.tsv') as fin:
    fin.readline()
    print ('index\tprediction')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sent1, sent2 = tokens[7], tokens[8]
        tokens = roberta.encode(sent1, sent2)
        prediction = roberta.predict('sentence_classification_head', tokens, return_logits = True)
        prediction_rank = prediction[0][0].item() * 5.0
        print(str(index) + '\t' + str(prediction_rank))