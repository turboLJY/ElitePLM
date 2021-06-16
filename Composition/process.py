from transformers import BartTokenizerFast
from transformers import GPT2TokenizerFast
from transformers import ProphetNetTokenizer
from transformers import T5TokenizerFast

bart = BartTokenizerFast.from_pretrained('pretrained_model/bart-base')
gpt2 = GPT2TokenizerFast.from_pretrained('pretrained_model/gpt2-base')
prophetnet = ProphetNetTokenizer.from_pretrained('pretrained_model/prophetnet')
t5 = T5TokenizerFast.from_pretrained('pretrained_model/t5-base')
tokenizer = [bart, gpt2, prophetnet, t5]

for g in ['train', 'test', 'dev']:
    src = open(g + '.source').readlines()
    tgt = open(g + '.target').readlines()
    osrc = open(g + '.source', 'w')
    otgt = open(g + '.target', 'w')
    for s, t in zip(src, tgt):
        flag = 0
        for tok in tokenizer:
            if len(tok(t)['input_ids']) > 510:
                flag = 1
                break
        if flag == 0:
            osrc.write(s)
            otgt.write(t)


