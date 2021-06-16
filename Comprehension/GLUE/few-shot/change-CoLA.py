path = './CoLA/'
pathw = './CoLA-few-shot/'

def create(maxn, step):
    for i in range(step):
        f = open(pathw + 'train-' + str(maxn) + '-' + str(i) + '.json', 'w')
        f.write('{\"data\":')
        f.write('[')
        delta = i * maxn
        for index in range(maxn):
            if (index != 0):
                f.write(',')
            f.write(train[index + delta])
        f.write(']}')
        f.close()

train = []
with open(path + 'train.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sentence, label = tokens[3], tokens[1]
        sentence = sentence.replace('\"', '\\"')
        s = '{'
        s = s + '\"sentence\":' + '\"' + sentence + '\"' + ','
        s = s + '\"label\":' + '\"' + label + '\"'
        s = s + '}'
        train.append(s)
        
create(50, 3)
create(100, 3)
create(200, 3)
create(500, 3)

f = open(pathw + 'dev.json', 'w')
with open(path + 'dev.tsv') as fin:
    fin.readline()
    f.write('{\"data\":')
    f.write('[')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sentence, label = tokens[3], tokens[1]
        sentence = sentence.replace('\"', '\\"')
        if (index != 0):
            f.write(',')
        s = '{'
        s = s + '\"sentence\":' + '\"' + sentence + '\"' + ','
        s = s + '\"label\":' + '\"' + label + '\"'
        s = s + '}'
        f.write (s)
    f.write(']}')
f.close()

f = open(pathw + 'test.json', 'w')
with open(path + 'test.tsv') as fin:
    fin.readline()
    f.write('{\"data\":')
    f.write('[')
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sentence = tokens[1]
        sentence = sentence.replace('\"', '\\"')
        if (index != 0):
            f.write(',')
        s = '{'
        s = s + '\"sentence\":' + '\"' + sentence + '\"'
        s = s + '}'
        f.write (s)
    f.write(']}')
f.close