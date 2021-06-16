# %%
import json
import os
import random

SHOTS = 500

data = []
sample = []
questions = {}
random.seed(20211)
with open("./MultiRC/train.jsonl", "r") as f:
    lines = f.readlines()
    for line in lines:
        j = json.loads(line)
        data.append(j)

# %%
data[0]['passage']['questions'][0]

for i in range(SHOTS):
    psg = random.choice(data)
    quest = random.choice(psg['passage']['questions'])
    if psg['idx'] in questions:
        questions[psg['idx']].append(quest)
    else:
        questions[psg['idx']] = [quest]
# %%

# questions
# %%
passage_cnt = 0
question_cnt = 0
answer_cnt = 0

for qid in questions:
    qst = []
    curidx = 0
    for q in questions[qid]:
        cur_question = q
        cur_question['idx'] = question_cnt
        question_cnt += 1
        for ans in range(len(cur_question['answers'])):
            cur_question['answers'][ans]['idx'] = answer_cnt
            answer_cnt += 1
        qst.append(cur_question)
    

        
    cur_psg = {
        'idx' : passage_cnt,
        'version' : 1.1,
        'passage' : {
            'text' : data[qid]['passage']['text'],
            'questions' : qst 
        }
    }
    passage_cnt += 1
    sample.append(cur_psg)
# %%
with open("./MultiRC/" + str(SHOTS) + "/MultiRC/train.jsonl", "w") as f:
    for s in sample:
        f.write(json.dumps(s) + '\n')
# %%

# %%
