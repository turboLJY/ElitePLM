import tagme

tagme.GCUBE_TOKEN = '51f0abea-9dd2-4fd3-948a-96eaa3ec068e-843339462'

def get_ent_map(kg_embed_map_path="./kg_embed/entity_map.txt"):
    ent_map = {}
    with open(kg_embed_map_path) as fin:
        for line in fin:
            name, qid = line.strip().split("\t")
            ent_map[name] = qid
    return ent_map


def get_ents(text, ent_map, threshold=0.3):
    tagme.GCUBE_TOKEN = '51f0abea-9dd2-4fd3-948a-96eaa3ec068e-843339462'
    ann = tagme.annotate(text)
    ents = []
    # Keep annotations with a score higher than 0.3
    for a in ann.get_annotations(threshold):
        if a.entity_title not in ent_map:
            continue
        ents.append([ent_map[a.entity_title], a.begin, a.end, a.score])
    return ents


def get_ent_id(kg_embed_id_path="./kg_embed/entity2id.txt"):
    entity2id = {}
    with open(kg_embed_id_path) as fin:
        fin.readline()
        for line in fin:
            qid, eid = line.strip().split('\t')
            entity2id[qid] = int(eid)
    return entity2id


def get_ent_input(ents, entity2id):
    indexed_ents = []
    ent_mask = []
    for ent in ents:
        if ent != "UNK" and ent in entity2id:
            indexed_ents.append(entity2id[ent])
            ent_mask.append(1)
        else:
            indexed_ents.append(-1)
            ent_mask.append(0)
    ent_mask[0] = 1
    return indexed_ents, ent_mask


def truncate_seq_pair(tokens_a, tokens_b, ents_a, ents_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
            ents_a.pop()
        else:
            tokens_b.pop()
            ents_b.pop()
    return tokens_a, tokens_b, ents_a, ents_b


def merge_and_truncate(tokens_a, tokens_b, entities_a, entities_b, max_length):
    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        tokens_a, tokens_b, entities_a, entities_b = \
            truncate_seq_pair(tokens_a, tokens_b, entities_a, entities_b, max_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_length - 2:
            tokens_a = tokens_a[:(max_length - 2)]
            entities_a = entities_a[:(max_length - 2)]

    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    ents = ["UNK"] + entities_a + ["UNK"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        ents += entities_b + ["UNK"]
        segment_ids += [1] * (len(tokens_b) + 1)

    return tokens, ents, segment_ids
