from knowledge_bert import BertForMaskedLM, BertTokenizer
from utils import get_ent_id, get_ent_map, get_ents
from transformers import AutoTokenizer, default_data_collator
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class Ernie(object):
    def __init__(self):
        self.model = BertForMaskedLM.from_pretrained("./ernie_base")
        self.true_tokenizer = BertTokenizer.from_pretrained("./ernie_base")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        print("Loading entity embeddings ......")
        self.model.bert.set_ent_embeddings("./kg_embed/entity2vec.vec")

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_string

    def get_results(self, dataset, batch_size, device):
        self.model.to(device)

        # load kg embed information for ernie
        ent_map, ent2id = get_ent_map(), get_ent_id()

        max_seq_length = 128

        def preprocess_function(examples):
            sentence1, sentence2 = examples["masked_sentence"].split("[MASK]")
            if sentence1.strip() != "":
                tokens_a, entities_a = self.true_tokenizer.tokenize(sentence1.strip(),
                                                                    get_ents(sentence1.strip(), ent_map))
                tokens_b, entities_b = self.true_tokenizer.tokenize(sentence2.strip(),
                                                                    get_ents(sentence2.strip(), ent_map))
                tokens = ["[CLS]"] + tokens_a + ["[MASK]"] + tokens_b + ["[SEP]"]
                ents = ["UNK"] + entities_a + ["UNK"] + entities_b + ["UNK"]
            else:
                tokens_b, entities_b = self.true_tokenizer.tokenize(sentence2.strip(),
                                                                    get_ents(sentence2.strip(), ent_map))
                tokens = ["[CLS]"] + ["[MASK]"] + tokens_b + ["[SEP]"]
                ents = ["UNK"] + ["UNK"] + entities_b + ["UNK"]
            segment_ids = [0] * len(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_ids = self.true_tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            input_mask[input_ids.index(103)] = 0

            input_ent, ent_mask = [], []
            for ent in ents:
                if ent != "UNK" and ent in ent2id:
                    input_ent.append(ent2id[ent])
                    ent_mask.append(1)
                else:
                    input_ent.append(-1)
                    ent_mask.append(0)
            ent_mask[0] = 1

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            padding_ = [-1] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            input_ent += padding_
            ent_mask += padding

            result = {"obj_label_id": examples["obj_label_id"], "input_ids": input_ids, "attention_mask": input_mask,
                      "token_type_ids": segment_ids, "input_ents": input_ent, "ent_mask": ent_mask}

            return result

        dataset = dataset.map(preprocess_function, batch=False, num_proc=1)

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
        mask_token_id = self.tokenizer.mask_token_id

        predictions, references = np.asarray([]), np.asarray([])
        with torch.no_grad():
            for batch in tqdm(dataloader):
                labels = batch["obj_label_id"].numpy()
                inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                          "token_type_ids": batch["token_type_ids"], "input_ents": batch["input_ents"],
                          "ent_mask": batch["ent_mask"]}

                inputs = {k: v.to(device) for k, v in inputs.items()}

                mask_token_index = torch.where(inputs["input_ids"] == mask_token_id)

                token_logits = self.model(**inputs)
                mask_token_logits = token_logits[mask_token_index[0], mask_token_index[1], :]
                top_1_tokens = torch.topk(mask_token_logits, 1, dim=-1).indices[:, 0]

                predictions = np.append(predictions, top_1_tokens.cpu().numpy())
                references = np.append(references, labels)
        return predictions, references