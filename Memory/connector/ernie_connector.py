from knowledge_bert import BertForMaskedLM, BertTokenizer
from utils import get_ent_id, get_ent_map, get_ents
from transformers import AutoTokenizer, default_data_collator, AdamW, get_scheduler, set_seed
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score


class Ernie(object):
    def __init__(self):
        self.model = BertForMaskedLM.from_pretrained("./ernie_base")
        self.true_tokenizer = BertTokenizer.from_pretrained("./ernie_base")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

        print("Loading entity embeddings ......")
        self.model.bert.set_ent_embeddings("./kg_embed/entity2vec.vec")
        
        # load kg embed information for ernie
        self.ent_map, self.ent2id = get_ent_map(), get_ent_id()

        self.max_seq_length = 128

    def preprocess_function(self, examples):
        sentence1, sentence2 = examples["masked_sentence"].split("[MASK]")
        if sentence1.strip() != "":
            tokens_a, entities_a = self.true_tokenizer.tokenize(sentence1.strip(),
                                                                get_ents(sentence1.strip(), self.ent_map))
            tokens_b, entities_b = self.true_tokenizer.tokenize(sentence2.strip(),
                                                                get_ents(sentence2.strip(), self.ent_map))
            tokens = ["[CLS]"] + tokens_a + ["[MASK]"] + tokens_b + ["[SEP]"]
            ents = ["UNK"] + entities_a + ["UNK"] + entities_b + ["UNK"]
        else:
            tokens_b, entities_b = self.true_tokenizer.tokenize(sentence2.strip(),
                                                                get_ents(sentence2.strip(), self.ent_map))
            tokens = ["[CLS]"] + ["[MASK]"] + tokens_b + ["[SEP]"]
            ents = ["UNK"] + ["UNK"] + entities_b + ["UNK"]
        segment_ids = [0] * len(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_ids = self.true_tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        input_mask[input_ids.index(103)] = 0

        input_ent, ent_mask = [], []
        for ent in ents:
            if ent != "UNK" and ent in self.ent2id:
                input_ent.append(self.ent2id[ent])
                ent_mask.append(1)
            else:
                input_ent.append(-1)
                ent_mask.append(0)
        ent_mask[0] = 1

        # Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_length - len(input_ids))
        padding_ = [-1] * (self.max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        input_ent += padding_
        ent_mask += padding

        result = {"obj_label_id": examples["obj_label_id"], "input_ids": input_ids, "attention_mask": input_mask,
                  "token_type_ids": segment_ids, "input_ents": input_ent, "ent_mask": ent_mask}

        return result

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_string

    def get_results(self, dataset, batch_size, device):
        self.model.to(device)

        dataset = dataset.map(self.preprocess_function, batch=False, num_proc=1)

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

    def train(self, dataset, train_batch_size, eval_batch_size, device, results):
        self.model.to(device)
        set_seed(2021)

        dataset = dataset.map(self.preprocess_function, batch=False, num_proc=1)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
        loss_fct = CrossEntropyLoss()
        train_dataloader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
        eval_dataloader = DataLoader(dataset=dataset, batch_size=eval_batch_size, shuffle=False)
        num_epochs = 3
        max_train_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=max_train_steps,
        )

        eval_steps = max_train_steps // 50
        complete_steps = 0

        mask_token_id = self.tokenizer.mask_token_id

        for epoch in range(num_epochs):
            print(f"Training epoch {epoch + 1}.")
            for batch in tqdm(train_dataloader):
                self.model.train()
                
                labels = batch["obj_label_id"].numpy()
                inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                          "token_type_ids": batch["token_type_ids"], "input_ents": batch["input_ents"],
                          "ent_mask": batch["ent_mask"]}
                
                mask_token_index = torch.where(inputs.input_ids == mask_token_id)

                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = torch.tensor(labels, device=device)

                token_logits = self.model(**inputs).logits
                mask_token_logits = token_logits[mask_token_index[0], mask_token_index[1], :]
                loss = loss_fct(mask_token_logits.squeeze(dim=1), labels.view(-1))
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                complete_steps += 1
                if complete_steps % eval_steps == 0:
                    eval_times = complete_steps // eval_steps
                    print(f"Evaluating times {eval_times:2d}.")
                    self.model.eval()
                    predictions, references = np.asarray([]), np.asarray([])
                    with torch.no_grad():
                        for eval_batch in tqdm(eval_dataloader):
                            labels = eval_batch["obj_label_id"].numpy()
                            inputs = {"input_ids": eval_batch["input_ids"],
                                      "attention_mask": eval_batch["attention_mask"],
                                      "token_type_ids": eval_batch["token_type_ids"],
                                      "input_ents": eval_batch["input_ents"],
                                      "ent_mask": eval_batch["ent_mask"]}
                            
                            mask_token_index = torch.where(inputs.input_ids == mask_token_id)

                            inputs = {k: v.to(device) for k, v in inputs.items()}

                            token_logits = self.model(**inputs).logits
                            mask_token_logits = token_logits[mask_token_index[0], mask_token_index[1], :]
                            top_1_tokens = torch.topk(mask_token_logits, 1, dim=-1).indices[:, 0]

                            predictions = np.append(predictions, top_1_tokens.cpu().numpy())
                            references = np.append(references, np.asarray(labels))
                    acc = accuracy_score(predictions, references)
                    results.write(f"Eval {eval_times:02d}: {acc}\n")