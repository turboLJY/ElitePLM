from transformers import BartForConditionalGeneration, AutoTokenizer, AdamW, get_scheduler, set_seed
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score


class Bart(object):
    def __init__(self, model_name_or_path, cache_dir, config=None):
        self.model = BartForConditionalGeneration.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, use_fast=True)

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(" " + string.strip())
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return indexed_string

    def get_results(self, dataset, batch_size, device):
        self.model.to(device)

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        mask_token_id = self.tokenizer.mask_token_id

        predictions, references = np.asarray([]), np.asarray([])
        with torch.no_grad():
            for batch in tqdm(dataloader):
                text_inputs, labels = batch["masked_sentence"], batch["obj_label_id"]
                text_inputs = [text.replace("[MASK]", self.tokenizer.mask_token) for text in text_inputs]

                inputs = self.tokenizer(text_inputs, padding=True, return_tensors="pt")
                mask_token_index = torch.where(inputs.input_ids == mask_token_id)

                inputs = {k: v.to(device) for k, v in inputs.items()}

                token_logits = self.model(**inputs).logits
                mask_token_logits = token_logits[mask_token_index[0], mask_token_index[1], :]
                top_1_tokens = torch.topk(mask_token_logits, 1, dim=-1).indices[:, 0]

                predictions = np.append(predictions, top_1_tokens.cpu().numpy())
                references = np.append(references, np.asarray(labels))
        return predictions, references

    def train(self, dataset, train_batch_size, eval_batch_size, device, results):
        self.model.to(device)
        set_seed(2021)

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
            print(f"Training epoch {epoch+1}.")
            for batch in tqdm(train_dataloader):
                self.model.train()
                text_inputs, labels = batch["masked_sentence"], batch["obj_label_id"]
                text_inputs = [text.replace("[MASK]", self.tokenizer.mask_token) for text in text_inputs]

                inputs = self.tokenizer(text_inputs, padding=True, return_tensors="pt")
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
                        for batch in tqdm(eval_dataloader):
                            text_inputs, labels = batch["masked_sentence"], batch["obj_label_id"]
                            text_inputs = [text.replace("[MASK]", self.tokenizer.mask_token) for text in text_inputs]

                            inputs = self.tokenizer(text_inputs, padding=True, return_tensors="pt")
                            mask_token_index = torch.where(inputs.input_ids == mask_token_id)

                            inputs = {k: v.to(device) for k, v in inputs.items()}

                            token_logits = self.model(**inputs).logits
                            mask_token_logits = token_logits[mask_token_index[0], mask_token_index[1], :]
                            top_1_tokens = torch.topk(mask_token_logits, 1, dim=-1).indices[:, 0]

                            predictions = np.append(predictions, top_1_tokens.cpu().numpy())
                            references = np.append(references, np.asarray(labels))
                    acc = accuracy_score(predictions, references)
                    results.write(f"Eval {eval_times:02d}: {acc}\n")
