import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast, BertLMHeadModel, BertConfig
from math import ceil


class UniLM():

    def __init__(self, config, dataset):
        super(UniLM, self).__init__(config, dataset)

        self.version = '-' + config['version'] if 'version' in config else ''
        self.pretrained_model_path = config['pretrained_model_path'] + self.version
        self.tokenizer = BertTokenizerFast.from_pretrained(self.pretrained_model_path)
        self.configuration = BertConfig.from_pretrained(self.pretrained_model_path)
        self.model = BertLMHeadModel.from_pretrained(self.pretrained_model_path, config=self.configuration)

        self.max_source_length = config['max_source_length']
        self.max_target_length = config['max_target_length'] if config['max_target_length'] else config['max_source_length']

        if config['dataset'] in ['CNN_DM', 'WritingPrompts']:
            self.model.config.max_position_embeddings = self.max_source_length = 672
            self.position_embeddings = nn.Embedding(672, self.configuration.hidden_size)
            self.position_embeddings.weight.data[:512] = self.model.bert.embeddings.position_embeddings.weight.data
            self.model.bert.embeddings.position_embeddings = self.position_embeddings
            self.model.bert.embeddings.register_buffer("position_ids", torch.arange(672).expand((1, -1)))
        
        self.max_source_length -= self.max_target_length

        self.no_repeat_ngram_size = 3 if config['task_type'] in ['summarization', 'translation'] else None
        self.length_penalty = 2.0 if config['dataset'] == 'CNN_DM' else None
        self.truncate_pos = 'target' if config['dataset'] == 'WritingPrompts' else 'source'

        self.padding_token_idx = self.tokenizer.pad_token_id
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']
        generate_corpus = []
        for src in source_text:
            input_ids = self.tokenize_text(src, self.max_source_length)
            input_ids = torch.cat((input_ids, torch.tensor([103]).to(self.device))).unsqueeze(0)
            token_type_ids = torch.zeros_like(input_ids)
            token_type_ids = torch.cat((token_type_ids, torch.tensor([[1]]).to(self.device)), dim=-1)

            sample_outputs = self.model.generate(
                input_ids, token_type_ids=token_type_ids,
                num_beams=4, max_length=input_ids.size(1) - 1 + self.max_target_length, early_stopping=True,
                no_repeat_ngram_size=self.no_repeat_ngram_size, length_penalty=self.length_penalty
            )
            generated_text = self.tokenizer.decode(sample_outputs[0][input_ids.size(1)-1:], skip_special_tokens=True)
            generated_text = generated_text.split()
            generate_corpus.append(generated_text)
        return generate_corpus

    def tokenize_text(self, text, max_length):
        texts = ' '.join(text)
        encoding_dict = self.tokenizer(texts, max_length=max_length, truncation=True, return_tensors="pt")
        input_ids = encoding_dict['input_ids'].to(self.device)[0]
        return input_ids

    def forward(self, corpus, epoch_idx=-1, nll_test=False):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids = []
        target_ids = []
        src_length = []
        input_masks = []
        mask_weights = torch.tensor([3., 7.]).to(self.device)
        for src, tgt in zip(source_text, target_text):
            src_ids = self.tokenize_text(src, self.max_source_length)
            tgt_ids = self.tokenize_text(tgt, self.max_target_length)[1:]
            target_id = torch.cat((src_ids, tgt_ids))
            target_ids.append(target_id)

            src_mask = torch.zeros_like(src_ids)
            tgt_mask = torch.multinomial(mask_weights, tgt_ids.size(0), replacement=True)
            input_mask = torch.cat((src_mask, tgt_mask))
            input_id = torch.cat((src_ids, tgt_ids * (1-tgt_mask) + 103 * tgt_mask))
            src_length.append(src_ids.size(0))
            input_ids.append(input_id)
            input_masks.append(input_mask)

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.padding_token_idx)
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=self.padding_token_idx)
        input_masks = pad_sequence(input_masks, batch_first=True, padding_value=self.padding_token_idx)
        batch_size, max_length = input_ids.shape
        attn_masks = input_ids != self.padding_token_idx
        token_type_ids = torch.zeros_like(input_ids).to(self.device)
        sequence_attn_masks = torch.ones(max_length, max_length).to(self.device)
        sequence_attn_masks = torch.tril(sequence_attn_masks)
        sequence_attn_masks = sequence_attn_masks.unsqueeze(0).repeat(batch_size, 1, 1)
        for i, l in enumerate(src_length):
            token_type_ids[i][l:] = 1
            sequence_attn_masks[i,:l,:l] = 1

        extended_attention_mask = (1.0 - sequence_attn_masks.unsqueeze(1)) * -10000.0
        outputs = self.model(input_ids, attention_mask=attn_masks, token_type_ids=token_type_ids, extended_attention_mask=extended_attention_mask, use_cache=False)

        token_logits = outputs.logits
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), target_ids.view(-1))
        loss = loss.reshape_as(target_ids) * input_masks

        length = ((target_ids != self.padding_token_idx) * input_masks).sum(dim=1).float()
        length = length + (length == 0).long()
        loss = loss.sum(dim=1) / length
        return loss.mean()
