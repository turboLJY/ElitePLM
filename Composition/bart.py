import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BartTokenizerFast, BartConfig, BartForConditionalGeneration


class BART():

    def __init__(self, config, dataset):
        super(BART, self).__init__(config, dataset)

        self.max_source_length = config['max_source_length']
        self.max_target_length = config['max_target_length'] if config['max_target_length'] else config['max_source_length']

        self.version = '-' + config['version'] if 'version' in config else ''
        self.pretrained_model_path = config['pretrained_model_path'] + self.version
        self.tokenizer = BartTokenizerFast.from_pretrained(self.pretrained_model_path)
        self.configuration = BartConfig.from_pretrained(self.pretrained_model_path)

        self.no_repeat_ngram_size = 3 if config['task_type'] in ['summarization', 'translation'] else None
        self.length_penalty = 2.0 if config['dataset'] == 'CNN_DM' else None

        self.model = BartForConditionalGeneration.from_pretrained(self.pretrained_model_path, config=self.configuration)
        self.padding_token_idx = self.tokenizer.pad_token_id
        self.loss = nn.CrossEntropyLoss(ignore_index=self.padding_token_idx, reduction='none')

    def generate(self, batch_data, eval_data):
        source_text = batch_data['source_text']
        input_ids, attn_masks = self.tokenize_text(source_text, self.max_source_length)

        sample_outputs = self.model.generate(
            input_ids, attention_mask=attn_masks, num_beams=4, max_length=self.max_target_length, early_stopping=True,
            no_repeat_ngram_size=self.no_repeat_ngram_size, length_penalty=self.length_penalty
        )
        generated_text = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True)
        generate_corpus = [text.split() for text in generated_text]
        return generate_corpus

    def tokenize_text(self, text, max_length):
        texts = [(' '.join(t)).replace('[SEP]', '</s>') for t in text]
        encoding_dict = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

        input_ids = encoding_dict['input_ids'].to(self.device)
        attn_masks = encoding_dict['attention_mask'].to(self.device)

        return input_ids, attn_masks

    def compute_labelsmooth_loss(self, logits, labels, masks):
        probs = F.log_softmax(logits, dim=-1) # b * l * v
        loss = self.loss(probs.view(-1, probs.size(-1)), labels.view(-1))
        loss = loss.reshape_as(labels)
        length = masks.sum(dim=1)
        nll_loss = (loss.sum(dim=1) / length).mean()

        if smoothing is not None and smoothing > 0:
            probs = -probs.mean(dim=-1) # b * l
            probs.masked_fill_(~masks.bool(), 0.)
            smooth_loss = (probs.sum(dim=-1) / length).mean()
            return nll_loss * (1 - smoothing) + smooth_loss * smoothing
        else:
            return nll_loss

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids, attn_masks = self.tokenize_text(source_text, self.max_source_length)
        target_ids, decoder_attn_masks = self.tokenize_text(target_text, self.max_target_length)

        decoder_input_ids = target_ids[:, :-1].contiguous()
        decoder_attn_masks = decoder_attn_masks[:, :-1].contiguous()
        decoder_target_ids = target_ids[:, 1:].contiguous()

        outputs = self.model(
            input_ids,
            attention_mask=attn_masks,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attn_masks,
            use_cache=False
        )

        token_logits = outputs.logits
        loss = self.loss(token_logits.view(-1, token_logits.size(-1)), decoder_target_ids.view(-1))
        loss = loss.reshape_as(decoder_target_ids)

        length = (decoder_target_ids != self.padding_token_idx).sum(dim=1).float()
        loss = loss.sum(dim=1) / length
        return loss.mean()
