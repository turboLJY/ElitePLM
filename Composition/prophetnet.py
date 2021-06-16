import torch
import torch.nn as nn

from transformers import ProphetNetConfig, ProphetNetTokenizer, ProphetNetForConditionalGeneration


class ProphetNet(Seq2SeqGenerator):

    def __init__(self, config, dataset):
        super(ProphetNet, self).__init__(config, dataset)

        self.max_source_length = config['max_source_length']
        self.max_target_length = config['max_target_length'] if config['max_target_length'] else config['max_source_length']

        self.version = '-' + config['version'] if 'version' in config else ''
        self.pretrained_model_path = config['pretrained_model_path'] + self.version
        self.config = ProphetNetConfig.from_pretrained(self.pretrained_model_path)
        self.tokenizer = ProphetNetTokenizer.from_pretrained(self.pretrained_model_path)
        self.model = ProphetNetForConditionalGeneration.from_pretrained(self.pretrained_model_path, config=self.config)

        self.no_repeat_ngram_size = 3 if config['task_type'] in ['translation', 'summarization'] else None
        self.length_penalty = 2.0 if config['dataset'] == 'CNN_DM' else None
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
        texts = [' '.join(t) for t in text]
        encoding_dict = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")

        input_ids = encoding_dict['input_ids'].to(self.device)
        attn_masks = encoding_dict['attention_mask'].to(self.device)

        return input_ids, attn_masks

    def forward(self, corpus, epoch_idx=-1):
        source_text = corpus['source_text']
        target_text = corpus['target_text']

        input_ids, input_att = self.tokenize_text(source_text, self.max_source_length)
        target_ids, decoder_input_att = self.tokenize_text(target_text, self.max_target_length)

        decoder_target_ids = target_ids[:, 1:].contiguous()
        ngram_decoder_target_ids = target_ids[:, 2:].contiguous()
        decoder_input_ids = target_ids[:, :-1].contiguous()
        decoder_input_att = decoder_input_att[:, :-1].contiguous()

        outputs = self.model(
            input_ids,
            attention_mask=input_att,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_input_att,
            use_cache=False
        )

        # token_logits (Torch.Tensor): shape: [batch_size, decoder_sequence_length, vocab_size]
        token_logits = outputs.logits
        loss_main_stream = self.loss(token_logits.reshape(-1, token_logits.size(-1)), decoder_target_ids.reshape(-1))
        loss_main_stream = loss_main_stream.reshape_as(decoder_target_ids)
        length = (decoder_target_ids != self.padding_token_idx).sum(dim=1).float()
        loss_main_stream = loss_main_stream.sum(dim=1) / length.float()

        # token_logits_ngram (Torch.Tensor): shape: [batch_size, ngram - 1, decoder_sequence_length, vocab_size] -> [batch_size, decoder_sequence_length - 1, vocab_size]
        token_logits_ngram = outputs.logits_ngram.squeeze(1)[:, :-1]
        loss_predict_stream = self.loss(
            token_logits_ngram.reshape(-1, token_logits_ngram.size(-1)), ngram_decoder_target_ids.reshape(-1)
        )
        loss_predict_stream = loss_predict_stream.reshape_as(ngram_decoder_target_ids)
        ngram_length = (ngram_decoder_target_ids != self.padding_token_idx).sum(dim=1).float()
        ngram_length = torch.where(ngram_length != 0., ngram_length, torch.tensor(1.).to(self.device))
        loss_predict_stream = loss_predict_stream.sum(dim=1) / ngram_length.float()

        loss = (loss_main_stream + loss_predict_stream) / 2
        return loss.mean()
