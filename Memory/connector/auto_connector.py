from connector import (
    albert_connector,
    bert_connector,
    roberta_connector,
    gpt2_connector,
    bart_connector,
    xlnet_connector,
    prophetnet_connector,
    unilm_connector,
    t5_connector,
)

name_or_path_to_connector = {"albert": albert_connector.Albert,
                             "bert": bert_connector.Bert,
                             "roberta": roberta_connector.Roberta,
                             "gpt2": gpt2_connector.GPT2,
                             "facebook/bart": bart_connector.Bart,
                             "xlnet": xlnet_connector.XLNet,
                             "microsoft/prophetnet": prophetnet_connector.ProphetNet,
                             "microsoft/unilm": unilm_connector.UniLM,
                             "t5": t5_connector.T5,
                             }


def auto_connector(model_name_or_path, cache_dir, config):
    """ Return the model corresponding to the model_name

    Args:
        model_name_or_path: model's name in huggingface's corpus
        cache_dir: the cache path we store and reload the model
        config: model config if needed (only for unilm now)

    """
    model_name_or_path = model_name_or_path.split("-")[0]
    return name_or_path_to_connector[model_name_or_path](model_name_or_path=model_name_or_path,
                                                         cache_dir=cache_dir,
                                                         config=config)
