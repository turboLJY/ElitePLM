from datasets import load_dataset, Dataset
from tqdm import tqdm


MASK = "[MASK]"


def parse_template(template, subject_label, object_label):
    """ Return a new masked sentence based on the template

    Args:
        template: form of new masked sentences
        subject_label: subject of the masked sentence
        object_label: object of the masked sentence

    """
    SUBJ_SYMBOL = "[X]"
    OBJ_SYMBOL = "[Y]"
    template = template.replace(SUBJ_SYMBOL, subject_label)
    template = template.replace(OBJ_SYMBOL, object_label)
    return template


def filter_samples(model_connector, all_samples):
    """ Filter bad samples in the samples

    Args:
        model_connector: model to be evaluated whose tokenizer and vocab will be used to filter the dataset
        all_samples: samples whose masked sentences have been converted to template form

    """
    new_samples = []
    samples_excluded = 0

    for sample in tqdm(all_samples):
        excluded = False

        obj_label_ids = model_connector.get_id(sample["obj_label"])
        # We only consider words that are segmented into a single token
        if len(obj_label_ids) != 1 or obj_label_ids[0] == model_connector.tokenizer.unk_token_id:
            samples_excluded += 1
            excluded = True
        else:
            sample["obj_label_id"] = obj_label_ids[0]

        if excluded:
            pass
        elif "judgments" in sample:
            # only for Google-RE
            num_no = 0
            num_yes = 0
            for x in eval(sample["judgments"]):
                if x["judgment"] == "yes":
                    num_yes += 1
                else:
                    num_no += 1
            if num_no > num_yes:
                # SKIP NEGATIVE EVIDENCE
                samples_excluded += 1
            else:
                new_samples.append(sample)
        else:
            new_samples.append(sample)

    return new_samples, samples_excluded


def filter_dataset(model_connector, dataset, max_sentence_length):
    """ Filter bad samples in the dataset

    Args:
        model_connector: model to be evaluated whose tokenizer and vocab will be used to filter the dataset
        dataset: the dataset to be filtered
        max_sentence_length: max length of the masked sentences

    """
    new_samples = []
    samples_excluded = 0
    for i in tqdm(range(dataset.num_rows)):
        sample = dataset[i]
        if "obj_label" in sample:

            excluded = False

            obj_label_ids = model_connector.get_id(sample["obj_label"])
            # We only consider words that are segmented into a single token
            if len(obj_label_ids) != 1 or obj_label_ids[0] == model_connector.tokenizer.unk_token_id:
                samples_excluded += 1
                excluded = True
            else:
                sample["obj_label_id"] = obj_label_ids[0]

            if "sub" in sample and sample["sub"] == "":
                samples_excluded += 1
                excluded = True

            masked_sentence = sample["masked_sentence"]
            if len(masked_sentence.split()) > max_sentence_length:
                samples_excluded += 1
                excluded = True
            else:
                words = masked_sentence.split(MASK)
                if "GPT2" in str(type(model_connector.model)) and words[-1].strip() != ".":
                    samples_excluded += 1
                    excluded = True
                if len(words) > 2:
                    sample["masked_sentence"] = "".join(words[:-1]) + MASK + words[-1]

            if excluded:
                pass
            else:
                new_samples.append(sample)
        else:
            samples_excluded += 1
    return new_samples, samples_excluded


def load_data(subset_name, model_connector):
    """ Load dataset for evaluating

    Args:
        subset_name: dataset name to be loaded
        model_connector: model to be evaluated whose tokenizer and vocab will be used to filter the dataset

    """
    if subset_name == "wiki":
        dataset = Dataset.load_from_disk("./wiki")
    else:
        dataset = load_dataset("lama.py", subset_name)["train"]
    origin_nums = dataset.num_rows

    has_template = True if "template" in dataset.features else False

    print("Filter examples ......")

    if has_template:
        facts = set()
        for (sub, obj, template) in tqdm(zip(dataset["sub_label"], dataset["obj_label"], dataset["template"])):
            if (sub, obj, template) not in facts:
                facts.add((sub, obj, template))
        print(f"distinct template facts: {len(facts)}")
        all_samples = []
        for sub, obj, template in list(facts):
            sample = {"sub_label": sub, "obj_label": obj}
            # substitute all sentences with a standard template
            sample["masked_sentence"] = parse_template(
                template.strip(), sample["sub_label"].strip(), MASK
            )
            all_samples.append(sample)
        all_samples, samples_excluded = filter_samples(
            model_connector=model_connector, all_samples=all_samples
        )
    else:
        dataset = dataset.filter(lambda example: "[MASK]" in example["masked_sentence"])
        all_samples, samples_excluded = filter_dataset(
            model_connector=model_connector, dataset=dataset, max_sentence_length=100
        )

    print(f"Origin nums: {origin_nums}, Final nums: {len(all_samples)}.")

    results = {"masked_sentence": [], "obj_label_id": []}
    for sample in all_samples:
        results["masked_sentence"].append(sample["masked_sentence"])
        results["obj_label_id"].append(sample["obj_label_id"])
    dataset = Dataset.from_dict(results)

    return dataset
