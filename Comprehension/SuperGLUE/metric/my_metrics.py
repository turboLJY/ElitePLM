from sklearn.metrics import f1_score
from metric.record_evaluation import evaluate as evaluate_record


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, f1_avg="binary"):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=f1_avg)
    return {
        "accuracy": acc,
        "f1": f1,
    }


def evaluate_multirc(ids_preds, labels):
    """
    Computes F1 score and Exact Match for MultiRC predictions.
    """
    question_map = {}
    for id_pred, label in zip(ids_preds, labels):
        question_id = "{}-{}".format(id_pred["idx"]["paragraph"], id_pred["idx"]["question"])
        pred = id_pred["prediction"]
        if question_id in question_map:
            question_map[question_id].append((pred, label))
        else:
            question_map[question_id] = [(pred, label)]
    f1s, ems = [], []
    for question, preds_labels in question_map.items():
        question_preds, question_labels = zip(*preds_labels)
        f1 = f1_score(y_true=question_labels, y_pred=question_preds, average="macro")
        f1s.append(f1)
        em = int(sum([p == l for p, l in preds_labels]) == len(preds_labels))
        ems.append(em)
    f1_m = sum(f1s) / len(f1s)
    em = sum(ems) / len(ems)
    f1_a = f1_score(y_true=labels, y_pred=[id_pred["prediction"] for id_pred in ids_preds])
    return {"exact_match": em, "f1_m": f1_m, "f1_a": f1_a}


class MyMetrics:
    def __init__(self, task_name):
        self.config_name = task_name

    def compute(self, predictions, references):
        if self.config_name == "cb":
            return acc_and_f1(predictions, references, f1_avg="macro")
        elif self.config_name == "record":
            dataset = [
                {
                    "qas": [
                        {"id": ref["idx"]["query"], "answers": [{"text": ans} for ans in ref["answers"]]}
                        for ref in references
                    ]
                }
            ]
            predictions = {pred["idx"]["query"]: pred["prediction_text"] for pred in predictions}
            return evaluate_record(dataset, predictions)[0]
        elif self.config_name == "multirc":
            return evaluate_multirc(predictions, references)
        elif self.config_name in ["copa", "rte", "wic", "wsc", "boolq"]:
            return {"accuracy": simple_accuracy(predictions, references)}
