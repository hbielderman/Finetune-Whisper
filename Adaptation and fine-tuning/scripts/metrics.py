'''import evaluate

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred, tokenizer):
    try:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {
            "wer": wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": cer_metric.compute(predictions=pred_str, references=label_str),
        }
    except Exception as e:
        print("Metric computation failed:", e)
        return {}'''
from jiwer import wer, cer

def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = wer(label_str, pred_str)
    cer_score = cer(label_str, pred_str)

    return {
        "wer": wer_score,
        "cer": cer_score
    }
