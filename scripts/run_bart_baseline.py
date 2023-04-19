from collections import Counter
from datasets import load_metric
import string
import re

import json
from transformers import AutoTokenizer, BartForConditionalGeneration
import torch.utils.data as data_utils

from tqdm import tqdm

BATCH_SIZE = 60

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)
def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)
def get_scores(predictions, answers):
    print(f'Results for {len(predictions)} data')
    f1 = em = total = 0
    for prediction, ground_truths in zip(predictions, answers):
        total += 1
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total

    metric = load_metric("sacrebleu")
    metric.add_batch(predictions=predictions, references=answers)
    sacrebleu = metric.compute()["score"]

    print(f"F1: {f1: .2f}")
    print(f"EM: {em: .2f}")
    print(f"sacrebleu: {sacrebleu: .2f}")

with open("../data/dpr_retrieved_test_nq.json", "r") as f:
    data = json.load(f)

qstns = []
retrieved = []
for d in data:
    qstns.append(d['cur_qstn'])
    retrieved.append(d['retrieved'][0][0])
data = []
dataloader = data_utils.DataLoader(qstns, BATCH_SIZE)

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

predictions = []

for idx, batch in tqdm(enumerate(dataloader)):
    #if idx % 100 == 0:
    #print(idx)
    batch_inputs = []
    for i in range(len(batch)):
        cur_qs = batch[i]
        retr = retrieved[(idx * 4) + i]
    #for text, score in batch['retrieved'][:5]:
        batch_inputs.append(cur_qs + " [SEP] " + retr)
    inputs = tokenizer(batch_inputs, max_length=1024, padding=True, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=50, early_stopping=True)
    outputs = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    #score_outputs = []
    #bart_scores = summary_ids["sequences_scores"]

    #for oid, o in enumerate(outputs):
    #    score_outputs.append((o, float(bart_scores[oid].item()) + batch["retrieved"][oid][1]))

    #score_outputs = sorted(score_outputs, key = lambda x : x[1], reverse = True)
    predictions.extend(outputs)

#print(predictions)

with open('../data/bart_generation_test_wo_agent.json','w') as tfile:
	tfile.write('\n'.join(predictions))
        
references = [line.strip() for line in open("../data/mdd_all/dd-generation-structure/test.target", "r").readlines()]
answers = [[reference] for reference in references]

get_scores(predictions, answers)