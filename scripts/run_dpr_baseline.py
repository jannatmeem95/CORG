import torch.utils.data as data_utils
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
import faiss
import json
from datasets import load_from_disk

def strip_title(title):
    if title.startswith('"'):
        title = title[1:]
    if title.endswith('"'):
        title = title[:-1]
    return title

BATCH_SIZE = 70

questions = [line.strip() for line in open("../data/mdd_all/dd-generation-structure/test.source", "r").readlines()]
gold_answers = [line.strip() for line in open("../data/mdd_all/dd-generation-structure/test.target", "r").readlines()]
gold_pids = [line.strip() for line in open("../data/mdd_all/dd-generation-structure/test.pids", "r").readlines()]
gold_docs = [line.strip() for line in open("../data/mdd_all/dd-generation-structure/test.titles", "r").readlines()]

dataloader = data_utils.DataLoader(questions, BATCH_SIZE)

index = faiss.read_index("../data/mdd_kb/knowledge_dataset-dpr-all-structure/my_knowledge_dataset_index.faiss")

dataset = load_from_disk("../data/mdd_kb/knowledge_dataset-dpr-all-structure/my_knowledge_dataset")

# q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")
# q_model = DPRQuestionEncoder.from_pretrained("sivasankalpp/dpr-multidoc2dial-structure-question-encoder")

q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

print(f'len questions: {len(questions)}')

anslist = []
r_1 = 0
r_5 = 0
r_10 = 0
r_1_p = 0
r_5_p = 0
r_10_p = 0
for idx, batch in enumerate(dataloader):
    print(f'batch idx: {idx}')
    query_tok = q_tokenizer(batch, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    q_embd = q_model(query_tok).pooler_output

    scores, pids = index.search(q_embd.detach().numpy(), 10)

    for i in range(len(batch)):
        innerjson = {}
        cur_qstn = batch[i]
        innerjson["cur_qstn"] = cur_qstn
        gold_ans = gold_answers[(idx * BATCH_SIZE) + i]
        innerjson["answer"] = gold_ans
        #print(f'batch: {idx} qstn: {i} gold idx: {(idx * BATCH_SIZE) + i}')
        retrieved = []

        for r in range(len(pids[i])):
            text = dataset[int(pids[i][r])]['text']
            retrieved.append((text, float(scores[i][r])))
        innerjson["retrieved"] = retrieved
        for r in range(len(pids[i])):
            title = dataset[int(pids[i][r])]['title']
            if strip_title(title) == gold_docs[(idx * BATCH_SIZE) + i]:
                if r == 0:
                    r_1 += 1
                if r < 5:
                    r_5 += 1
                if r < 10:
                    r_10 += 1
                break
        for r in range(len(pids[i])):
            #print(f'pids: {pids[i][r]} gold: {gold_pids[(idx * BATCH_SIZE) + i]}')
            if int(pids[i][r]) == int(gold_pids[(idx * BATCH_SIZE) + i]):
                if r == 0:
                    r_1_p += 1
                if r < 5:
                    r_5_p += 1
                if r < 10:
                    r_10_p += 1
                break
        anslist.append(innerjson)
m_r_1 = r_1 / len(questions)
m_r_5 = r_5 / len(questions)
m_r_10 = r_10 / len(questions)
m_r_1_p = r_1_p / len(questions)
m_r_5_p = r_5_p / len(questions)
m_r_10_p = r_10_p / len(questions)

with open('../data/dpr_retrieved_test_nq.json', 'w') as outfile:
    json.dump(anslist, outfile, indent = 4)

print(f'r@1: {m_r_1}')
print(f'r@5: {m_r_5}')
print(f'r@10: {m_r_10}')
print(f'p@1: {m_r_1_p}')
print(f'p@5: {m_r_5_p}')
print(f'p@10: {m_r_10_p}')


