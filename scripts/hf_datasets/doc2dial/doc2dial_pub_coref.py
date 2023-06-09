import json
import os
from types import CodeType

import datasets
import neuralcoref
import spacy

MAX_Q_LEN = 128
DATA_DIR = "../data"

logger = datasets.logging.get_logger(__name__)

nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp, greedyness=0.5)

_CITATION = """\
@inproceedings{feng-etal-2020-doc2dial,
    title = "doc2dial: A Goal-Oriented Document-Grounded Dialogue Dataset",
    author = "Feng, Song  and Wan, Hui  and Gunasekara, Chulaka  and Patel, Siva  and Joshi, Sachindra  and Lastras, Luis",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.652",
}
"""

_DESCRIPTION = """\
Doc2dial is dataset of goal-oriented dialogues that are grounded in the associated documents. \
It includes over 4500 annotated conversations with an average of 14 turns that are grounded \
in over 450 documents from four domains. Compared to the prior document-grounded dialogue datasets \
this dataset covers a variety of dialogue scenes in information-seeking conversations.
"""

_HOMEPAGE = "http://doc2dial.github.io/multidoc2dial/"


_URL = "https://doc2dial.github.io/multidoc2dial/file/"

_URLs = {
    "default": _URL + "multidoc2dial.zip",
    "domain": _URL + "multidoc2dial_domain.zip",
}


class Doc2dial(datasets.GeneratorBasedBuilder):
    "MultiDoc2Dial v1.0"

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="dialogue_domain",
            version=VERSION,
            description="This part of the dataset covers the dialgoue domain that has questions, answers and the associated doc ids",
        ),
        datasets.BuilderConfig(
            name="document_domain",
            version=VERSION,
            description="This part of the dataset covers the document domain which details all the documents in the various domains",
        ),
        datasets.BuilderConfig(
            name="multidoc2dial",
            version=VERSION,
            description="Load MultiDoc2Dial dataset for machine reading comprehension tasks by domain",
        ),
        datasets.BuilderConfig(
            name="multidoc2dial_dmv",
            version=VERSION,
            description="Load MultiDoc2Dial dataset for machine reading comprehension tasks by domain",
        ),
        datasets.BuilderConfig(
            name="multidoc2dial_ssa",
            version=VERSION,
            description="Load MultiDoc2Dial dataset for machine reading comprehension tasks by domain",
        ),
        datasets.BuilderConfig(
            name="multidoc2dial_va",
            version=VERSION,
            description="Load MultiDoc2Dial dataset for machine reading comprehension tasks by domain",
        ),
        datasets.BuilderConfig(
            name="multidoc2dial_studentaid",
            version=VERSION,
            description="Load MultiDoc2Dial dataset for machine reading comprehension tasks by domain",
        ),
    ]

    DEFAULT_CONFIG_NAME = "multidoc2dial"

    def coref_resolve(self, utterances):
        doc = nlp(utterances)
        resolved = doc._.coref_resolved
        res_split = resolved.split('**')
        querystr = str(res_split[-1])

        return querystr

    def _info(self):

        if self.config.name == "dialogue_domain":
            features = datasets.Features(
                {
                    "dial_id": datasets.Value("string"),
                    "doc_id": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "turns": [
                        {
                            "turn_id": datasets.Value("int32"),
                            "role": datasets.Value("string"),
                            "da": datasets.Value("string"),
                            "references": [
                                {
                                    "id_sp": datasets.Value("string"),
                                    "label": datasets.Value("string"),
                                }
                            ],
                            "utterance": datasets.Value("string"),
                        }
                    ],
                }
            )

        elif "document_domain" in self.config.name:
            features = datasets.Features(
                {
                    "domain": datasets.Value("string"),
                    "doc_id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "doc_text": datasets.Value("string"),
                    "spans": [
                        {
                            "id_sp": datasets.Value("string"),
                            "tag": datasets.Value("string"),
                            "start_sp": datasets.Value("int32"),
                            "end_sp": datasets.Value("int32"),
                            "text_sp": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "parent_titles": datasets.features.Sequence(
                                {
                                    "id_sp": datasets.Value("string"),
                                    "text": datasets.Value("string"),
                                    "level": datasets.Value("string"),
                                }
                            ),
                            "id_sec": datasets.Value("string"),
                            "start_sec": datasets.Value("int32"),
                            "text_sec": datasets.Value("string"),
                            "end_sec": datasets.Value("int32"),
                        }
                    ],
                    "doc_html_ts": datasets.Value("string"),
                    "doc_html_raw": datasets.Value("string"),
                }
            )

        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "da": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                    "utterance": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        my_urls = _URLs

        # data_dir = dl_manager.download_and_extract(my_urls)
        data_dir = DATA_DIR

        if self.config.name == "dialogue_domain":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "multidoc2dial/multidoc2dial_dial_train.json"),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "multidoc2dial/multidoc2dial_dial_validation.json"),
                    },
                ),
            ]
        elif self.config.name == "document_domain":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "multidoc2dial/multidoc2dial_doc.json"),
                    },
                )
            ]
        elif "multidoc2dial_" in self.config.name:
            domain = self.config.name.split("_")[-1]
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "multidoc2dial_domain", domain, "multidoc2dial_dial_validation.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "multidoc2dial_domain", domain, "multidoc2dial_dial_train.json"
                        ),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(
                            data_dir, "multidoc2dial_domain", domain, "multidoc2dial_dial_test.json"
                        ),
                    },
                ),
            ]
        elif self.config.name == "multidoc2dial":
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "multidoc2dial/multidoc2dial_dial_validation.json"),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "multidoc2dial/multidoc2dial_dial_train.json"),
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "filepath": os.path.join(data_dir, "multidoc2dial/multidoc2dial_dial_test.json"),
                    },
                ),
            ]

    def _load_doc_data_rc(self, filepath):
        # doc_filepath = os.path.join(os.path.dirname(filepath), "multidoc2dial_doc.json")
        doc_filepath = os.path.join(DATA_DIR, "multidoc2dial/multidoc2dial_doc.json")
        with open(doc_filepath, encoding="utf-8") as f:
            data = json.load(f)["doc_data"]
        return data

    def _get_answers_rc(self, references, spans, doc_text):
        """Obtain the grounding annotation for a given dialogue turn"""
        if not references:
            return []
        start, end = -1, -1
        ls_sp = []
        for ele in references:
            id_sp = ele["id_sp"]
            start_sp, end_sp = spans[id_sp]["start_sp"], spans[id_sp]["end_sp"]
            if start == -1 or start > start_sp:
                start = start_sp
            if end < end_sp:
                end = end_sp
            ls_sp.append(doc_text[start_sp:end_sp])
        answer = {"text": doc_text[start:end], "answer_start": start}
        return [answer]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        if self.config.name == "dialogue_domain":
            logger.info("generating examples from = %s", filepath)
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for domain in data["dial_data"]:
                    for doc_id in data["dial_data"][domain]:
                        for dialogue in data["dial_data"][domain][doc_id]:

                            x = {
                                "dial_id": dialogue["dial_id"],
                                "domain": domain,
                                "doc_id": doc_id,
                                "turns": dialogue["turns"],
                            }

                            yield dialogue["dial_id"], x

        elif self.config.name == "document_domain":

            logger.info("generating examples from = %s", filepath)
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                for domain in data["doc_data"]:
                    for doc_id in data["doc_data"][domain]:

                        yield doc_id, {
                            "domain": domain,
                            "doc_id": doc_id,
                            "title": data["doc_data"][domain][doc_id]["title"],
                            "doc_text": data["doc_data"][domain][doc_id]["doc_text"],
                            "spans": [
                                {
                                    "id_sp": data["doc_data"][domain][doc_id]["spans"][i]["id_sp"],
                                    "tag": data["doc_data"][domain][doc_id]["spans"][i]["tag"],
                                    "start_sp": data["doc_data"][domain][doc_id]["spans"][i]["start_sp"],
                                    "end_sp": data["doc_data"][domain][doc_id]["spans"][i]["end_sp"],
                                    "text_sp": data["doc_data"][domain][doc_id]["spans"][i]["text_sp"],
                                    "title": data["doc_data"][domain][doc_id]["spans"][i]["title"],
                                    "parent_titles": data["doc_data"][domain][doc_id]["spans"][i]["parent_titles"],
                                    "id_sec": data["doc_data"][domain][doc_id]["spans"][i]["id_sec"],
                                    "start_sec": data["doc_data"][domain][doc_id]["spans"][i]["start_sec"],
                                    "text_sec": data["doc_data"][domain][doc_id]["spans"][i]["text_sec"],
                                    "end_sec": data["doc_data"][domain][doc_id]["spans"][i]["end_sec"],
                                }
                                for i in data["doc_data"][domain][doc_id]["spans"]
                            ],
                            "doc_html_ts": data["doc_data"][domain][doc_id]["doc_html_ts"],
                            "doc_html_raw": data["doc_data"][domain][doc_id]["doc_html_raw"],
                        }

        elif "multidoc2dial" in self.config.name:
            logger.info("generating examples from = %s", filepath)
            doc_data = self._load_doc_data_rc(filepath)
            d_doc_data = {}
            for domain, d_doc in doc_data.items():
                for doc_id, data in d_doc.items():
                    d_doc_data[doc_id] = data
            with open(filepath, encoding="utf-8") as f:
                dial_data = json.load(f)["dial_data"]
                for domain, dialogues in dial_data.items():
                    for dial in dialogues:
                        all_prev_utterances = []
                        for idx, turn in enumerate(dial["turns"]):
                            doc_id = turn["references"][0]["doc_id"]
                            doc = d_doc_data[doc_id]
                            utterance_line = turn["utterance"].replace("\n", " ").replace("\t", " ")
                            if turn["role"] == "agent":
                                continue
                            all_prev_utterances.append(utterance_line)
                            # if turn["role"] == "agent":
                            #     continue
                            if idx + 1 < len(dial["turns"]):
                                if (
                                    dial["turns"][idx + 1]["role"] == "agent"
                                    and dial["turns"][idx + 1]["da"] != "respond_no_solution"
                                ):
                                    turn_to_predict = dial["turns"][idx + 1]
                                else:
                                    continue
                            else:
                                continue
                            #question_str = utterance_line + "[SEP]" + "||".join(reversed(all_prev_utterances[:-1]))
                            utterances = ' '.join(all_prev_utterances[:-1])
                            utterances = utterances + ' ** ' + utterance_line
                            question_str = self.coref_resolve(utterances)
                            id_ = "{}_{}".format(dial["dial_id"], turn["turn_id"])
                            qa = {
                                "id": id_,
                                "title": doc_id,
                                "context": doc["doc_text"],
                                "question": question_str,
                                "da": turn["da"],
                                "answers": self._get_answers_rc(
                                    turn_to_predict["references"], doc["spans"], doc["doc_text"]
                                ),
                                "utterance": turn_to_predict["utterance"],
                                "domain": domain,
                            }
                            yield id_, qa