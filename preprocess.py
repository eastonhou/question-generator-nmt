import json
import os
import utils
import config
import re
from collections import defaultdict

stop_words = set(utils.read_all_lines(config.stopwords_file))

def create_vocab(filename):
    char_vocab = defaultdict(lambda: 0)
    for line in utils.read_all_lines(filename):
        for word in line.split(' '):
            for char in word:
                char_vocab[char] += 1
    char_vocab = sorted(char_vocab.items(), key=lambda x:-x[1])
    utils.write_all_lines(config.vocab_file, ['{}:{}'.format(w,n) for w,n in char_vocab])


def prepare_dataset_with_document(source, target):
    lines = []
    for line in utils.read_all_lines(source):
        sample = json.loads(line)
        documents = [doc for doc in sample['documents'] if doc['is_selected']]
        questions = [doc['title'] for doc in documents]
        para_indices = [doc['most_related_para'] for doc in documents]
        answers = [doc['paragraphs'][k] for doc, k in zip(documents, para_indices)]
        for q, a in zip(questions, answers):
            lines.append(rip_marks(a))
            lines.append(rip_marks(q))
            lines.append('<P>')
    utils.write_all_lines(target, lines)


def prepare_dataset_with_question_answers(source, target):
    lines = []
    for line in utils.read_all_lines(source):
        sample = json.loads(line)
        question = sample['question']
        for answer in sample['answers']:
            if len(answer) > len(question)*2 and len(answer) >= 20:
                lines.append(answer)
                lines.append(question)
                lines.append('<P>')
    utils.write_all_lines(target, lines)


def rip_marks(text):
    r = re.sub(r'<([A-Za-z0-9 /\"=]+)>', r'', text)
    r = re.sub(r'&[a-zA-Z]+;', r'', r)
    r = r.replace('　', '')
    r = r.replace('\t', '')
    r = r.strip()
    return r


if __name__ == '__main__':
    prepare_dataset_with_document(config.raw_train_file, config.train_file)
    prepare_dataset_with_document(config.raw_dev_file, config.dev_file)
    create_vocab(config.train_file)
