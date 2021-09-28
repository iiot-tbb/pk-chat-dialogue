import os
import argparse

from plato.args import str2bool
from plato.args import parse_args
from plato.data.dataset import Dataset
from plato.data.field import BPETextField
from palto.data.tokenizer import Tokenizer




''''
对话脚本，时间太长，，有些忘记写到哪里了，
现在需要捋一遍头绪。
从run.py可以找出对应的infer的部分
从infer入手

输入一个context，要先把它处理成 _eou_格式
再把_eou_的句子拿到preprocess处理变成编码的形式。
再把编码后的句子加载到dataloader中，
在做infer，infer后得到的输入contact到context中

'''
def main():
    #parser = argparse.ArgumentParser()
    
    #BPETextField.add_cmdline_argument(parser)
    #Dataset.add_cmdline_argument(parser)
    
    #args = parse_args(parser)
    pass
min_utt_len=1
max_utt_len = 50
filtered = False  
max_ctx_turn=16
def numericalize(tokenizer,tokens):
        assert isinstance(tokens, list)
        if len(tokens) == 0:
            return []
        element = tokens[0]
        if isinstance(element, list):
            return [numericalize(s) for s in tokens]
        else:
            return tokenizer.convert_tokens_to_ids(tokens)

def denumericalize(tokenizer,numbers):
    assert isinstance(numbers, list)
    if len(numbers) == 0:
        return []
    element = numbers[0]
    if isinstance(element, list):
        return [denumericalize(x) for x in numbers]
    else:
        return tokenizer.decode(
            numbers, ignore_tokens=[bos_token, eos_token, pad_token])
def utt_filter_pred(self, utt):
    min_utt_len=1
    max_utt_len = 50
    filtered = False    
        return min_utt_len <= len(utt) \
            and (not filtered or len(utt) <= max_utt_len)
def build_examples_multi_turn_with_knowledge(data_utter, data_type="train"):
        pad_token = "[PAD]"
        bos_token = "[BOS]"
        eos_token = "[EOS]"
        unk_token = "[UNK]"
        max_knowledge_len =20
        max_knowledge_num = 16
        vocab_path = "model/Bert/vocab.txt"
        special_tokens = [pad_token, bos_token, eos_token, unk_token]
        tokenizer = Tokenizer(vocab_path=vocab_path,
                                   special_tokens=special_tokens,
                                   tokenizer_type="multi_knowledge")
        def vocab_size():
            return tokenizer.vocab_size

        def num_specials():
            return len(special_tokens)

        def pad_id():
            return tokenizer.convert_tokens_to_ids([self.pad_token])[0]

        def bos_id(self):
            return self.tokenizer.convert_tokens_to_ids([self.bos_token])[0]

        def eos_id(self):
            return self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

        def unk_id(self):
            return self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]

        def bot_id(self):
            return 0

        def user_id(self):
            return 1
        examples = []
        ignored = 0
        knowledge, src, tgt = data_utter.strip("\n").split("\t")
        tgt = tokenizer.tokenize(tgt)
        knowledge = [tokenizer.tokenize(k) for k in knowledge.split(" __eou__ ")]
        knowledge = [k[:max_knowledge_len]
                     for k in knowledge[-max_knowledge_num:]]
        src = [tokenizer.tokenize(s) for s in src.split(" __eou__ ")]

        if (utts_filter_pred(src) and all(map(utt_filter_pred, src)) 
                and utt_filter_pred(tgt)) or data_type == "test":
            src = [s[-max_utt_len:] for s in src[-max_ctx_turn:]]
            src = [numericalize(s) + [self.eos_id] for s in src]
            knowledge = [self.numericalize(k) + [self.eos_id] for k in knowledge]
            tgt = [self.bos_id] + self.numericalize(tgt) + [self.eos_id]
            if data_type != "test":
                tgt = tgt[:self.max_utt_len + 2]
            ex = {"src": src, "knowledge": knowledge, "tgt": tgt}
            examples.append(ex)
        else:
            ignored += 1
        print(f"Built {len(examples)} {data_type.upper()} examples ({ignored} filtered)")
        return examples    
def build_example_multi_turn_with_knowledge(req):
    examples = []
    src = [self.tokenizer.tokenize(s) for s in req["context"]]
    src = [s[-self.max_utt_len:] for s in src[-self.max_ctx_turn:]]
    src = [self.numericalize(s) + [self.eos_id] for s in src]
    knowledge = [self.tokenizer.tokenize(k) for k in req["knowledge"]]
    knowledge = [k[:self.max_knowledge_len] for k in knowledge]
    knowledge = [self.numericalize(k) + [self.eos_id] for k in knowledge]
    ex = {"src": src, "knowledge": knowledge}
    examples.append(ex)
    return examples

if __name__ == "__main__":
    import json
    with open("./data/DDE_Dialog/dial.train.Bert.jsonl",'r') as f:
        first=""
        for s in f:
            first=json.loads(s.strip())
            break
    print(build_example_multi_turn_with_knowledge(first))
