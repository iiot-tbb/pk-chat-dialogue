#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Field class
"""

from itertools import chain
import json
import numpy as np
import pickle
import time
from tqdm import tqdm

from plato.args import str2bool
from plato.data.tokenizer import Tokenizer


def max_lens(X):
    lens = [len(X)]
    while isinstance(X[0], list):
        lens.append(max(map(len, X)))
        X = [x for xs in X for x in xs]
    return lens


def list2np(X, padding=0, dtype="int64"):
    shape = max_lens(X)
    ret = np.full(shape, padding, dtype=np.int32)

    if len(shape) == 1:
        ret = np.array(X)
    elif len(shape) == 2:
        for i, x in enumerate(X):
            ret[i, :len(x)] = np.array(x)
    elif len(shape) == 3:
        for i, xs in enumerate(X):
            for j, x in enumerate(xs):
                ret[i, j, :len(x)] = np.array(x)
    return ret.astype(dtype)

class BPETextField(object):

    pad_token = "[PAD]"
    bos_token = "[BOS]"
    eos_token = "[EOS]"
    unk_token = "[UNK]"
    intro_token = "[INTRODUCTION]"
    abs_token = "[ABSTRACT]"
    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("BPETextField")
        group.add_argument("--vocab_path", type=str, required=True,
                           help="The vocabulary file path.")
        group.add_argument("--filtered", type=str2bool, default=False,
                           help="Whether to filter the data with too long utterance/context. "
                           "If the data is unfiltered, it will be truncated.")
        group.add_argument("--max_len", type=int, default=230,
                           help="The maximum length of context or knowledges.")
        group.add_argument("--min_utt_len", type=int, default=1,
                           help="The minimum length of utterance.")
        group.add_argument("--max_utt_len", type=int, default=50,
                           help="The maximum length of utterance.")
        group.add_argument("--min_ctx_turn", type=int, default=1,
                           help="The minimum turn of context.")
        group.add_argument("--max_ctx_turn", type=int, default=16,
                           help="The maximum turn of context.")
        group.add_argument("--max_knowledge_num", type=int, default=16,
                           help="The maximum number of knowledges.")
        group.add_argument("--max_knowledge_len", type=int, default=30,
                           help="The maximum length of each knowledges.")
        group.add_argument("--tokenizer_type", type=str, default="Bert",
                           choices=["Bert", "GPT2"],
                           help="The type of tokenizer.")
        return group

    def __init__(self, hparams):
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token, self.intro_token, self.abs_token]
        self.tokenizer = Tokenizer(vocab_path=hparams.vocab_path,
                                   special_tokens=special_tokens,
                                   tokenizer_type=hparams.tokenizer_type)

        self.filtered = hparams.filtered
        self.max_len = hparams.max_len
        self.min_utt_len = hparams.min_utt_len
        self.max_utt_len = hparams.max_utt_len
        self.min_ctx_turn = hparams.min_ctx_turn
        self.max_ctx_turn = hparams.max_ctx_turn - 1 # subtract reply turn
        self.max_knowledge_num = hparams.max_knowledge_num
        self.max_knowledge_len = hparams.max_knowledge_len
        return

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def num_specials(self):
        return len(self.special_tokens)

    @property
    def pad_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.pad_token])[0]

    @property
    def bos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.bos_token])[0]

    @property
    def eos_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.eos_token])[0]

    @property
    def intro_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.intro_token])[0]

    @property
    def unk_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.unk_token])[0]
    
    @property
    def abs_id(self):
        return self.tokenizer.convert_tokens_to_ids([self.abs_token])[0]
    
    @property
    def bot_id(self):
        return 0

    @property
    def user_id(self):
        return 1

    @property
    def knowledge_id(self):
        return 2

    def numericalize(self, tokens):#序列化
        assert isinstance(tokens, list)
        if len(tokens) == 0:
            return []
        element = tokens[0]
        if isinstance(element, list):
            return [self.numericalize(s) for s in tokens]
        else:
            return self.tokenizer.convert_tokens_to_ids(tokens)

    def denumericalize(self, numbers): #反序列化
        assert isinstance(numbers, list)
        if len(numbers) == 0:
            return []
        element = numbers[0]
        if isinstance(element, list):
            return [self.denumericalize(x) for x in numbers]
        else:
            return self.tokenizer.decode(
                numbers, ignore_tokens=[self.bos_token, self.eos_token, self.pad_token])

    def save_examples(self, examples, filename):
        print(f"Saving examples to '{filename}' ...")
        start = time.time()
        if filename.endswith("pkl"):
            with open(filename, "wb") as fp:
                pickle.dump(examples, fp)
        elif filename.endswith("jsonl"):
            with open(filename, "w", encoding="utf-8") as fp:
                for ex in examples:
                    fp.write(json.dumps(ex) + "\n")
        else:
            raise ValueError(f"Unsport file format: {filename}")
        elapsed = time.time() - start
        print(f"Saved {len(examples)} examples (elapsed {elapsed:.2f}s)")

    def load_examples(self, filename):
        print(f"Loading examples from '{filename}' ...")
        start = time.time()
        if filename.endswith("pkl"):
            with open(filename, "rb") as fp:
                examples = pickle.load(fp)
        else:
            with open(filename, "r", encoding="utf-8") as fp:
                examples = list(map(lambda s: json.loads(s.strip()), fp))
        elapsed = time.time() - start
        print(f"Loaded {len(examples)} examples (elapsed {elapsed:.2f}s)")
        return examples

    def utt_filter_pred(self, utt):
        return self.min_utt_len <= len(utt) \
            and (not self.filtered or len(utt) <= self.max_utt_len)

    def utts_filter_pred(self, utts):
        return self.min_ctx_turn <= len(utts) \
            and (not self.filtered or len(utts) <= self.max_ctx_turn)

    def build_example_multi_turn(self, req):
        examples = []
        src = [self.tokenizer.tokenize(s) for s in req["context"]]
        src = [s[-self.max_utt_len:] for s in src[-self.max_ctx_turn:]]
        src = [self.numericalize(s) + [self.eos_id] for s in src]
        ex = {"src": src}
        examples.append(ex)
        return examples

    def build_example_multi_turn_with_knowledge(self, req):
        examples = []
        src = [self.tokenizer.tokenize(s) for s in req["context"]]
        src = [s[-self.max_utt_len:] for s in src[-self.max_ctx_turn:]]
        src = [self.numericalize(s) + [self.eos_id] for s in src]
        knowledge = [self.tokenizer.tokenize(k) for k in req["knowledge"]]
        knowledge = [k[:self.max_knowledge_len] for k in knowledge] ##这块可能需要修改
        knowledge = [self.numericalize(k) + [self.eos_id] for k in knowledge]
        ex = {"src": src, "knowledge": knowledge}
        examples.append(ex)
        return examples

    def build_examples_multi_turn(self, data_file, data_type="train"):
        print(f"Reading examples from '{data_file}' ...")
        examples = []
        ignored = 0

        with open(data_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=None):
                src, tgt = line.strip("\n").split("\t")
                tgt = self.tokenizer.tokenize(tgt)
                src = [self.tokenizer.tokenize(s) for s in src.split(" __eou__ ")]

                if (self.utts_filter_pred(src) and all(map(self.utt_filter_pred, src))
                        and self.utt_filter_pred(tgt)) or data_type == "test":
                    src = [s[-self.max_utt_len:] for s in src[-self.max_ctx_turn:]]
                    src = [self.numericalize(s) + [self.eos_id] for s in src]
                    tgt = [self.bos_id] + self.numericalize(tgt) + [self.eos_id]
                    if data_type != "test":
                        tgt = tgt[:self.max_utt_len + 2]
                    ex = {"src": src, "tgt": tgt}
                    examples.append(ex)
                else:
                    ignored += 1
        print(f"Built {len(examples)} {data_type.upper()} examples ({ignored} filtered)")
        return examples

    def build_examples_multi_turn_with_knowledge(self, data_file, data_type="train"):
        '''
        分别对src，knwoledge和tgt处理。这块只保证了每一条knowledge ，context，tgt分别满足单条的最大长度，
        因此合起来可能大于max_len,所以需要看后面的collate_fn是否对过长的文本进行了截断。
        '''
        print(f"Reading examples from '{data_file}' ...")
        examples = []
        ignored = 0

        with open(data_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=None):
                knowledge, src, tgt = line.strip("\n").split("\t")
                tgt = self.tokenizer.tokenize(tgt)
                knowledge = [self.tokenizer.tokenize(k) for k in knowledge.split(" __eou__ ")]
                knowledge = [k[:self.max_knowledge_len]
                             for k in knowledge[-self.max_knowledge_num:]]
                src = [self.tokenizer.tokenize(s) for s in src.split(" __eou__ ")]

                if (self.utts_filter_pred(src) and all(map(self.utt_filter_pred, src)) 
                        and self.utt_filter_pred(tgt)) or data_type == "test":
                    src = [s[-self.max_utt_len:] for s in src[-self.max_ctx_turn:]]
                    src = [self.numericalize(s) + [self.eos_id] for s in src]
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

    def build_examples_multi_turn_with_knowledgei_topic_turn(self, data_file, data_type="train"):
        '''
        在这里我们对原模型的数据输入进行修改，加入了判断是否需要进行topic转换的数据，分为正样本和负样本，
        正样本代表需要进行话题切换的样本，这样的样本从其它的样本中获得。
        而对于负样本，则需要用当前的数据进行构建。
        
        '''
        print(f"Reading examples from '{data_file}' ...")
        examples = []
        ignored = 0

        with open(data_file, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=None):
                knowledge, src, tgt, negative, postive = line.strip("\n").split("\t")
                tgt = self.tokenizer.tokenize(tgt)
                postive = self.tokenizer.tokenize(postive)
                negative = self.tokenizer.tokenize(negative)
                knowledge = [self.tokenizer.tokenize(k) for k in knowledge.split(" __eou__ ")]
                knowledge = [k[:self.max_knowledge_len]
                             for k in knowledge[-self.max_knowledge_num:]]
                
                            
                src = [self.tokenizer.tokenize(s) for s in src.split(" __eou__ ")]
                if (self.utts_filter_pred(src) and all(map(self.utt_filter_pred, src)) 
                        and self.utt_filter_pred(tgt)) or data_type == "test":
                    src = [s[-self.max_utt_len:] for s in src[-self.max_ctx_turn:]]
                    src = [self.numericalize(s) + [self.eos_id] for s in src]
                    knowledge = [self.numericalize(k) + [self.eos_id] for k in knowledge]
                    tgt = [self.bos_id] + self.numericalize(tgt) + [self.eos_id]
                    negative = [self.bos_id] + self.numericalize(negative) + [self.eos_id] 
                    postive = [self.bos_id] + self.numericalize(postive) + [self.eos_id]
                    negative = negative[:self.max_utt_len+2]
                    postive = negative[:self.max_utt_len+2]
                    if data_type != "test":
                        tgt = tgt[:self.max_utt_len + 2]

                    ex = {"src": src, "knowledge": knowledge, "tgt": tgt,"postive":postive,"negative":negative}
                    examples.append(ex)
                else:
                    ignored += 1
        print(f"Built {len(examples)} {data_type.upper()} examples ({ignored} filtered)")
        return examples
    
    def collate_fn_multi_turn(self, samples):
        batch_size = len(samples)

        src = [sp["src"] for sp in samples]

        src_token, src_pos, src_turn, src_role = [], [], [], []
        for utts in src:
            utt_lens = [len(utt) for utt in utts]

            # Token ids
            src_token.append(list(chain(*utts))[-self.max_len:])

            # Position ids
            pos = [list(range(l)) for l in utt_lens]
            src_pos.append(list(chain(*pos))[-self.max_len:])

            # Turn ids
            turn = [[len(utts) - i] * l for i, l in enumerate(utt_lens)]
            src_turn.append(list(chain(*turn))[-self.max_len:])

            # Role ids
            role = [[self.bot_id if (len(utts) - i) % 2 == 0 else self.user_id] * l
                    for i, l in enumerate(utt_lens)]
            src_role.append(list(chain(*role))[-self.max_len:])

        src_token = list2np(src_token, padding=self.pad_id)
        src_pos = list2np(src_pos, padding=self.pad_id)
        src_turn = list2np(src_turn, padding=self.pad_id)
        src_role = list2np(src_role, padding=self.pad_id)

        batch = {}
        batch["src_token"] = src_token
        batch["src_mask"] = (src_token != self.pad_id).astype("int64")#mask位的id为0，正常话语的id是1
        batch["src_pos"] = src_pos
        batch["src_type"] = src_role
        batch["src_turn"] = src_turn

        if "tgt" in samples[0]:
            tgt = [sp["tgt"] for sp in samples]

            # Token ids & Label ids
            tgt_token = list2np(tgt, padding=self.pad_id)

            # Position ids
            tgt_pos = np.zeros_like(tgt_token)
            tgt_pos[:] = np.arange(tgt_token.shape[1], dtype=tgt_token.dtype)

            # Turn ids
            tgt_turn = np.zeros_like(tgt_token)

            # Role ids
            tgt_role = np.full_like(tgt_token, self.bot_id)

            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
            batch["tgt_pos"] = tgt_pos
            batch["tgt_type"] = tgt_role
            batch["tgt_turn"] = tgt_turn
            batch["k_max_len"] = 0 #和有知识部分进行统一
        return batch, batch_size

    def collate_fn_multi_turn_with_knowledge(self, samples):
        batch_size = len(samples)

        src = [sp["src"] for sp in samples]
        knowledge = [sp["knowledge"] for sp in samples]
        with_topic_transfer = False
        if "postive" in samples[0]:with_topic_transfer =True
        src_token, src_pos, src_turn, src_role = [], [], [], []
        knowledge_token,knowledge_pos,knowledge_turn,knowledge_role = [],[],[],[] #新加knwoledge的部分
        for utts, ks in zip(src, knowledge):
            utt_lens = [len(utt) for utt in utts]
            k_lens = [len(k) for k in ks]
            k_max_len = max(k_lens+[self.max_len])
            # Token ids
            token = list(chain(*utts))[-self.max_len:]
            token.extend(list(chain(*ks))[-self.max_len:])
            src_token.append(token)

            # Position ids
            pos = list(chain(*[list(range(l)) for l in utt_lens]))[-self.max_len:] #chain是把多个list连成一个list访问
            pos.extend(list(chain(*[list(range(l)) for l in k_lens]))[-self.max_len:])
            src_pos.append(pos)

            # Turn ids
            turn = list(chain(*[[len(utts) - i] * l for i, l in enumerate(utt_lens)]))[-self.max_len:]
            turn.extend(list(chain(*[[i] * l for i, l in enumerate(k_lens)]))[-self.max_len:])
            src_turn.append(turn)

            # Role ids
            role = list(chain(*[[self.bot_id if (len(utts)-i) % 2 == 0 else self.user_id] * l
                                for i, l in enumerate(utt_lens)]))[-self.max_len:]
            role.extend(list(chain(*[[self.knowledge_id] * l for l in k_lens]))[-self.max_len:])
            src_role.append(role)
        
        src_token = list2np(src_token, padding=self.pad_id)
        src_pos = list2np(src_pos, padding=self.pad_id)
        src_turn = list2np(src_turn, padding=self.pad_id)
        src_role = list2np(src_role, padding=self.pad_id)

        batch = {}
        batch["src_token"] = src_token
        batch["src_mask"] = (src_token != self.pad_id).astype("int64")#mask位的id为0，正常话语的id是1
        batch["src_pos"] = src_pos
        batch["src_type"] = src_role
        batch["src_turn"] = src_turn
        batch["k_max_len"] = k_max_len #标记知识的最大范围。
        if "tgt" in samples[0]:
            tgt = [sp["tgt"] for sp in samples]

            # Token ids & Label ids
            tgt_token = list2np(tgt, padding=self.pad_id)

            # Position ids
            tgt_pos = np.zeros_like(tgt_token)
            tgt_pos[:] = np.arange(tgt_token.shape[1], dtype=tgt_token.dtype)

            # Turn ids
            tgt_turn = np.zeros_like(tgt_token)

            # Role ids
            tgt_role = np.full_like(tgt_token, self.bot_id)

            batch["tgt_token"] = tgt_token
            batch["tgt_mask"] = (tgt_token != self.pad_id).astype("int64")
            batch["tgt_pos"] = tgt_pos
            batch["tgt_type"] = tgt_role
            batch["tgt_turn"] = tgt_turn
            if "postive" in samples[0]:
                postive = [sp["postive"] for sp in samples]
                postive_token = list2np(postive,padding = self.pad_id)
                
                postive_token_pos = np.zeros_like(postive_token)
                postive_token_pos[:] = np.arange(postive_token_pos.shape[1],dtype = postive_token.dtype)
                
                postive_role = np.full_like(postive_token,self.user_id)
                postive_turn = np.zeros_like(postive_token)

                batch["postive_token"] = postive_token
                batch["postive_token_pos"] = postive_token_pos
                batch["postive_type"] = postive_role
                batch["postive_turn"] = postive_turn
                batch["postive_mask"] = (postive_token != self.pad_id).astype("int64")

            if "negative" in samples[0]:
                
                negative = [sp["negative"] for sp in samples]
                negative_token = list2np(negative,padding = self.pad_id)
                
                negative_token_pos = np.zeros_like(negative_token)
                negative_token_pos[:] = np.arange(negative_token_pos.shape[1],dtype = negative_token.dtype)
                negative_role = np.full_like(negative_token,self.user_id)
                negative_turn = np.zeros_like(negative_token)
                batch["negative_token"] = negative_token
                batch["negative_token_pos"] = negative_token_pos
                batch["negative_type"] = negative_role
                batch["negative_turn"] = negative_turn
                batch["negative_mask"] = (negative_token != self.pad_id).astype("int64")
        return batch, batch_size
