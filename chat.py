import os
import argparse

from plato.args import str2bool
from plato.args import parse_args
from plato.data.dataset import Dataset
from plato.data.field import BPETextField
from plato.models.generator import Generator
import json
import numpy as np
import paddle.fluid as fluid


from plato.args import str2bool
from plato.data.data_loader import DataLoader
from plato.data.dataset import LazyDataset
from plato.trainer import Trainer
from plato.models.model_base import ModelBase
import plato.modules.parallel as paralle

# parser = argparse.ArgumentParser()
# target = {"context":""}
# BPETextField.add_cmdline_argument(parser)
# args = parse_args(parser)
# print(args)
# bpe = BPETextField(args.BPETextField)
# BUILD_EXAMPLES_FN = {
#         "multi": bpe.build_example_multi_turn,
#         "multi_knowledge": bpe.build_example_multi_turn_with_knowledge
#     }
# build_examples_fn = BUILD_EXAMPLES_FN["multi_knowledge"]
# import json
# with open("./example2.json","r") as f:
#     req = json.load(f)
# print(len(build_examples_fn(req)))
# print(len(build_examples_fn(req)[0]["src"]))
# print(len(build_examples_fn(req)[0]["knowledge"]))


# do_infer=True


#     if hparams.do_infer:
#         raw_test_file = os.path.join(hparams.data_dir, "dial.test")
#         test_file = raw_test_file + f".{hparams.tokenizer_type}.jsonl"
#         assert os.path.exists(test_file), f"{test_file} isn't exist"
#         test_dataset = LazyDataset(test_file)
#         test_loader = DataLoader(test_dataset, hparams.Trainer, collate_fn=collate_fn, is_test=hparams.do_infer)

#     def to_tensor(array):
#         array = np.expand_dims(array, -1)
#         return fluid.dygraph.to_variable(array)

#     if hparams.use_data_distributed:
#         place = fluid.CUDAPlace(parallel.Env().dev_id)
#     else:
#         place = fluid.CUDAPlace(0)

#     with fluid.dygraph.guard(place):
#         # Construct Model
#         model = ModelBase.create("Model", hparams, generator=generator)

#         # Construct Trainer
#         trainer = Trainer(model, to_tensor, hparams.Trainer)

#         if hparams.do_train:
#             # Training process
#             for epoch in range(hparams.num_epochs):
#                 trainer.train_epoch(train_loader, valid_loader)

#         if hparams.do_test:
#             # Validation process
#             trainer.evaluate(test_loader, need_save=False)

#         if hparams.do_infer:
#             # Inference process
#             def split(xs, sep, pad):
#                 """ Split id list by separator. """
#                 out, o = [], []
#                 for x in xs:
#                     if x == pad:
#                         continue
#                     if x != sep:
#                         o.append(x)
#                     else:
#                         if len(o) > 0:
#                             out.append(list(o))
#                             o = []
#                 if len(o) > 0:
#                     out.append(list(o))
#                 assert(all(len(o) > 0 for o in out))
#                 return out

#             def parse_context(batch):
#                 """ Parse context. """
#                 return bpe.denumericalize([split(xs, bpe.eos_id, bpe.pad_id)
#                                            for xs in batch.tolist()])

#             def parse_text(batch):
#                 """ Parse text. """
#                 return bpe.denumericalize(batch.tolist())

#             infer_parse_dict = {
#                 "src": parse_context,
#                 "tgt": parse_text,
#                 "preds": parse_text
#             }
#             trainer.infer(test_loader, infer_parse_dict, num_batches=hparams.num_infer_batches)


if __name__ == "__main__":
    text = "START Guo Shaoyun Blessed love __eou__ Blessed love to star Guo Shaoyun __eou__ Guo Shaoyun comment Tvber in my eyes __eou__ Guo Shaoyun date of birth 1970-8-25 __eou__ Guo Shaoyun height 168cm __eou__ Guo Shaoyun Gender female __eou__ Guo Shaoyun occupation performer __eou__ Guo Shaoyun field Star __eou__ Blessed love Comments on time.com Teacher Bai's play is to watch! __eou__ Blessed love Release date information It was shown last month __eou__ Blessed love to star Guo Shaoyun __eou__ Blessed love type Fantasy __eou__ Blessed love field film __eou__ Guo Shaoyun describe a girl from a rich family __eou__ Guo Shaoyun Ancestral home Hong Kong, China __eou__ Guo Shaoyun nation Han nationality	Who do you know? __eou__ There's a lot of money there. Who are you talking about? __eou__ Guo Shaoyun, do you know? __eou__ I've heard of her.	Mm-hmm! She's also the star of the movie God bless love! Have you seen it?"
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--do_train", type=str2bool, default=False,
                        help="Whether to run trainning.")
    parser.add_argument("--do_test", type=str2bool, default=False,
                        help="Whether to run evaluation on the test dataset.")
    parser.add_argument("--do_infer", type=str2bool, default=False,
                        help="Whether to run inference on the test dataset.")
    parser.add_argument("--num_infer_batches", type=int, default=None,
                        help="The number of batches need to infer.\n"
                        "Stay 'None': infer on entrie test dataset.")
    parser.add_argument("--hparams_file", type=str, default=None,
                        help="Loading hparams setting from file(.json format).")
    
    BPETextField.add_cmdline_argument(parser)
    Dataset.add_cmdline_argument(parser)
    Trainer.add_cmdline_argument(parser)
    ModelBase.add_cmdline_argument(parser)
    Generator.add_cmdline_argument(parser)
    args = parse_args(parser)



    bpe = BPETextField(args.BPETextField)
    generator = Generator.create(args.Generator, bpe=bpe) # 生成
    args.Model.num_token_embeddings = bpe.vocab_size
    knowledge, src, tgt = text.strip("\n").split("\t")
    req = dict(context = [src],knowledge=[knowledge]) #将对话进行分别为knoledge,src,tgt
    BUILD_EXAMPLES_FN = {
        "multi": bpe.build_examples_multi_turn,
        "multi_knowledge": bpe.build_examples_multi_turn_with_knowledge
    }
    build_example_fn = bpe.build_example_multi_turn_with_knowledge
  
    collate_fn = bpe.collate_fn_multi_turn_with_knowledge #padding 
    data= build_example_fn(req)
    dataset = Dataset(data)
    test_loader = DataLoader(dataset, args.Trainer, collate_fn=collate_fn, is_test=args.do_infer)
    #raw_test_file = os.path.join(args.data_dir, "dial.test")
    #test_file = raw_test_file + f".{args.tokenizer_type}.jsonl"
    #assert os.path.exists(test_file), f"{test_file} isn't exist"
    #test_dataset = LazyDataset(test_file)
    #test_loader = DataLoader(test_dataset, args.Trainer, collate_fn=collate_fn, is_test=args.do_infer)

    print(json.dumps(args,indent=4))
    #for s in test_loader:
    #    print(s)
    #print(len(test_loader))

    def to_tensor(array):
        array = np.expand_dims(array, -1)
        return fluid.dygraph.to_variable(array)

    #place = fluid.CUDAPlace(0)
    if args.use_data_distributed:
        place = fluid.CUDAPlace(args.Env().dev_id)
    else:
        place = fluid.CUDAPlace(0)

    with fluid.dygraph.guard(place):
        # Construct Model
        model = ModelBase.create("Model", args, generator=generator)

        # Construct Trainer
        trainer = Trainer(model, to_tensor, args.Trainer)

        
        # Inference process
        def split(xs, sep, pad):
            """ Split id list by separator. """
            out, o = [], []
            for x in xs:
                if x == pad:
                    continue
                if x != sep:
                    o.append(x)
                else:
                    if len(o) > 0:
                        out.append(list(o))
                        o = []
            if len(o) > 0:
                out.append(list(o))
            assert(all(len(o) > 0 for o in out))
            return out

        def parse_context(batch):
            """ Parse context. """
            return bpe.denumericalize([split(xs, bpe.eos_id, bpe.pad_id)
                                        for xs in batch.tolist()])

        def parse_text(batch):
            """ Parse text. """
            return bpe.denumericalize(batch.tolist())

        infer_parse_dict = {
            "src": parse_context,
            "tgt": parse_text,
            "preds": parse_text
        }
        msessage = trainer.infer_chat(test_loader, infer_parse_dict, num_batches=args.num_infer_batches)
        print(msessage)
