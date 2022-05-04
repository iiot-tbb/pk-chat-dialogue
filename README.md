# 实验记录
## 20210928
##### 内存溢出，设置的是最大句子长度是256,所以要检验一下这个问题。
##### uni-transformer的最长长度是256,比较一下是哪个更长一些

## 20210930
##### 导出了知识图谱中的实体和边的信息。
##### duconv的数据是机器人先发问，因此训练的时候需要注意。
##### duRecdical 有机器人主动发问的，也有人主动发问的
##### KG中的实体和关系
##### 实体类型标识:www.acekg.cn/concept/
##### 边类型标识

- www.acekg.cn/property/
- www.acekg.cn/relation/
## 20211001
##### 记录用到的虚拟环境：
- plato
- plato_ceshi 可以跑mutil_gpu
- palto_abs 增加了summarize的模块，看是否影响到了plato环境
- plato_visual 现在用的环境。


## 20211002
##### 在编写数据集的时候，遇到很多重名的人，因此我打算先忽略那些人名相同人的，只考虑不同姓名人，
##### 在编写对话数据的时候，可以将那些用不到的字段剔除掉。

## 20211005
##### :可以去掉的字段
- label
- sameas
- preflabel
- url或page中的一项
- tag
##### 突然想到可以增加一个判断知识是否是正确的知识来扩大训练任务。

## 20211009 
##### 特殊标记[ABSTRACT]专门是为了abstract字段设计的,[INTRODUCTION]当文本太长时做替换，为了拿到部分语意信息，可以做缩略版的intro
##### 加入一个学习任务判断知识是否属于当前knowledge。

## 20211016

##### 数据集制作翻译完成。数据制造完成。
##### 下一步工作计划:
- 处理之前公共数据集中的最长长度限制。
- 理清max_len的最大长度。
- 训练模型。
- 设计关键词提取算法：针对当前场景。
- 将数据导入neo4j
- 论文撰写且与模型各部分的优化。


##### 关于对话数据集中数据的长度 
- dailydialog :  (3, 865)
- personchat :  (22, 582)
- DSTC :  (22, 666)
- DDE :  (71, 522)

根据最长长度的猜测,模型中有机制来选取长度合适的句子,因此，需要关注源码中关于句子长度的处理方式。
上面的句子长度主要在 `field.py` 文件里，处理方式大致如下:
- 每一个utt有一个最长长度
- 总的utt+knwledge的长度不超过256
- 有最少轮次和最多轮次数目。
- 原则上可以不用自己处理句子长度的问题。
- 也可以加入自己的裁原则

## 20211020
- 对数据进行了shuffle。
- 修改了数据汇总存在'\n'的问题。
- 数据处理步骤，
    - `Baidu_Text_transAPI.py`翻译
    - `Transform_to_plato_form.py`进行转换格式
    - `Ace_Dialog/shuffle.py`进行shuffle。
    - `merge_all.py`进行合并数据

## 20211021评测结果
- 20 epoch DailyDialog :BLEU_1-0.388   BLEU_2-0.304   INTRA_DIST_1-0.936   INTRA_DIST_2-0.987   INTER_DIST_1-0.055   INTER_DIST_2-0.303   LEN-10.993   TIME-34869.708
- 7 epoch AceDialog and plato and AceDialog BLEU_1-0.417   BLEU_2-0.329   INTRA_DIST_1-0.936   INTRA_DIST_2-0.983   INTER_DIST_1-0.139   INTER_DIST_2-0.335   LEN-10.360   TIME-1835.647
- 27 epoch and plato and AceDialog BLEU_1-0.594   BLEU_2-0.486   INTRA_DIST_1-0.929   INTRA_DIST_2-0.995   INTER_DIST_1-0.134   INTER_DIST_2-0.362   LEN-12.341   TIME-1690.157
- 20 epoch and plato  and AceDialog BLEU_1-0.415   BLEU_2-0.354   INTRA_DIST_1-0.966   INTRA_DIST_2-0.966   INTER_DIST_1-0.165   INTER_DIST_2-0.361   LEN-7.407   TIME-1507.834
    Recall:0.08292419662518917
    Precison:0.23521822898554215
    F1:0.11312418492557835
    Recall/Precision/F1:0.0829/0.2352/0.1131
    3beam BLEU_1-0.424   BLEU_2-0.361   INTRA_DIST_1-0.964   INTRA_DIST_2-0.968   INTER_DIST_1-0.165   INTER_DIST_2-0.366   LEN-7.543   TIME-1362.190

- 47 epoch and plato and AceDialog   BLEU_1-0.960   BLEU_2-0.863   INTRA_DIST_1-0.932   INTRA_DIST_2-0.981   INTER_DIST_1-0.168   INTER_DIST_2-0.457   LEN-11.346   TIME-1595.76
    Recall:0.09926801010102905
    Precison:0.2176272480981623
    F1:0.1247091653397901
    Recall/Precision/F1:0.0993/0.2176/0.1247

    

- 20 epoch and pointer_network 37配比 BLEU_1-0.100   BLEU_2-0.076   INTRA_DIST_1-0.991   INTRA_DIST_2-0.975   INTER_DIST_1-0.055   INTER_DIST_2-0.127   LEN-4.427   TIME-1723.121
- 13 epoch context_pointer  BLEU_1-0.269   BLEU_2-0.227   INTRA_DIST_1-0.978   INTRA_DIST_2-0.967   INTER_DIST_1-0.173   INTER_DIST_2-0.367   LEN-5.684   TIME-1616.154
- 17 epoch context_pointer BLEU_1-0.346   BLEU_2-0.289   INTRA_DIST_1-0.966   INTRA_DIST_2-0.980   INTER_DIST_1-0.156   INTER_DIST_2-0.341   LEN-7.252   TIME-1465.656
- 20 epoch context_pointer BLEU_1-0.426   BLEU_2-0.359   INTRA_DIST_1-0.958   INTRA_DIST_2-0.967   INTER_DIST_1-0.156   INTER_DIST_2-0.353   LEN-7.690   TIME-2009.194
    Recall:0.0783972685126393
    Precison:0.2318819376991122
    F1:0.10930082172545103
    Recall/Precision/F1:0.0784/0.2319/0.1093

- 20 eopoch context_pointer2 beam_size 3 BLEU_1-0.379   BLEU_2-0.317   INTRA_DIST_1-0.959   INTRA_DIST_2-0.972   INTER_DIST_1-0.159   INTER_DIST_2-0.364   LEN-7.443   TIME-1325.052

- 25 epoche context_pointer2 beam_size 3
BLEU_1-0.484   BLEU_2-0.411   INTRA_DIST_1-0.955   INTRA_DIST_2-0.974   INTER_DIST_1-0.164   INTER_DIST_2-0.383   LEN-8.058   TIME-1407.879
BLEU_1-0.541   BLEU_2-0.466   INTRA_DIST_1-0.957   INTRA_DIST_2-0.970   INTER_DIST_1-0.163   INTER_DIST_2-0.398   LEN-8.194   TIME-1364.405
Recall/Precision/F1:0.0799/0.2229/0.1094

Recall/Precision/F1:0.0825/0.2162/0.1101
BLEU_1-0.630   BLEU_2-0.549   INTRA_DIST_1-0.949   INTRA_DIST_2-0.983   INTER_DIST_1-0.162   INTER_DIST_2-0.401   LEN-9.233   TIME-1175.168
Recall/Precision/F1:0.0825/0.2229/0.1204
- 20 epoch and context_pointer DSTC 
 BLEU_1-0.001   BLEU_2-0.001   INTRA_DIST_1-0.980   INTRA_DIST_2-1.000   INTER_DIST_1-0.054   INTER_DIST_2-0.199   LEN-7.222   TIME-7124.544

Bleu_1: 0.7321
Bleu_2: 0.5954
Bleu_3: 0.4927
Bleu_4: 0.4087
METEOR: 0.2513
ROUGE_L: 0.5500
CIDEr: 1.0035
- 20 epoch and gpt DSTC
BLEU_1-0.002   BLEU_2-0.001   INTRA_DIST_1-0.960   INTRA_DIST_2-0.930   INTER_DIST_1-0.027   INTER_DIST_2-0.060   LEN-5.451   TIME-6851.663
Bleu_1: 0.3338
Bleu_2: 0.1891
Bleu_3: 0.0939
Bleu_4: 0.0459
METEOR: 0.0894
ROUGE_L: 0.2458
CIDEr: 0.0893

DSTC_pointer2_network
Bleu_1: 0.7004
Bleu_2: 0.5640
Bleu_3: 0.4634
Bleu_4: 0.3822
METEOR: 0.2412
ROUGE_L: 0.5351
CIDEr: 0.9234

- 20 poech and pointer1 ACE
[Infer][0]   BLEU_1-0.202   BLEU_2-0.173   INTRA_DIST_1-0.958   INTRA_DIST_2-0.856   INTER_DIST_1-0.258   INTER_DIST_2-0.514   LEN-4.748   TIME-1268.691


lstrmcrf:precision: 0.549 recall: 0.630 f1: 0.587
f1/precison/recall:0.4657534246575343 0.38636363636363635 0.5862068965517241
mohu:f1/precision/recall:0.6027397260273972 0.5 0.7586206896551724

f1/precison/recall:0.9032171939660927 0.9058776275271121 0.9005723412751231
mohu:0.9551461754104927 0.9579595662069889 0.9523492612804473


validation loss: 3.58266, precision: 0.764 recall: 0.742 f1: 0.753 accuracy: 0.922 
overall best f1 is 0.776867 at 12 epoch

trans_tpoic--f1/precision/recall: 0.9856707661062846 1.0 0.9717463848720801

0.9774795268425841 1.0 0.9559510567296997   TIME-29551.885

## 20211107
- 记录到一个bug,创建的变量参数必须在forward里面用到，否则就会出现未分配内存的情况。


## 20211110
- 首先用的指针网络指向context部分的embedding向量。目前效果很差，一共目前为止试了以下几种方法:
    - 3,7配比pointer weight和 nll weight。
    - 5，5配比pointer weight和 nll weight。
    - 自适应学习权重
    - 还未考虑最后一层隐状态embedding的输出
- 如果上面效果很差，可以将知识外部化考虑

## 20211128
- 改变了lambda的计算方式，用tgt pooling和context pooling做点乘后算atention来计算pointer network的比例。(先看看效果。)

## 20211130
- bug,没有assgin会共享内存,不用字典竟然没事
# 文件说明

#### `chat.py`:对话文件，生成的对话聊天
#### `../data_en/Baidu_text_transAPI.py`:将中文数据翻译成英文
#### `../data_en/Transform_to_plate_form.py`:将`json`格式的文本转换问纯文本形式,方便后续模型处理
#### `../data_en/get_DDE_Data.py`:取得实验室数据中的三元组数据
#### `../data_en/duconvEn_tran.txt`知识对话数据,`../data_en/duRecDialEn_train2.txt`推荐对话数据 
#### `../data_en/make_abstract.py`缩短摘要的长度
#### `../make_dialog.py`:制作对话数据
#### `../data_en/duRec_palto.txt`,`../data_en/duconv_plato.txt` 符合输入的对话数据
# 修改记录
- `tokenizer.py` 的 162行加入自己的never_split
# 采用的造对话数据的方式

- 给出关于某个知识的三元组信息，一问一答回复。
- 接入翻译API，我说一句中文，翻译成英文。
- 存到某一个文件中。

# idea：
- 把cbow变成预测是否是实体，或者有没有更换实体的信息。
- 加入 pointer network
- 直接用 pointer network 指向送入网络的知识不太好实现，可以把知识外部化。再思考一下两个方案的可行性。
- 引入闲聊的方式的一个目的是，如果知识引用错误，也不会生成不合理的句子，另一方面，如果用户真实情况下并未说出带有知识的问题，也可以给出流畅的回复，而不是错误的应用网络可能记忆到的知识，因此在训练闲聊任务时，一方面，加入错误的外部知识，另一方面，加入特殊的标记字段来表示知识无引用。


# 关于判断topic transfer 的工作

- 首先，在模型里加入这一部分
- 第二，在数据中进行修改，(直觉上对数据集的修改比较麻烦，先做第一步比较好)


# 论文问题：

``` 
make all                      # 编译生成 main.pdf
make clean                    # 删除编译所产生的中间文件
make cleanall                 # 删除 main.pdf 和所有中间文件
make wordcount                # 论文字数统计 
rm -Rf "$(biber --cache)"清理缓存。
```

