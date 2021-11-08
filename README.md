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
- 47 epoch and plato and AceDialog   BLEU_1-0.960   BLEU_2-0.863   INTRA_DIST_1-0.932   INTRA_DIST_2-0.981   INTER_DIST_1-0.168   INTER_DIST_2-0.457   LEN-11.346   TIME-1595.763

## 20211107
- 记录到一个bug,创建的变量参数必须在forward里面用到，否则就会出现未分配内存的情况。

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
