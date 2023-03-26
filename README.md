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
#### `../data_en/Baidu_text_transAPI.py`:将公开数据集中文数据翻译成英文
#### `../data_en/Transform_to_plate_form.py`:将`json`格式的文本转换问纯文本形式,方便后续模型处理
#### `../data_en/get_DDE_Data.py`:取得实验室数据中的三元组数据
#### `../data_en/duconvEn_tran.txt`知识对话数据,`../data_en/duRecDialEn_train2.txt`推荐对话数据 
#### `../data_en/make_abstract.py`缩短摘要的长度
#### `../make_dialog.py`:制作对话数据，及将后处理后得到的数据翻译为英文数据
#### `../data_en/duRec_palto.txt`,`../data_en/duconv_plato.txt` 符合输入的对话数据
#### `../data_en/op_neo4j,py` 插入到neo4j中
### `../data_en/postProcess.py` 对初始对话数据做一些调整。
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

# 工程部署上线问题：
lstm:uvicorn fastap:app --host 0.0.0.0 --port 8074 --reload

- 当前在对话生成部分，由于加入了判别模块，因此，原有可以跑2batch的大小，现在只能跑1batch且速度慢，关于该部分的
解决方法，初步设想是将模型参数加载为原先训练好的模型的部分，然后在此基础上微调。

- 关于entity抽取部分，该部分主要的问题是抽取过于细节，拿全部的对话数据的样本有些过拟合当前需求部分的样本。解决方案是：
    - 只考虑在地学部分的样本
    - 引入正则表达式去抽取括号的内容，也就是说让用户对自己要查找的实体标注{}，(感觉这样有点不自然，but目前来说可以作为一个冷启动的方案去实现。)

- 关于话题转换部分的判断，目前是最让人头疼的地方，在之前的抽取正负例子时，设计过于简单，只是拿上轮对话和当前对话的内容切含有关键词信息的句子作为正负样本，因此在判断转换时能力较弱，所以，我认为在这里设计时，应强调当用户问出一个新问题时(例如是询问新的实体的内容，才会判断为需要进行转换，其它情况一律不做转换，这里要配合entity抽取部分，如果用正则表达式直接抽出，则应转换话题。)。

- 先找3个比较有代表性的例子，然后对这几个例子进行对话，发现其中问题，调整。
    - sedimentation on antarctic sea floor
    - aapg bulletin
    - haiti
    - wilkes station

在Transform_to_plato_form.py里:
negative是本文中的带知识文本
postive是上一轮中的带知识文本
且存入数据时，postive在前面，negative在后面

但是在Plato/field.py里取出数据的时候，是把前面positve当作negative的样本，negative作为positive的样本。

且在plato模型训练的时候，postive的样本让其接近1，negative的样本使其接近0。
因此，也就是说当用户的话语中存在带本轮中知识时，使其接近1，反之使其接近0.




User:  hi,xiaobo,how are you?

Bot:  let me check . . .
User:  can you tell me somthing about {sedimentation on antarctic sea floor}?


Bot:  well , it's a little long . it's about timescale and lower cretaceous .
User:  ok，I want to know what timescale does it mentioned?

Bot:  well , this article is published in aapg bulletin , which is a high-quality article
User:  ok ,what else do you know about sedimentation on antarctic sea floor?

Bot:  this article is written by eugene domack , published in aapg bulletin
User:  ok，how about it timescale?

Bot:  mm-hmm . it belongs to timescale late pliocene .
User:  ok, thank you very much.

Bot:  you're welcome . it's a great book . it's a very interesting book .
User:  haha,I also know somthing about the {haiti},do you know it ?


Bot:  yes , its alpha code is ht .
User:  haha,is haiti is their official name?

Bot:  yes , its official name is ht . you can go to see it when you have time .
User:  haha, you know a lot, thank you for your information.

Bot:  you're welcome . this is a little fun . i wonder if you can help me ?
User:  I also want to know a place named {wilkes station}.

Bot:  the longitude of this place is-86 . 778
User:  ok,how about it's laittude? 

Bot:  yes , the longitude of this place is-86 . 778
User:  what is the geohash of wilkes station?

Bot:  its geohash value is 35 . 928
User:  ok, it' still not good enough,byebye.

Bot:  bye-bye .




User:  hello,how are you?
entity:: None
now_score:  0.44770148396492004
[{'preds': "i'm good . can i help you ?", 'scores': -4.454295635223389}]
Bot:  i'm good . can i help you ?
User:  yes, I wang to know some information about {sedimentation on antarctic sea floor},do you know something about it?
entity:: sedimentation on antarctic sea floor
now_score:  0.5219379663467407
[{'preds': 'yes , i know . its longitude is-95 . 4487 and its corresponding longitude is-95 . 4487 .', 'scores': -7.049464702606201}]
Bot:  yes , i know . its longitude is-95 . 4487 and its corresponding longitude is-95 . 4487 .
User:  ok, what is the laittitude of it ?      
entity:: None
now_score:  0.44597145915031433
[{'preds': 'uh huh , i know . its longitude is-95 . 4487 and its geohash is 29 . 03619 .', 'scores': -7.130092144012451}]
Bot:  uh huh , i know . its longitude is-95 . 4487 and its geohash is 29 . 03619 .
User:  ok, thank you for your information.
entity:: None
now_score:  0.47723332047462463
[{'preds': "you're welcome . the longitude of this place is-95 . 4487 and the geohash value is 29 . 03619 .", 'scores': -10000000000.0}]
Bot:  you're welcome . the longitude of this place is-95 . 4487 and the geohash value is 29 . 03619 .
User:  ok, I also want to know {aapg bulletin}。
entity:: aapg bulletin
now_score:  0.5431324243545532
[{'preds': 'oh , i know . this is an entity of location type .', 'scores': -7.32192325592041}]
Bot:  oh , i know . this is an entity of location type .
User:  ok, what else do you know about aapg bulletin?
entity:: None
now_score:  0.41006240248680115
[{'preds': 'well , i only know that its longitude is-95 . 4487 and its longitude is-95 . 4487 .', 'scores': -4.84411096572876}]
Bot:  well , i only know that its longitude is-95 . 4487 and its longitude is-95 . 4487 .
User:  oh, thank you , I also want to know something about {haiti}.
entity:: haiti
now_score:  0.5342296957969666
[{'preds': "ha ha , you're welcome . i'm sorry i didn't help you .", 'scores': -8.773594856262207}]
Bot:  ha ha , you're welcome . i'm sorry i didn't help you .
User:  do you know somthing about {wilkes station}?
entity:: wilkes station
now_score:  0.4594629108905792
[{'preds': 'yes , it is located at-95 . 4487 and is located in the location category .', 'scores': -10.47912311553955}]
Bot:  yes , it is located at-95 . 4487 and is located in the location category .
User:  what else do you know about it?
entity:: None
now_score:  0.39715877175331116
[{'preds': 'i know that its longitude is-95 . 4487 and latitude is 29 . 03619 .', 'scores': -4.401233196258545}]
Bot:  i know that its longitude is-95 . 4487 and latitude is 29 . 03619 .
User:  anything else ?
entity:: None
now_score:  0.41474100947380066
[{'preds': 'its longitude is-95 . 44887 and its geohash is 9vhjhjhjbk0 .', 'scores': -4.219842433929443}]
Bot:  its longitude is-95 . 44887 and its geohash is 9vhjhjhjbk0 .
User:  ok ,thank you for your knowledge,byebye~
entity:: None
now_score:  0.5021130442619324
[{'preds': "you're welcome . bye", 'scores': -1.9833500385284424}]
Bot:  you're welcome . bye