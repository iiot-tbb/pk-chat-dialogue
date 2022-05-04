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
UnifiedTransformer
"""

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import FC
import paddle.fluid.layers as layers
from paddle.fluid.layers.nn import shape

from plato.args import str2bool
from plato.modules.embedder import Embedder
import plato.modules.functions as F
from plato.modules.layer_norm import LayerNorm
from plato.modules.transformer_block import TransformerBlock
from plato.models.model_base import ModelBase


class UnifiedTransformer(ModelBase):
    """
    Implement unified transformer.
    """

    @classmethod
    def add_cmdline_argument(cls, group):
        """ Add cmdline argument. """
        group.add_argument("--num_token_embeddings", type=int, default=-1,
                           help="The number of tokens in vocabulary. "
                           "It will be automatically calculated after loading vocabulary.")
        group.add_argument("--num_pos_embeddings", type=int, default=512,
                           help="The maximum number of position.")
        group.add_argument("--num_type_embeddings", type=int, default=2,
                           help="The number of different type of tokens.")
        group.add_argument("--num_turn_embeddings", type=int, default=16,
                           help="The maximum number of turn.")
        group.add_argument("--num_latent", type=int, default=20,
                           help="The number of latent.")
        group.add_argument("--tau", type=float, default=0.67,
                           help="The parameter of gumbel softmax.")
        group.add_argument("--with_bow", type=str2bool, default=True,
                           help="Whether to use BoW loss.")
        group.add_argument("--hidden_dim", type=int, default=768,
                           help="The size of hidden vector in transformer.")
        group.add_argument("--num_heads", type=int, default=12,
                           help="The number of heads in multi head attention.")
        group.add_argument("--num_layers", type=int, default=12,
                           help="The number of layers in transformer.")
        group.add_argument("--padding_idx", type=int, default=0,
                           help="The padding index.")
        group.add_argument("--dropout", type=float, default=0.1,
                           help="The dropout ratio after multi head attention and feed forward network.")
        group.add_argument("--embed_dropout", type=float, default=0.0,
                           help="The dropout ratio of embedding layers.")
        group.add_argument("--attn_dropout", type=float, default=0.1,
                           help="The dropout ratio of multi head attention.")
        group.add_argument("--ff_dropout", type=float, default=0.1,
                           help="The dropout ratio of feed forward network.")
        group.add_argument("--use_discriminator", type=str2bool, default=False,
                           help="Whether to use discriminator loss.")
        group.add_argument("--dis_ratio", type=float, default=1.0,
                           help="The ratio of discriminator loss.")
        group.add_argument("--weight_sharing", type=str2bool, default=True,
                           help="Whether to share weight between token embedding and "
                           "predictor FC layer.")
        group.add_argument("--pos_trainable", type=str2bool, default=True,
                           help="Whether to train position embeddings.")
        group.add_argument("--two_layer_predictor", type=str2bool, default=False,
                           help="Use two layer predictor. "
                           "Traditional BERT use two FC layers to predict masked token.")
        group.add_argument("--bidirectional_context", type=str2bool, default=True,
                           help="Whether to use bidirectional self-attention in context tokens.")
        group.add_argument("--label_smooth", type=float, default=0.0,
                           help="Use soft label to calculate NLL loss and BoW loss.")
        group.add_argument("--initializer_range", type=float, default=0.02,
                           help="Use to initialize parameters.")

        group.add_argument("--lr", type=float, default=5e-5,
                           help="The inital learning rate for Adam.")
        group.add_argument("--weight_decay", type=float, default=0.0,
                           help="The weight decay for Adam.")
        group.add_argument("--max_grad_norm", type=float, default=5.0,
                           help="The maximum norm of gradient.")
        group.add_argument("--use_pointer_network",type =int,default = -1,
                           help = "use pointer network to process knowledge") #新加入判断是否采用指针网络
        group.add_argument("--use_topic_trans_judge",type = str2bool,default = False,
                            help = "use topic transfer to judge whether to get the new knowledge") #新加入判别部分判断是否需要转换主题。
        return group

    def __init__(self, name_scope, hparams, generator, dtype="float32"):
        super().__init__(name_scope, hparams)
        self.generator = generator
        self.num_token_embeddings = hparams.num_token_embeddings
        self.num_pos_embeddings = hparams.num_pos_embeddings
        self.num_type_embeddings = hparams.num_type_embeddings
        self.num_turn_embeddings = hparams.num_turn_embeddings
        self.num_latent = hparams.num_latent
        self.tau = hparams.tau
        self.with_bow = hparams.with_bow
        self.hidden_dim = hparams.hidden_dim
        self.num_heads = hparams.num_heads
        self.num_layers = hparams.num_layers
        self.padding_idx = hparams.padding_idx
        self.dropout = hparams.dropout
        self.embed_dropout = hparams.embed_dropout
        self.attn_dropout = hparams.attn_dropout
        self.ff_dropout = hparams.ff_dropout
        self.use_discriminator = hparams.use_discriminator
        self.weight_sharing = hparams.weight_sharing
        self.pos_trainable = hparams.pos_trainable
        self.two_layer_predictor = hparams.two_layer_predictor
        self.bidirectional_context = hparams.bidirectional_context
        self.label_smooth = hparams.label_smooth
        self.initializer_range = hparams.initializer_range
        self.use_pointer_network = hparams.use_pointer_network
        self.use_topic_trans = hparams.use_topic_trans_judge

        self.embedder = Embedder(self.full_name(),
                                 self.hidden_dim,
                                 self.num_token_embeddings,
                                 self.num_pos_embeddings,
                                 self.num_type_embeddings,
                                 self.num_turn_embeddings,
                                 padding_idx=self.padding_idx,
                                 dropout=self.embed_dropout,
                                 pos_trainable=self.pos_trainable)
        self.embed_layer_norm = LayerNorm(self.full_name(),
                                          begin_norm_axis=2,
                                          epsilon=1e-12,
                                          param_attr=fluid.ParamAttr(
                                              regularizer=fluid.regularizer.L2Decay(0.0)),
                                          bias_attr=fluid.ParamAttr(
                                              regularizer=fluid.regularizer.L2Decay(0.0)))
        if self.use_topic_trans:#主题转换预测
            self.transtor = FC(name_scope = self.full_name()+".transtor",
                                        size = 1,
                                        act ="sigmoid")
        self.layers = []
        for i in range(hparams.num_layers):
            layer = TransformerBlock(self.full_name(),
                                     self.hidden_dim,
                                     self.num_heads,
                                     self.dropout,
                                     self.attn_dropout,
                                     self.ff_dropout)
            self.layers.append(layer)
            self.add_sublayer(f"layer_{i}", layer)

        if self.num_latent > 0: #隐变量预测
            self.post_network = FC(name_scope=self.full_name() + ".post_network",
                                   size=self.num_latent,
                                   bias_attr=False)

            if self.use_discriminator: #判别器预测
                self.dis_ratio = hparams.dis_ratio
                self.discriminator = FC(name_scope=self.full_name() + ".discriminator",
                                        size=1,
                                        act="sigmoid")
        

        if self.two_layer_predictor: #cbow预测
            self.pre_predictor = FC(name_scope=self.full_name() + ".pre_predictor",
                                    size=self.hidden_dim,
                                    num_flatten_dims=2, #假设维度为[4,5,6,7,8],则flatten后为[4*5,6*7*8]
                                    act="gelu")
            if self.num_latent > 0 and self.with_bow:
                self.pre_bow_predictor = FC(name_scope=self.full_name() + ".pre_bow_predictor",
                                            size=self.hidden_dim,
                                            act="gelu")
        if not self.weight_sharing:
            self.predictor = FC(name_scope=self.full_name() + ".predictor",
                                size=self.num_token_embeddings,
                                num_flatten_dims=2,
                                bias_attr=False)
        if self.num_latent > 0 and self.with_bow:
            self.bow_predictor = FC(name_scope=self.full_name() + ".bow_predictor",
                                    size=self.num_token_embeddings,
                                    bias_attr=False)

        self.max_grad_norm = hparams.max_grad_norm
        if self.max_grad_norm is not None:
            self.grad_clip = fluid.dygraph_grad_clip.GradClipByGlobalNorm(hparams.max_grad_norm)
        else:
            self.grad_clip = None
        self.weight_decay = hparams.weight_decay
        self.optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=hparams.lr,
            regularization=fluid.regularizer.L2Decay(self.weight_decay))

        self._dtype = dtype

        # DataDistributed
        self.before_backward_fn = None
        self.after_backward_fn = None
        return

    def _create_parameters(self):
        """ Create model's paramters. """
        # self.lada_d = self.create_parameter(
        #        attr=fluid.ParamAttr(
        #            name="baba_d",
        #            initializer=fluid.initializer.NormalInitializer(scale=self.initializer_range)),
        #        shape=[1, 1, self.hidden_dim],
        #        dtype=self._dtype)
        if self.use_pointer_network == 0:
            #pass
            self.lada = layers.create_parameter(name='lambda_point',shape=[1],default_initializer=fluid.initializer.NumpyArrayInitializer(np.array([0.0])),dtype=self._dtype)
        
        
        if self.use_topic_trans:
            #self.ceshi =layers.create_parameter(name=self.full_name()+'transtor/FC_0.w_0',shape=[self.hidden_dim,1],default_initializer=fluid.initializer.NormalInitializer(scale=self.initializer_range),dtype=self._dtype)
            #self.ceshi3 =layers.create_parameter(name=self.full_name()+'transtor/FC_0.w_0@GRAD',shape=[self.hidden_dim,1],default_initializer=fluid.initializer.NormalInitializer(scale=self.initializer_range),dtype=self._dtype)
            #self.ceshi2 =layers.create_parameter(name=self.full_name()+'transtor/FC_0.b_0',shape=[1],default_initializer=fluid.initializer.NormalInitializer(scale=self.initializer_range),dtype=self._dtype)
            pass
        if self.num_latent > 0:
            self.mask_embed = self.create_parameter(
                attr=fluid.ParamAttr(
                    name="mask_embed",
                    initializer=fluid.initializer.NormalInitializer(scale=self.initializer_range)),
                shape=[1, 1, self.hidden_dim],
                dtype=self._dtype)
            self.latent_embeddings = self.create_parameter(
                attr=fluid.ParamAttr(
                    name="latent_embeddings",
                    initializer=fluid.initializer.NormalInitializer(scale=self.initializer_range)),
                shape=[self.num_latent, self.hidden_dim],
                dtype=self._dtype)
      
        sequence_mask = np.tri(self.num_pos_embeddings, self.num_pos_embeddings, dtype=self._dtype)#左下为1，右上为0
        self.sequence_mask = self.create_parameter(
            attr=fluid.ParamAttr(
                name="sequence_mask",
                initializer=fluid.initializer.NumpyArrayInitializer(sequence_mask), #用numpy初始化数据
                trainable=False),
            shape=sequence_mask.shape,
            dtype=sequence_mask.dtype)
        return

    def _load_params(self):
        """ Load saved paramters. """
        if self.init_checkpoint is not None:
            print(f"Loading parameters from {self.init_checkpoint}")
            if hasattr(fluid, "load_dygraph"):
                # >= 1.6.0 compatible
                models, optimizers = fluid.load_dygraph(self.init_checkpoint)
            else:
                models, optimizers = fluid.dygraph.load_persistables(self.init_checkpoint)
            parameters = {param.name: param for param in self.parameters()}
            #for pa in parameters:print(pa)
      
            #for name, param in models.items():
            #    print(name)
            for name, param in models.items():
                if name in parameters:
                    if param.shape != parameters[name].shape:
                        print(f"part of parameter({name}) random normlize initialize")
                        if hasattr(param, "numpy"):
                            arr = param.numpy()
                        else:
                            value = param.value()
                            tensor = value.get_tensor()
                            arr = np.array(tensor)
                        z = np.random.normal(scale=self.initializer_range,
                                             size=parameters[name].shape).astype("float32")
                        if name == "Model/UnifiedTransformer_0/Embedder_0/Embedding_0.w_0":
                            z[-param.shape[0]:] = arr
                        else:
                            z[:param.shape[0]] = arr
                        z = fluid.dygraph.to_variable(z)
                        models[name] = z
            for name in parameters:
                if name not in models:
                    if parameters[name].trainable:
                        print(f"parameter({name}) random normlize initialize")
                        z = np.random.normal(scale=self.initializer_range,
                                             size=parameters[name].shape).astype("float32")
                        models[name] = fluid.dygraph.to_variable(z)
                    else:
                        models[name] = parameters[name]
            if self.use_topic_trans and "Model/UnifiedTransformer_0.discriminator/FC_0.w_0" in models:
                name1 = self.full_name()+".transtor/FC_0.w_0"
                name2 = self.full_name()+".transtor/FC_0.b_0"
                models[name1] = models["Model/UnifiedTransformer_0.discriminator/FC_0.w_0"]
                models[name2] = models["Model/UnifiedTransformer_0.discriminator/FC_0.b_0"]
            self.load_dict(models)
            print(f"Loaded parameters from {self.init_checkpoint}")

    def _create_mask(self, input_mask, append_head=False, auto_regressive=False):# knwoledge部分在PGN上不用加mask
        """
        Create attention mask.

        @param : input_mask
        @type : Variable(shape: [batch_size, max_seq_len])

        @param : auto_regressive
        @type : bool
        """
        seq_len = input_mask.shape[1]

        input_mask = layers.cast(input_mask, self._dtype)
        mask1 = layers.expand(input_mask, [1, 1, seq_len])
        mask2 = layers.transpose(mask1, [0, 2, 1])
        mask = layers.elementwise_mul(mask1, mask2) #mask elemwise乘mask.T是把 相应应该mask掉的地方mask掉。可以自己举个例子。很简单。

        if append_head:
            mask = layers.concat([mask[:, :1, :], mask], axis=1)
            mask = layers.concat([mask[:, :, :1], mask], axis=2)# 句子上加入了latent_token
            seq_len += 1

        #print(seq_len)
        #print(mask.shape)
        #533
        #[2, 533, 533]

        if auto_regressive: #自回归是MLM任务吗,左下为1，右上为0，mask的位置目前是0表示，不被mask用1表示
            #print('autorege',seq_len)
            #print('autorege',mask.shape)
            #print('autorege',self.sequence_mask.shape)
            seq_mask = self.sequence_mask[:seq_len, :seq_len]
            #print('autorege',seq_mask.shape)
            #autorege 533
            #autorege [2, 533, 533]
            #autorege [512, 512]
            #autorege [512, 512]

            mask = layers.elementwise_mul(mask, seq_mask)

        mask = 1 - mask
        return mask #到这为止，mask位为1，正常为0

    def _join_mask(self, mask1, mask2): #合并mask，合并context部分的双向mask和tgt部分的的单向mask，目的是组成一个更大的包含两部分的mask矩阵,join的mask是从create部分来的
        """ Merge source attention mask and target attention mask.

        @param : mask1 : source attention mask
        @type : Variable(shape: [batch_size, max_src_len, max_src_len])

        @param : mask1 : target attention mask,难道这不是mask2吗
        @type : Variable(shape: [batch_size, max_tgt_len, max_tgt_len])
        """
        batch_size = mask1.shape[0] #
        seq_len1 = mask1.shape[1]
        seq_len2 = mask2.shape[1]
        seq_len = seq_len1 + seq_len2

        mask_lu = mask1 #[batch_size, seq_len1, seq_len1] left_upper
        mask_ru = layers.fill_constant([batch_size, seq_len1, seq_len2], self._dtype, 1) #[batch_size,seq_len1,seq_len2] right_upper，这时候mask值为1
        mask3 = layers.expand(mask2[:, :, :1], [1, 1, seq_len1]) #[batch_size,seq_len2,seq_len1] mask2的最后一个维度扩展到seq_len1的维度 
        mask4 = layers.expand(mask1[:, :1], [1, seq_len2, 1]) #[batch_size,seq_len2,seq_len1]
        mask_lb = mask3 + mask4 - mask3 * mask4 #[batch_size,seq_len2,seq_len1] left_bottem
        mask_rb = mask2 #[batch_size, seq_len2, seq_len2] right_bottom
        mask_u = layers.concat([mask_lu, mask_ru], axis=2)#[batch_size,seq_len1,seq_len1+seq_len2] upper
        mask_b = layers.concat([mask_lb, mask_rb], axis=2) #[batch_size,seq_len2,seq_len1+seq_len2] bottom
        mask = layers.concat([mask_u, mask_b], axis=1) #[batch_size,seq_len1+seq_len2,seq_len1+seq_len2]
        return mask

    def _posteriori_network(self, input_mask, embed, batch_size, src_len, tgt_len):
        """ Basic posteriori network implement. """
        mask_embed = self.mask_embed#+self.lada_d
        mask_embed = layers.expand(mask_embed, [batch_size, 1, 1])
        mask_embed = self.embed_layer_norm(mask_embed)
        post_embed = layers.concat([mask_embed, embed], axis=1)

        mask = self._create_mask(input_mask, auto_regressive=not self.bidirectional_context,
                                 append_head=True)

        for layer in self.layers:
            post_embed = layer(post_embed, mask, None)

        post_embed = post_embed[:, 0]
        post_logits = self.post_network(post_embed)
        post_probs = layers.softmax(post_logits, axis=-1)
        post_logits = layers.log(post_probs)
        return post_embed, post_probs, post_logits

    def _discriminator_network(self, input_mask, embed, batch_size, src_len, tgt_len, pos_embed):
        """ Basic discriminator network implement. """
        # if batch_size <= 1:
        #     raise ValueError("Warmming: If you use discriminator loss in traning, the batch_size must be greater than 1.")

        src_embed = embed[:, :src_len]
        tgt_embed = embed[:, src_len:]
        if batch_size > 1:
            neg_tgt_embed = layers.concat([tgt_embed[1:], tgt_embed[:1]], axis=0)#相当于把label以循环列表的方式往后移动了一位。
        else:
            # Cannot train discriminator if batch_size == 1
            neg_tgt_embed = tgt_embed #这块可以修改，随机sample呢？
        neg_embed = layers.concat([src_embed, neg_tgt_embed], axis=1)

        # Create generation network mask
        src_mask = input_mask[:, :src_len]
        tgt_mask = input_mask[:, src_len:]
        if batch_size > 1:
            neg_tgt_mask = layers.concat([tgt_mask[1:], tgt_mask[:1]], axis=0)
        else:
            # Cannot train discriminator if batch_size == 1
            neg_tgt_mask = tgt_mask
        neg_mask = layers.concat([src_mask, neg_tgt_mask], axis=1)
        mask = self._create_mask(neg_mask, auto_regressive=not self.bidirectional_context,
                                 append_head=True)

        mask_embed = self.mask_embed
        mask_embed = layers.expand(mask_embed, [batch_size, 1, 1])
        mask_embed = self.embed_layer_norm(mask_embed)
        neg_embed= layers.concat([mask_embed, neg_embed], axis=1)

        for layer in self.layers:
            neg_embed = layer(neg_embed, mask, None)

        neg_embed = neg_embed[:, 0]

        pos_probs = self.discriminator(pos_embed)
        neg_probs = self.discriminator(neg_embed)

        return pos_probs, neg_probs

    def _transfer_network(self,postive_mask,negative_mask,postive_embed,negative_embed,batch_size):
        """ Basic transfer network implement. """
        mask_embed = self.mask_embed
        mask_embed = layers.expand(mask_embed, [batch_size, 1, 1])
        mask_embed = self.embed_layer_norm(mask_embed)
        post_embed = layers.concat([mask_embed, postive_embed], axis=1)
        negative_embed = layers.concat([mask_embed,negative_embed],axis=1)

        mask_postive = self._create_mask(postive_mask, auto_regressive=False,
                                 append_head=True) 
        mask_negative =self._create_mask(negative_mask, auto_regressive=False,
                                 append_head=True)
        #print("size-----------",mask_postive.shape,post_embed.shape)
        #print("size-----------",mask_negative.shape,negative_embed.shape)

        for layer in self.layers:
            pos_embed = layer(post_embed,mask_postive,None)
        
        pos_embed = pos_embed[:,0]
        pos_probs = self.transtor(pos_embed)
        for layer in self.layers: 
            neg_embed = layer(negative_embed, mask_negative, None) 
        
        neg_embed = neg_embed[:,0]

        
        neg_probs = self.transtor(neg_embed) 
        #print("pos_prob:",pos_probs,"nega:",neg_probs)
        return pos_probs,neg_probs
    def _generation_network(self, input_mask, embed, batch_size, src_len, tgt_len, latent_embed,knw_len,src_type,src_token,tgt_token):
        """ Basic generation network implement. """
        #print(embed.shape)
        #print(knw_len)
        #knw_len = knw_len.numpy()[0]
        if self.num_latent > 0:
            latent_embed = F.unsqueeze(latent_embed, [1])#btach_size,1,hidenn_embedding
            latent_embed = self.embed_layer_norm(latent_embed)
            dec_embed = layers.concat([latent_embed, embed], axis=1)
        else:
            dec_embed = embed

        # Create generation network mask
        src_mask = input_mask[:, :src_len] #分别拿到context的mask
        
        tgt_mask = input_mask[:, src_len:] #拿到tgt的mask
        enc_mask = self._create_mask(src_mask, auto_regressive=not self.bidirectional_context,
                                     append_head=self.num_latent > 0)
        dec_mask = self._create_mask(tgt_mask, auto_regressive=True)
        mask = self._join_mask(enc_mask, dec_mask)

        for layer in self.layers:
            dec_embed = layer(dec_embed, mask, None)

        if self.num_latent > 0:
            latent_embed = dec_embed[:, 0]
        else:
            latent_embed = None
        
        context_embed_after_att = dec_embed[:,:-tgt_len]
        if self.num_latent>0:
            context_embed_after_att = context_embed_after_att[:,1:]

        dec_embed = dec_embed[:, -tgt_len:]
        
        if self.two_layer_predictor:
            dec_embed = self.pre_predictor(dec_embed)
        if self.weight_sharing:             #把token_embedding 换成knoledge_embedding 即为指针网络。
            token_embedding = self.embedder.token_embedding.weight
            dec_logits = layers.matmul(
                x=dec_embed,
                y=token_embedding,
                transpose_y=True
            )
        else:
            dec_logits = self.predictor(dec_embed)

        dec_probs = layers.softmax(dec_logits, axis=-1)
        #knw_len=None
        #src_token = None
        #src_type = None
        # if self.use_pointer_network:
            
        #     knowledge_embed = dec_embed[:,:-tgt_len]
        #     #print(knowledge_embed.shape)
        #     knowledge_embed = knowledge_embed[:,-knw_len:]
        #     know_token = src_token[:,-knw_len:]
        #     #print("kwn_len",knw_len)
        #     #print("src_type",src_type.shape)
        #     typs_kno = src_type[:,:src_len][:,-knw_len:].numpy().astype('float32')
        #     #print("shape_types_kno",typs_kno.shape)
        #     typs_kno[typs_kno!=2] = 0
        #     typs_kno[typs_kno==2] = 1
        #     typs_kno2 = src_type[:,:src_len][:,-knw_len:].numpy().astype('float32')
        #     typs_kno2[typs_kno!=2] = -1e10
        #     typs_kno2[typs_kno==2] = 0
        #     typs_kno = fluid.dygraph.to_variable(typs_kno)
        #     typs_kno2 = fluid.dygraph.to_variable(typs_kno2)
        #     typs_kno = layers.squeeze(input=typs_kno, axes=[2])
        #     typs_kno2 = layers.squeeze(input=typs_kno2, axes=[2])
        #     typs_kno = F.unsqueeze(typs_kno, [1])
        #     typs_kno = layers.expand(typs_kno, [1, dec_embed.shape[1], 1])
        #     typs_kno2 = F.unsqueeze(typs_kno2, [1])
        #     typs_kno2 = layers.expand(typs_kno2, [1, dec_embed.shape[1], 1])
        #     typs_kno.stop_gradient =True
        #     typs_kno2.stop_gradient =True
        #     #print("dec_emb",dec_embed.shape)
        #     #print("knowel_emd",knowledge_embed.shape)
        #     pointer_logits = layers.matmul(
        #         x = dec_embed,
        #         y = knowledge_embed,
        #         transpose_y=True
        #     )
        #     pointer_logits = layers.elementwise_mul(pointer_logits, typs_kno)#x = [1, 15, 230],y=[1,230]
        #     pointer_logits = layers.elementwise_add(pointer_logits, typs_kno2)
        #     pointer_probs = layers.softmax(pointer_logits,axis=-1)
        #     pointer_probs = layers.elementwise_mul(pointer_probs, typs_kno)
            
        #     know_onehot = layers.one_hot(know_token, self.num_token_embeddings)#batch_size,know_len,vocab_size
        #     know_onehot.stop_gradient =True
        #     lada = fluid.layers.sigmoid(self.lada)
        #     pointer_probs = layers.matmul(x=pointer_probs,y = know_onehot)#X's shape: [1, 15, 230], Y's shape: [1, 175, 30522]
        #     #print("prbos,pointer_probs",dec_probs.shape,pointer_probs.shape)
        #     dec_probs = layers.elementwise_add(x=self.lada[0]*dec_probs,y=(1-self.lada[1])*pointer_probs) #auto
        #     dec_probs = layers.elementwise_add(x=dec_probs*lada,y=pointer_probs*(1-lada)) #auto
        #     #dec_probs = layers.elementwise_add(x=dec_probs,y=pointer_probs) #55
        #     #self.lada = self.lada/layers.reduce_sum(self.lada) 
        if self.use_pointer_network==0:
            '''
            把指针指向knowledge改为指向整个context。
            '''
            #knowledge_embed = dec_embed[:,:-tgt_len]
            context_embed = embed[:,:-tgt_len]
           
            
            #print(knowledge_embed.shape)
            context_token = src_token.numpy() #batchsize seq_len 1
            #print(context_token.shape) #batchsize seq_len 1
            #print("kwn_len",knw_len)
            #print("src_type",src_type.shape) #batchsize seq_len 1
            ###
            ### target_mean
            ###
            typs_context = src_type.numpy().astype('float32')
            #print("shape_types_kno",typs_kno.shape)
            typs_context[context_token==0] = 0 #batch_size,seq_len,1
            typs_context[context_token!=0] = 1
            
            typs_context2 = src_type.numpy().astype('float32')
            typs_context2[context_token==0] = -1e10
            typs_context2[context_token!=0] = 0

            typs_context = fluid.dygraph.to_variable(typs_context)
            typs_context2 = fluid.dygraph.to_variable(typs_context2)
            typs_context = layers.squeeze(input=typs_context, axes=[2])
            typs_context2 = layers.squeeze(input=typs_context2, axes=[2])
            typs_context = F.unsqueeze(typs_context, [1])
            typs_context = layers.expand(typs_context, [1, dec_embed.shape[1], 1])
            typs_context2 = F.unsqueeze(typs_context2, [1])
            typs_context2 = layers.expand(typs_context2, [1, dec_embed.shape[1], 1])
            typs_context.stop_gradient =True
            typs_context2.stop_gradient =True
         
            
            
            pointer_logits = layers.matmul(
                x = dec_embed,
                y = context_embed,
                transpose_y=True
            )
            pointer_logits = layers.elementwise_mul(pointer_logits, typs_context)#x = [1, 15, 230],y=[1,15,230]
            pointer_logits = layers.elementwise_add(pointer_logits, typs_context2)
            pointer_probs = layers.softmax(pointer_logits,axis=-1)
            pointer_probs = layers.elementwise_mul(pointer_probs, typs_context)
            context_token = fluid.dygraph.to_variable(context_token) 
            know_onehot = layers.one_hot(context_token, self.num_token_embeddings)#batch_size,know_len,vocab_size
            know_onehot.stop_gradient =True
            lada = fluid.layers.sigmoid(self.lada)
            pointer_probs = layers.matmul(x=pointer_probs,y = know_onehot)#X's shape: [1, 15, 230], Y's shape: [1, 175, 30522]
            #print("prbos,pointer_probs",dec_probs.shape,pointer_probs.shape)
            #dec_probs = layers.elementwise_add(x=self.lada[0]*dec_probs,y=(1-self.lada[1])*pointer_probs) #auto
            dec_probs = layers.elementwise_add(x=dec_probs*lada,y=pointer_probs*(1-lada)) #auto
            #print(att.shape)
            #print(dec_probs.shape)
            #dec_probs = layers.elementwise_add(x=dec_probs,y=pointer_probs) #55
            #self.lada = self.lada/layers.reduce_sum(self.lada)

        if self.use_pointer_network==1:
            '''
            和上面的 0 相比，它的pointer loss的权重计算方式发生了改变。
            '''
            #knowledge_embed = dec_embed[:,:-tgt_len]
            context_embed = embed[:,:-tgt_len]
           
            
            #print(knowledge_embed.shape)
            context_token = src_token.numpy() #batchsize seq_len 1
           
            typs_context = src_type.numpy().astype('float32')

            
            #print("shape_types_kno",typs_kno.shape)
            typs_context[context_token==0] = 0 #batch_size,seq_len,1
            typs_context[context_token!=0] = 1
            
            typs_context2 = src_type.numpy().astype('float32')
            typs_context2[context_token==0] = -1e10
            typs_context2[context_token!=0] = 0
            #print("ceshicopy",typs_context==typs_context2)
            
            typs_context = fluid.dygraph.to_variable(typs_context)
            typs_context2 = fluid.dygraph.to_variable(typs_context2)
            #print(typs_context.numpy())
            #print(typs_context2.numpy())
            context_padding = layers.expand(typs_context,[1,1,self.hidden_dim]) # lamda_att
            typs_context = layers.squeeze(input=typs_context, axes=[2])
            #
            #context_mean
            #
            context_len = layers.reduce_sum(typs_context,dim = 1,keep_dim=True) # batch_size,1 # lamda_att
            #print("context_len_shape",context_len.shape)
            #print("context_len",context_len.numpy())
            context_len = F.unsqueeze(context_len,[1])
            typs_context2 = layers.squeeze(input=typs_context2, axes=[2])
            typs_context = F.unsqueeze(typs_context, [1])
            typs_context = layers.expand(typs_context, [1, dec_embed.shape[1], 1])
            typs_context2 = F.unsqueeze(typs_context2, [1])
            typs_context2 = layers.expand(typs_context2, [1, dec_embed.shape[1], 1])
            typs_context.stop_gradient =True
            typs_context2.stop_gradient =True
            context_len.stop_gradient = True #lamda_att
            context_padding.stop_gradient = True # lamda_att
            context_embed_notpad = layers.elementwise_mul(context_embed,context_padding) #lamda_att
            
            context_embed_mean = layers.elementwise_div(layers.reduce_sum(context_embed_notpad,dim=1,keep_dim=True),context_len)#batch_size,1, dim #lambda_att
            
            att = layers.matmul(
                x = dec_embed,
                y = context_embed_mean,
                transpose_y=True
            )*self.hidden_dim**-0.5
            
            att = fluid.layers.sigmoid(att)
           
            pointer_logits = layers.matmul(
                x = dec_embed,
                y = context_embed,
                transpose_y=True
            )
            #print("pointer_logits_shape",pointer_logits.shape)
            pointer_logits = layers.elementwise_mul(pointer_logits, typs_context)#x = [1, 15, 230],y=[1,15,230]
            pointer_logits = layers.elementwise_add(pointer_logits, typs_context2)
            pointer_probs = layers.softmax(pointer_logits,axis=-1)
            pointer_probs = layers.elementwise_mul(pointer_probs, typs_context)
            context_token = fluid.dygraph.to_variable(context_token) 
            know_onehot = layers.one_hot(context_token, self.num_token_embeddings)#batch_size,know_len,vocab_size
            know_onehot.stop_gradient =True
            
            pointer_probs = layers.matmul(x=pointer_probs,y = know_onehot)#X's shape: [1, 15, 230], Y's shape: [1, 175, 30522]
            
            dec_probs = layers.elementwise_add(x=dec_probs*(1-att),y=pointer_probs*att) #auto
            


        if self.use_pointer_network==2:
            '''
            和上面的 1相比，将原先对token进行attention改为对经过tranformer的。
            '''
            #knowledge_embed = dec_embed[:,:-tgt_len]
            context_embed = embed[:,:-tgt_len]
            #print(context_embed.shape)
            #print(context_embed_after_att.shape)
            
            #print(knowledge_embed.shape)
            context_token = src_token.numpy() #batchsize seq_len 1
           
            typs_context = src_type.numpy().astype('float32')
           
            typs_context[context_token==0] = 0 #batch_size,seq_len,1
            typs_context[context_token!=0] = 1
            
            typs_context2 = src_type.numpy().astype('float32')
            typs_context2[context_token==0] = -1e10
            typs_context2[context_token!=0] = 0
            #print("ceshicopy",typs_context==typs_context2)
            
            typs_context = fluid.dygraph.to_variable(typs_context)
            typs_context2 = fluid.dygraph.to_variable(typs_context2)
            #print(typs_context.numpy())
            #print(typs_context2.numpy())
            context_padding = layers.expand(typs_context,[1,1,self.hidden_dim]) # lamda_att
            typs_context = layers.squeeze(input=typs_context, axes=[2])
            #
            #context_mean
            #
            context_len = layers.reduce_sum(typs_context,dim = 1,keep_dim=True) # batch_size,1 # lamda_att
            #print("context_len_shape",context_len.shape)
            #print("context_len",context_len.numpy())
            context_len = F.unsqueeze(context_len,[1])
            typs_context2 = layers.squeeze(input=typs_context2, axes=[2])
            typs_context = F.unsqueeze(typs_context, [1])
            typs_context = layers.expand(typs_context, [1, dec_embed.shape[1], 1])
            typs_context2 = F.unsqueeze(typs_context2, [1])
            typs_context2 = layers.expand(typs_context2, [1, dec_embed.shape[1], 1])
            typs_context.stop_gradient =True
            typs_context2.stop_gradient =True
            context_len.stop_gradient = True #lamda_att
            context_padding.stop_gradient = True # lamda_att
            #context_embed_notpad = layers.elementwise_mul(context_embed,context_padding) #lamda_att
            context_embed_notpad = layers.elementwise_mul(context_embed_after_att,context_padding) #lamda_att
      
            context_embed_mean = layers.elementwise_div(layers.reduce_sum(context_embed_notpad,dim=1,keep_dim=True),context_len)#batch_size,1, dim #lambda_att
          
          
            
            att = layers.matmul(
                x = dec_embed,
                y = context_embed_mean,
                transpose_y=True
            )*self.hidden_dim**-1
           
            att = fluid.layers.sigmoid(att)
          
            pointer_logits = layers.matmul(
                x = dec_embed,
                y = context_embed,
                transpose_y=True
            )
            pointer_logits = layers.elementwise_mul(pointer_logits, typs_context)#x = [1, 15, 230],y=[1,15,230]
            pointer_logits = layers.elementwise_add(pointer_logits, typs_context2)
            pointer_probs = layers.softmax(pointer_logits,axis=-1)
            pointer_probs = layers.elementwise_mul(pointer_probs, typs_context)
            context_token = fluid.dygraph.to_variable(context_token) 
            know_onehot = layers.one_hot(context_token, self.num_token_embeddings)#batch_size,know_len,vocab_size
            know_onehot.stop_gradient =True
            #lada = fluid.layers.sigmoid(self.lada)
            pointer_probs = layers.matmul(x=pointer_probs,y = know_onehot)#X's shape: [1, 15, 230], Y's shape: [1, 175, 30522]
            
            dec_probs = layers.elementwise_add(x=dec_probs*(1-att),y=pointer_probs*att) #auto 


        if self.use_pointer_network==3:
            '''
            和上面的 2相比，加入了覆盖机制防止出现重复的词引用。
            '''
            
            context_embed = embed[:,:-tgt_len]
         
            context_token = src_token.numpy() #batchsize seq_len 1
           

            ###
            ### target_mean
            ###
            typs_context = src_type.numpy().astype('float32')
           

            typs_context[context_token==0] = 0 #batch_size,seq_len,1
            typs_context[context_token!=0] = 1
            
            typs_context2 = src_type.numpy().astype('float32')
            typs_context2[context_token==0] = -1e10
            typs_context2[context_token!=0] = 0
           
           
            
            typs_context = fluid.dygraph.to_variable(typs_context)
            typs_context2 = fluid.dygraph.to_variable(typs_context2)
          
            context_padding = layers.expand(typs_context,[1,1,self.hidden_dim]) # lamda_att
            typs_context = layers.squeeze(input=typs_context, axes=[2])
            #
            #context_mean
            #
            context_len = layers.reduce_sum(typs_context,dim = 1,keep_dim=True) # batch_size,1 # lamda_att
            #print("context_len_shape",context_len.shape)
            #print("context_len",context_len.numpy())
            context_len = F.unsqueeze(context_len,[1])
            typs_context2 = layers.squeeze(input=typs_context2, axes=[2])
            typs_context = F.unsqueeze(typs_context, [1])
            typs_context = layers.expand(typs_context, [1, dec_embed.shape[1], 1])
            typs_context2 = F.unsqueeze(typs_context2, [1])
            typs_context2 = layers.expand(typs_context2, [1, dec_embed.shape[1], 1])
            typs_context.stop_gradient =True
            typs_context2.stop_gradient =True
            context_len.stop_gradient = True #lamda_att
            context_padding.stop_gradient = True # lamda_att
            
            context_embed_notpad = layers.elementwise_mul(context_embed_after_att,context_padding) #lamda_att
            
            context_embed_mean = layers.elementwise_div(layers.reduce_sum(context_embed_notpad,dim=1,keep_dim=True),context_len)#batch_size,1, dim #lambda_att
            
            
            
            att = layers.matmul(
                x = dec_embed,
                y = context_embed_mean,
                transpose_y=True
            )*self.hidden_dim**-1
           

            att = fluid.layers.sigmoid(att)
            

            pointer_logits = layers.matmul(
                x = dec_embed,
                y = context_embed,
                transpose_y=True
            )
            pointer_logits = layers.elementwise_mul(pointer_logits, typs_context)#x = [1, 15, 230],y=[1,15,230]
            pointer_logits = layers.elementwise_add(pointer_logits, typs_context2)
            
            for i in range(0,pointer_logits.shape[0]):
                temp = 0
                for j in range(0,pointer_logits,shape[1]):
                    pointer_logits[i][j] = pointer_logits[i][j]+temp
                    temp = temp+ pointer_logits[i][j]
            pointer_probs = layers.softmax(pointer_logits,axis=-1)
            
            
            pointer_probs = layers.elementwise_mul(pointer_probs, typs_context)
            context_token = fluid.dygraph.to_variable(context_token) 
            know_onehot = layers.one_hot(context_token, self.num_token_embeddings)#batch_size,know_len,vocab_size
            know_onehot.stop_gradient =True
            
            pointer_probs = layers.matmul(x=pointer_probs,y = know_onehot)#X's shape: [1, 15, 230], Y's shape: [1, 175, 30522]
            
            
            dec_probs = layers.elementwise_add(x=dec_probs*(1-att),y=pointer_probs*att) #auto 




        return latent_embed, dec_probs

    def _forward(self, src_token,src_mask,src_pos,src_type,src_turn,k_max_len,postive_token=None,
        postive_token_pos=None, postive_type=None,postive_turn=None,postive_mask=None,negative_token=None,negative_token_pos=None,
        negative_type=None,negative_turn=None,negative_mask=None,tgt_token=None,tgt_mask=None,tgt_pos=None,tgt_type=None,tgt_turn=None,is_training=False):
        """ Real forward process of model in different mode(train/test). """
        outputs = {}
        #if k_max_len !=None:
        knw_max_len = k_max_len
        #else:
        #    knw_max_len = 0
        

        # src_token = inputs["src_token"]
        # src_mask = inputs["src_mask"]
        # src_pos = inputs["src_pos"]
        # src_type = inputs["src_type"]
        # src_turn = inputs["src_turn"]
        #print("src_typppppp",src_type.shape)
        #print(src_type.numpy())
        tgt_token =tgt_token[:, :-1]
        tgt_mask = tgt_mask[:, :-1]
        tgt_pos =  tgt_pos[:, :-1]
        tgt_type = tgt_type[:, :-1]
        tgt_turn = tgt_turn[:, :-1]


        #print("src_mask_transformer:",src_mask.shape)

        input_mask = layers.concat([src_mask, tgt_mask], axis=1)
        input_mask.stop_gradient = True
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
        embed = layers.concat([src_embed, tgt_embed], axis=1)
        embed = self.embed_layer_norm(embed)

        if self.use_topic_trans :
            postive_token = postive_token
            postive_pos = postive_token_pos
            postive_type = postive_type
            postive_turn = postive_turn
            postive_mask = postive_mask

            negative_token =negative_token
            negative_pos = negative_token_pos
            negative_type = negative_type
            negative_turn = negative_turn
            negative_mask = negative_mask

            postive_embed = self.embedder(postive_token,postive_pos,postive_type,postive_turn)
            negative_embed = self.embedder(negative_token,negative_pos,negative_type,negative_turn)
            postive_input_mask = layers.concat([src_mask,postive_mask],axis=1)
            postive_input_mask.stop_gradient = True
            negative_input_mask = layers.concat([src_mask,negative_mask],axis=1)
            negative_input_mask.stop_gradient = True

            postive_embed = layers.concat([src_embed, postive_embed], axis=1)
            postive_embed = self.embed_layer_norm(postive_embed)

            negative_embed = layers.concat([src_embed,negative_embed],axis=1)
            negative_embed = self.embed_layer_norm(negative_embed)


        batch_size = src_token.shape[0]
        src_len = src_token.shape[1]
        tgt_len = tgt_token.shape[1]
        
        if self.use_topic_trans:
            tran_positive_probs,trans_negative_probs = self._transfer_network(
                postive_input_mask,negative_input_mask,postive_embed,negative_embed,batch_size
            )

            outputs["trans_positive_probs"] = tran_positive_probs
            outputs["trans_negative_probs"] = trans_negative_probs


        if self.num_latent > 0:
            post_embed, post_probs, post_logits = self._posteriori_network(
                input_mask, embed, batch_size, src_len, tgt_len)
            outputs["post_logits"] = post_logits

            if self.use_discriminator:
                pos_probs, neg_probs = self._discriminator_network(
                    input_mask, embed, batch_size, src_len, tgt_len, post_embed)
                outputs["pos_probs"] = pos_probs
                outputs["neg_probs"] = neg_probs

            if is_training:
                z = F.gumbel_softmax(post_logits, self.tau) #gumbel_softmax的作用
            else:
                indices = layers.argmax(post_logits, axis=1)
                z = layers.one_hot(F.unsqueeze(indices, [1]), self.num_latent)
            latent_embeddings = self.latent_embeddings #[latend_num,hidden_size]
            latent_embed = layers.matmul(z, latent_embeddings) #[batch_size,hidden_size]
            outputs["latent_embed"] = latent_embed
        else:
            latent_embed = None

        


        latent_embed, dec_probs = self._generation_network(
            input_mask, embed, batch_size, src_len, tgt_len, latent_embed,knw_max_len,src_type,src_token,tgt_token)
        outputs["dec_probs"] = dec_probs
       
           
        if self.num_latent > 0 and self.with_bow: 
            if self.two_layer_predictor:
                latent_embed = self.pre_bow_predictor(latent_embed)
            bow_logits = self.bow_predictor(latent_embed)
            bow_probs = layers.softmax(bow_logits)
            outputs["bow_probs"] = bow_probs

        return outputs

    

    def _collect_metrics(self, inputs, outputs): #loss计算
        """ Calculate loss function by using inputs and outputs. """
        metrics = {}

        tgt_len = layers.reduce_sum(layers.reduce_sum(inputs["tgt_mask"], dim=1) - 1)
        tgt_len.stop_gradient = True
        # generator loss
        label = inputs["tgt_token"][:, 1:]
        if self.label_smooth > 0:
            one_hot_label = layers.one_hot(label, self.num_token_embeddings)
            smooth_label = layers.label_smooth(one_hot_label, epsilon=self.label_smooth,
                                               dtype=self._dtype)
            nll = layers.cross_entropy(outputs["dec_pred"], smooth_label, soft_label=True,
                                       ignore_index=self.padding_idx)
        else:
            nll = layers.cross_entropy(outputs["dec_probs"], label, ignore_index=self.padding_idx)
        nll = layers.reduce_sum(nll, dim=1)
        token_nll = layers.reduce_sum(nll) / tgt_len
        nll = layers.reduce_mean(nll)
        metrics["nll"] = nll
        metrics["token_nll"] = token_nll
        loss = nll

        # pointer losee


        if self.num_latent > 0 and self.with_bow:
            bow_probs = F.unsqueeze(outputs["bow_probs"], [1])#[batch_size,vocab_size] ,label[batch_size,seq_len]
            bow_probs = layers.expand(bow_probs, [1, label.shape[1], 1])#[batch_size,sen_len,vocab_size]
            if self.label_smooth > 0:
                bow = layers.cross_entropy(bow_probs, smooth_label, soft_label=True,
                                           ignore_index=self.padding_idx)
            else:
                bow = layers.cross_entropy(bow_probs, label, ignore_index=self.padding_idx)
            bow = layers.reduce_sum(bow, dim=1)
            token_bow = layers.reduce_sum(bow) / tgt_len
            bow = layers.reduce_mean(bow)
            metrics["bow"] = bow
            metrics["token_bow"] = token_bow
            loss = loss + bow

        if self.num_latent > 0 and self.use_discriminator:
            dis = 0.0 - (layers.log(outputs["pos_probs"]) + layers.log(1.0 - outputs["neg_probs"]))
            dis = layers.reduce_mean(dis)
            metrics["dis"] = dis
            loss = loss + dis * self.dis_ratio
        if self.use_topic_trans:
            trans = 0.0 - (layers.log(outputs["trans_positive_probs"]) + layers.log(1.0 - outputs["trans_negative_probs"]))
            trans = layers.reduce_mean(trans)
            metrics["trans"] = trans
            loss = loss+ trans
        metrics["loss"] = loss
        metrics["token_num"] = tgt_len
        return metrics

    def _optimize(self, loss):
        """ Optimize loss function and update model. """
        if self.before_backward_fn is not None:
            loss = self.before_backward_fn(loss)
        loss.backward()
        if self.after_backward_fn is not None:
            self.after_backward_fn()
        self.optimizer.minimize(loss,
                                grad_clip=self.grad_clip,
                                parameter_list=self.parameters())
        self.clear_gradients()
        return

    def _init_state(self, inputs):
        """ Initialize decode state. """
        state = {}
        
        #初始化状态
        src_token = inputs["src_token"]
        src_mask = inputs["src_mask"]
        src_pos = inputs["src_pos"]
        src_type = inputs["src_type"]
        src_turn = inputs["src_turn"]

        batch_size = src_token.shape[0]
        seq_len = src_token.shape[1]

        tmp_src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        
        src_embed = fluid.dygraph.to_variable(tmp_src_embed.numpy())

        src_embed = self.embed_layer_norm(src_embed)
        
        #创造mask矩阵
        mask = self._create_mask(src_mask, append_head=self.num_latent > 0,auto_regressive=not self.bidirectional_context)
        mask2 = self._create_mask(src_mask, append_head=self.num_latent > 0,auto_regressive=False)

        if self.num_latent > 0:
            src_embed = F.unsqueeze(src_embed, [1]) #[batch_size,1,seq_len,hidden_size]
            src_embed = layers.expand(src_embed, [1, self.num_latent, 1, 1])
            src_embed = layers.reshape(src_embed, [-1, seq_len, self.hidden_dim])#[batch_size*num_latent,seq_len,hidden_size]

            latent_embed = self.latent_embeddings #[num_latent,hdd_size]
            latent_embed = F.unsqueeze(latent_embed, [1])#[num_latent,1,hidden_size]
            latent_embed = layers.expand(latent_embed, [batch_size, 1, 1]) #[batch_size*num_latent,1,hidden_size]
            latent_embed = self.embed_layer_norm(latent_embed)

            enc_out = layers.concat([latent_embed, src_embed], axis=1)

            mask = F.unsqueeze(mask, [1]) # batch_size,1,seq_len,seq_len
            mask = layers.expand(mask, [1, self.num_latent, 1, 1]) # batch_size,num_latent,seq_len+1,seq_len+1
            mask = layers.reshape(mask, [-1, seq_len + 1, seq_len + 1])# batch_size*num_latent,seq_len+1,seq_len+1
            mask2 = F.unsqueeze(mask2, [1]) # batch_size,1,seq_len,seq_len
            mask2 = layers.expand(mask2, [1, self.num_latent, 1, 1]) # batch_size,num_latent,seq_len+1,seq_len+1
            mask2 = layers.reshape(mask2, [-1, seq_len + 1, seq_len + 1])# batch_size*num_latent,seq_len+1,seq_len+1
            if self.use_pointer_network >= 0:
                tmp_src_embed = F.unsqueeze(tmp_src_embed,[1])
                tmp_src_embed = layers.expand(tmp_src_embed,[1,self.num_latent,1,1])
                tmp_src_embed = layers.reshape(tmp_src_embed,[batch_size*self.num_latent,-1,self.hidden_dim])

                src_token_tmp = F.unsqueeze(src_token,[1])
                #print(src_token_tmp.shape)
                src_token_tmp = layers.expand(src_token_tmp,[1,self.num_latent,1,1])
                src_token_tmp  = layers.reshape(src_token_tmp ,[batch_size*self.num_latent,-1,1])

                src_type_tmp = F.unsqueeze(src_type,[1])
                src_type_tmp= layers.expand(src_type_tmp,[1,self.num_latent,1,1])
                src_type_tmp = layers.reshape(src_type_tmp,[batch_size*self.num_latent,-1,1])

                state["src_embed"] = tmp_src_embed
                state["src_token"] = src_token_tmp
                state["src_type"] = src_type_tmp

                # state["src_embed"] = tmp_src_embed
                # state["src_token"] = src_token#_tmp
                # state["src_type"] = src_type#_tmp
        else:
            if self.use_pointer_network >= 0:
                state["src_embed"] = tmp_src_embed
                state["src_token"] = src_token
                state["src_type"] = src_type
            enc_out = src_embed

        cache = {}
        for l, layer in enumerate(self.layers):
            cache[f"layer_{l}"] = {}
            enc_out = layer(enc_out, mask, cache[f"layer_{l}"])

        
        state["context_att_emd"] = enc_out[:,1:]
        state["cache"] = cache
        state["mask"] = mask2[:, :1] # 选取了第一个latent位置的mask。batch_size*num_latent,1,seq_len+1
        if self.num_latent > 0:
            state["batch_size"] = batch_size * self.num_latent
            shape = [batch_size * self.num_latent, 1, 1] # shape有两个1维度为什么？
        else:
            state["batch_size"] = batch_size
            shape = [batch_size, 1, 1]
        state["pred_mask"] = layers.ones(shape, self._dtype)
        state["pred_pos"] = layers.zeros(shape, "int64")
        state["pred_type"] = layers.zeros(shape, "int64")
        state["pred_turn"] = layers.zeros(shape, "int64")

        if "tgt_token" in inputs and self.num_latent > 0:
            tgt_token = inputs["tgt_token"][:, :-1]
            tgt_mask = inputs["tgt_mask"][:, :-1]
            tgt_pos = inputs["tgt_pos"][:, :-1]
            tgt_type = inputs["tgt_type"][:, :-1]
            tgt_turn = inputs["tgt_turn"][:, :-1]

            input_mask = layers.concat([src_mask, tgt_mask], axis=1)
            input_mask.stop_gradient = True
            src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
            tgt_embed = self.embedder(tgt_token, tgt_pos, tgt_type, tgt_turn)
            embed = layers.concat([src_embed, tgt_embed], axis=1)
            embed = self.embed_layer_norm(embed)

            batch_size = src_token.shape[0]
            src_len = src_token.shape[1]
            tgt_len = tgt_token.shape[1]

            post_embed, post_probs, post_logits = self._posteriori_network(
                input_mask, embed, batch_size, src_len, tgt_len)
            state["post_probs"] = post_probs

        return state

    def _decode(self, state):
        """ Decoding one time stamp. """
        # shape: [batch_size, 1, seq_len]
        # if beamsearch [batch_size*beam_size,1,seq_len]
        mask = state["mask"]
        # print("mask_shape",mask.shape[0])
        # print("batch_size",state["batch_size"])
        # print("num_latent",self.num_latent)
        beam_size = mask.shape[0]//(state["batch_size"])
        # shape: [batch_size, 1]
        #print("mask",mask.shape) #mask [100, 1, 147]
        pred_token = state["pred_token"]
        #print("pred_token",pred_token.shape) #pred_token [100, 1, 1]
        pred_mask = state["pred_mask"]
        pred_pos = state["pred_pos"]
        pred_type = state["pred_type"]
        pred_turn = state["pred_turn"]
        
        # list of shape(len: num_layers): [batch_size, seq_len, hidden_dim]
        cache = state["cache"]

        pred_embed = self.embedder(pred_token, pred_pos, pred_type, pred_turn)
        pred_embed = self.embed_layer_norm(pred_embed)

        # shape: [batch_size, 1, seq_len + 1]
        mask = layers.concat([mask, 1 - pred_mask], axis=2)

        # shape: [batch_size, 1, hidden_dim]
        for l, layer in enumerate(self.layers):
            pred_embed = layer(pred_embed, mask, cache[f"layer_{l}"])

        # shape: [batch_size, 1, vocab_size]
        if self.two_layer_predictor:
            pred_embed = self.pre_predictor(pred_embed)
        if self.weight_sharing:
            token_embedding = self.embedder.token_embedding.weight
            pred_logits = layers.matmul(
                x=pred_embed,
                y=token_embedding,
                transpose_y=True
            )
        else:
            pred_logits = self.predictor(pred_embed)
        


        pred_probs = layers.softmax(pred_logits, axis=-1)

        if self.use_pointer_network == 0:
            # context_embed = layers.assign(state["src_embed"])
            # context_token =layers.assign(state["src_token"])
            # typs_context = layers.assign(state["src_type"])
            # typs_context2 = layers.assign(state["src_type"])

            # context_embed = fluid.dygraph.to_variable(state["src_embed"].numpy())
            # context_token = fluid.dygraph.to_variable(state["src_token"].numpy())
            # typs_context =  fluid.dygraph.to_variable(state["src_type"].numpy())
            # typs_context2 = fluid.dygraph.to_variable(state["src_type"].numpy())

            context_embed = state["src_embed"]
            context_token = state["src_token"]
            typs_context =  state["src_type"]
            typs_context2 = state["src_type"]
           
            if self.num_latent==-10:
                context_embed = F.unsqueeze(context_embed,[1])
                #print("1",context_embed.shape)
                context_embed = layers.expand(context_embed,[1,self.num_latent,1,1])
                #print("2",context_embed.shape)
                context_embed = layers.reshape(context_embed,[state["batch_size"]*beam_size,-1,self.hidden_dim])
            
                context_token = F.unsqueeze(context_token,[1])
                #print(src_token_tmp.shape)
                context_token = layers.expand(context_token,[1,self.num_latent,1,1])
                context_token  = layers.reshape(context_token ,[state["batch_size"]*beam_size,-1,1])

                typs_context = F.unsqueeze(typs_context,[1])
                typs_context = layers.expand(typs_context,[1,self.num_latent,1,1])
                typs_context = layers.reshape(typs_context,[state["batch_size"]*beam_size,-1,1]) 
                
                typs_context2 = F.unsqueeze(typs_context2,[1])
                typs_context2 = layers.expand(typs_context2,[1,self.num_latent,1,1])
                typs_context2 = layers.reshape(typs_context2,[state["batch_size"]*beam_size,-1,1])
            #print(typs_context.shape)
            #print(context_embed.shape)   
            
            context_token = context_token.numpy()
            typs_context = typs_context.numpy().astype('float32')
            typs_context2 = typs_context2.numpy().astype('float32')
            typs_context[context_token==0] = 0 #batch_size,seq_len,1
            typs_context[context_token!=0] = 1
            typs_context2[context_token==0] = -1e10
            typs_context2[context_token!=0] = 0
            #print("fdfd",typs_context)
            #print("fdsfsdf",typs_context2)

            typs_context = fluid.dygraph.to_variable(typs_context)
            typs_context2 = fluid.dygraph.to_variable(typs_context2)
            typs_context = layers.squeeze(input=typs_context, axes=[2])
            typs_context2 = layers.squeeze(input=typs_context2, axes=[2])
            typs_context = F.unsqueeze(typs_context, [1])
            #typs_context = layers.expand(typs_context, [1, dec_embed.shape[1], 1])
            typs_context2 = F.unsqueeze(typs_context2, [1])
            #typs_context2 = layers.expand(typs_context2, [1, dec_embed.shape[1], 1])
            #typs_context.stop_gradient =True
            #typs_context2.stop_gradient =True
            #print("pred_embed",pred_embed.shape)
            #print("context_embed",context_embed.shape)
            pointer_logits = layers.matmul(
                x = pred_embed,
                y = context_embed,
                transpose_y=True
            )

            pointer_logits = layers.elementwise_mul(pointer_logits, typs_context)#x = [100, 15, 230],y=[5,15,230]
            pointer_logits = layers.elementwise_add(pointer_logits, typs_context2)
            pointer_probs = layers.softmax(pointer_logits,axis=-1)
            pointer_probs = layers.elementwise_mul(pointer_probs, typs_context)
            context_token = fluid.dygraph.to_variable(context_token) 
            know_onehot = layers.one_hot(context_token, self.num_token_embeddings)#batch_size,know_len,vocab_size
            #know_onehot = layers.expand(know_onehot,[self.num_latent,1,1])
            #know_onehot.stop_gradient =True
            lada = fluid.layers.sigmoid(self.lada)
            #print(lada) 
            pointer_probs = layers.matmul(x=pointer_probs,y = know_onehot)

            pred_probs = layers.elementwise_add(x=pred_probs*lada,y=pointer_probs*(1-lada)) 
        

        if self.use_pointer_network == 1:
            context_embed = state["src_embed"]
            context_token = state["src_token"]
            typs_context =  state["src_type"]
            typs_context2 = state["src_type"]

            context_token = context_token.numpy()
            typs_context = typs_context.numpy().astype('float32')
            typs_context2 = typs_context2.numpy().astype('float32')
            typs_context[context_token==0] = 0 #batch_size,seq_len,1
            typs_context[context_token!=0] = 1
            typs_context2[context_token==0] = -1e10
            typs_context2[context_token!=0] = 0

            typs_context = fluid.dygraph.to_variable(typs_context)
            typs_context2 = fluid.dygraph.to_variable(typs_context2)

            context_padding = layers.expand(typs_context,[1,1,self.hidden_dim]) # lamda_att
            typs_context = layers.squeeze(input=typs_context, axes=[2])

            context_len = layers.reduce_sum(typs_context,dim = 1,keep_dim=True)
            context_len = F.unsqueeze(context_len,[1])
            typs_context2 = layers.squeeze(input=typs_context2, axes=[2])
            typs_context = F.unsqueeze(typs_context, [1])
            #typs_context = layers.expand(typs_context, [1, pred_embed.shape[1], 1])
            typs_context2 = F.unsqueeze(typs_context2, [1])
            #typs_context2 = layers.expand(typs_context2, [1, pred_embed.shape[1], 1])
            typs_context.stop_gradient =True
            typs_context2.stop_gradient =True
            context_len.stop_gradient = True #lamda_att
            context_padding.stop_gradient = True # lamda_att
            context_embed_notpad = layers.elementwise_mul(context_embed,context_padding) #lamda_att
            context_embed_mean = layers.elementwise_div(layers.reduce_sum(context_embed_notpad,dim=1,keep_dim=True),context_len)#batch_size,1, dim #lambda_att

            att = layers.matmul(
                x = pred_embed,
                y = context_embed_mean,
                transpose_y=True
            )*self.hidden_dim**-0.5

            att = fluid.layers.sigmoid(att)

            pointer_logits = layers.matmul(
                x = pred_embed,
                y = context_embed,
                transpose_y=True
            )


            pointer_logits = layers.elementwise_mul(pointer_logits, typs_context)#x = [1, 15, 230],y=[1,15,230]
            pointer_logits = layers.elementwise_add(pointer_logits, typs_context2)
            pointer_probs = layers.softmax(pointer_logits,axis=-1)
            pointer_probs = layers.elementwise_mul(pointer_probs, typs_context)
            context_token = fluid.dygraph.to_variable(context_token) 
            know_onehot = layers.one_hot(context_token, self.num_token_embeddings)#batch_size,know_len,vocab_size
            know_onehot.stop_gradient =True

            pointer_probs = layers.matmul(x=pointer_probs,y = know_onehot)

            pred_probs = layers.elementwise_add(x=pred_probs*(1-att),y=pointer_probs*att)


        if self.use_pointer_network == 2:
            context_embed = state["src_embed"]
            context_token = state["src_token"]
            typs_context =  state["src_type"]
            typs_context2 = state["src_type"]

            context_att_emd = state["context_att_emd"]

            context_token = context_token.numpy()
            typs_context = typs_context.numpy().astype('float32')
            typs_context2 = typs_context2.numpy().astype('float32')
            typs_context[context_token==0] = 0 #batch_size,seq_len,1
            typs_context[context_token!=0] = 1
            typs_context2[context_token==0] = -1e10
            typs_context2[context_token!=0] = 0

            typs_context = fluid.dygraph.to_variable(typs_context)
            typs_context2 = fluid.dygraph.to_variable(typs_context2)

            context_padding = layers.expand(typs_context,[1,1,self.hidden_dim]) # lamda_att
            typs_context = layers.squeeze(input=typs_context, axes=[2])

            context_len = layers.reduce_sum(typs_context,dim = 1,keep_dim=True)
            context_len = F.unsqueeze(context_len,[1])
            typs_context2 = layers.squeeze(input=typs_context2, axes=[2])
            typs_context = F.unsqueeze(typs_context, [1])
            #typs_context = layers.expand(typs_context, [1, pred_embed.shape[1], 1])
            typs_context2 = F.unsqueeze(typs_context2, [1])
            #typs_context2 = layers.expand(typs_context2, [1, pred_embed.shape[1], 1])
            typs_context.stop_gradient =True
            typs_context2.stop_gradient =True
            context_len.stop_gradient = True #lamda_att
            context_padding.stop_gradient = True # lamda_att
            context_embed_notpad = layers.elementwise_mul(context_att_emd,context_padding) #lamda_att
            context_embed_mean = layers.elementwise_div(layers.reduce_sum(context_embed_notpad,dim=1,keep_dim=True),context_len)#batch_size,1, dim #lambda_att

            att = layers.matmul(
                x = pred_embed,
                y = context_embed_mean,
                transpose_y=True
            )*self.hidden_dim**-1

            att = fluid.layers.sigmoid(att)

            pointer_logits = layers.matmul(
                x = pred_embed,
                y = context_embed,
                transpose_y=True
            )


            pointer_logits = layers.elementwise_mul(pointer_logits, typs_context)#x = [1, 15, 230],y=[1,15,230]
            pointer_logits = layers.elementwise_add(pointer_logits, typs_context2)
            pointer_probs = layers.softmax(pointer_logits,axis=-1)
            pointer_probs = layers.elementwise_mul(pointer_probs, typs_context)
            context_token = fluid.dygraph.to_variable(context_token) 
            know_onehot = layers.one_hot(context_token, self.num_token_embeddings)#batch_size,know_len,vocab_size
            know_onehot.stop_gradient =True

            pointer_probs = layers.matmul(x=pointer_probs,y = know_onehot)

            pred_probs = layers.elementwise_add(x=pred_probs*(1-att),y=pointer_probs*att)



        #pred_logits = pred_logits[: , 0]
        pred_probs = pred_probs[:,0]
        #print(pred_logits.shape)
        pred_logits = layers.log(pred_probs)

        state["mask"] = mask
        return pred_logits, state

    def _ranking(self, 
                    src_token,
                    src_mask ,
                    src_pos ,
                    src_type ,
                    src_turn ,
                    src_embed,
                    predictions):
        """ Reranking generated responses. """
        
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)

        batch_size, num_latent, tgt_seq_len = predictions.shape

        # shape: [batch_size, num_latent, seq_len, 1]
        preds_token = F.unsqueeze(predictions, [3])
        preds_mask = F.not_equal(preds_token, self.padding_idx, "int64")
        preds_pos = layers.range(0, tgt_seq_len, 1, dtype="float32")
        preds_pos = F.unsqueeze(preds_pos, [0, 0, 1])
        preds_pos = layers.expand(preds_pos, [batch_size, num_latent, 1, 1])
        preds_pos = layers.cast(preds_pos, "int64")
        preds_type = layers.zeros_like(preds_token)
        preds_turn = layers.zeros_like(preds_token)

        scores = []
        for i in range(num_latent):
            pred_token = preds_token[:, i]
            pred_mask = preds_mask[:, i]
            pred_pos = preds_pos[:, i]
            pred_type = preds_type[:, i]
            pred_turn = preds_turn[:, i]

            input_mask = layers.concat([src_mask, pred_mask], axis=1)
            input_mask.stop_gradient = True
            pred_embed = self.embedder(pred_token, pred_pos, pred_type, pred_turn)
            embed = layers.concat([src_embed, pred_embed], axis=1)
            embed = self.embed_layer_norm(embed)

            mask_embed = self.mask_embed
            mask_embed = layers.expand(mask_embed, [batch_size, 1, 1])
            mask_embed = self.embed_layer_norm(mask_embed)

            out = layers.concat([mask_embed, embed], axis=1)
            mask = self._create_mask(input_mask, append_head=True)#,auto_regressive=not self.bidirectional_context)

            for layer in self.layers:
                out = layer(out, mask, None)

            mask_embed = out[:, 0]
            score = self.discriminator(mask_embed)
            scores.append(score[:, 0])
        scores = layers.stack(scores, axis=1)
        return scores

    def judge_topic_infer(self,src_token,src_mask,src_pos,src_type,src_turn,postive_token,
        postive_token_pos, postive_type,postive_turn,postive_mask,negative_token,negative_token_pos,
        negative_type,negative_turn
    ):
        src_token = src_token
        src_mask = src_mask
        src_pos = src_pos
        src_type = src_type
        src_turn = src_turn

        postive_token = postive_token
        postive_pos = postive_token_pos
        postive_type = postive_type
        postive_turn = postive_turn
        postive_mask = postive_mask

        negative_token = negative_token
        negative_pos = negative_token_pos
        negative_type = negative_type
        negative_turn = negative_turn
        negative_mask = negative_mask

        
        src_embed = self.embedder(src_token, src_pos, src_type, src_turn)
        postive_embed = self.embedder(postive_token,postive_pos,postive_type,postive_turn)
        negative_embed = self.embedder(negative_token,negative_pos,negative_type,negative_turn)

        postive_input_mask = layers.concat([src_mask,postive_mask],axis=1)
        negative_input_mask = layers.concat([src_mask,negative_mask],axis=1)

        postive_embed = layers.concat([src_embed, postive_embed], axis=1)
        postive_embed = self.embed_layer_norm(postive_embed)

        negative_embed = layers.concat([src_embed,negative_embed],axis=1)
        negative_embed = self.embed_layer_norm(negative_embed)
        
        batch_size = src_token.shape[0]

        tran_positive_probs,trans_negative_probs = self._transfer_network(
                postive_input_mask,negative_input_mask,postive_embed,negative_embed,batch_size
            )
        #batch_size,1
        trans_negative_probs = trans_negative_probs.numpy()
        tran_positive_probs = tran_positive_probs.numpy()

        #print(tran_positive_probs)
        #print("nexxt")
        #print(trans_negative_probs)
        trans_negative_probs[trans_negative_probs>=0.5] = 0
        trans_negative_probs[trans_negative_probs<0.5] =1

        tran_positive_probs[tran_positive_probs<0.5] = 0
        tran_positive_probs[tran_positive_probs>=0.5] = 1
        
        tp = np.sum(tran_positive_probs)
        tn = np.sum(trans_negative_probs)
        #print(tp)
        #print(tn)
        fn = batch_size-tp
        fp = batch_size-tn
        #print(fp)
        #print(fn)
        return tp,tn,fp,fn



    def _infer(self, inputs):
        """ Real inference process of model. """
        results = {}

        # Initial decode state.
        state = self._init_state(inputs)
        if "post_probs" in state:
            results["post_probs"] = state.pop("post_probs")

        # Generation process.
        gen_results = self.generator(self._decode, state)
        results.update(gen_results)

        if self.num_latent > 0:
            batch_size = state["batch_size"] // self.num_latent
            results["scores"] = layers.reshape(results["scores"], [batch_size, self.num_latent])
            results["log_p"] = results["scores"]
            results["src"] = layers.reshape(inputs["src_token"], [batch_size, -1])
            if "tgt_token" in inputs:
                results["tgt"] = layers.reshape(inputs["tgt_token"], [batch_size, -1])
            results["preds"] = layers.reshape(results["preds"], [batch_size, self.num_latent, -1])
            if self.use_discriminator:
                results["scores"] = self._ranking(inputs, results["preds"])
        else:
            batch_size = state["batch_size"]
            if "tgt_token" in inputs:
                results["tgt"] = layers.reshape(inputs["tgt_token"], [batch_size, -1])
        return results


UnifiedTransformer.register("UnifiedTransformer")
