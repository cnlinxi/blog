# kaldi中的解码过程

## 语言模型

### 困惑度

语言模型的评价方法为困惑度（perplexity，ppl），大小表示当前句子的“通顺”程度。

### WeNet利用WFST的一些参数

- `max_active`。和`beam`参数共同控制beam search的波束大小。
- `min_active`。
- `acoustic_scale`。声学模型给出的似然度放缩系数。
- `blank_skip_thresh`。跳过blank的阈值。
- `nbest`。

## 语音识别的建模单元

- 整句文本。如“Hello World”，对应的语音建模尺度为整条语音。
- 词。如孤立词“Good”、“World”，对应的语音建模尺度大约为每个词的发音范围。
- 音素。比如将“World”进一步表示为“/werld/”，其中每个音素（phone）作为基本单位，对应的语音建模尺度缩减为每个音素的发音范围。
- 三音素，也就是考虑了上下文的音素。如将音素“/d/”进一步表示为“{/l-d-sil/,/u-d-l/,...}”，对应的语音建模尺度是每个三音素的发音范围，长度与单音素差不多。
- 隐马尔科夫模型状态，也就是将每个三音素（triphone）都用一个三状态隐马尔科夫模型表示，并用每个状态作为建模粒度，对应的语音建模尺度将进一步缩短。

![](attachments/Pasted%20image%2020220510235633.png)

上图中，“DNN-HMM”表示深度神经网络-隐马尔科夫模型结构，“CTC”表示基于CTC损失函数的端到端结构，“Attention”表示基于注意力机制的端到端结构。

如上图所示，字词是由音素（Phone）组成；音素的上下文不同，因此同一个音素就有了不同的变体，结合了上下文的音素称之为三音素（Triphone），比如对于音素/d/，三音素/l-d-sil/、/u-d-l/是一对亲兄弟却是两家子；每个三音素又可以用一个独立的三状态HMM建模。由于很多三音素在语料中并未出现或者数量不多，并且可以通过决策树（Decision Tree）共享三音素的状态，所以对于拥有$N$个音素的语种，最终保留下来的三音素状态数量远小于$3N^3$，一般为几千，其被称为Senones。每一帧与每一个Senone的对应关系表示为三音素HMM的发射概率$P(x_i|s_j)$，其中，$s_j$表示第$j$个Senone，对应语音帧$x_i$的帧长通常为25ms，帧移为10ms。

![](attachments/Pasted%20image%2020220511003243.png)

上图展示了Phone、Triphone、Senone三者之间的关系，其中Senone是借助数学模型定义出来的音素变种，并没有直接的听觉感受；音素/sil/无实际发音，仅表示静音、字间停顿或者无意义的声音，$\#N$是Phone的个数，$\#N^3$和$\#3N^3$分别是Triphone、Senone的可能数量级，真实有效数量远小于数量级。

## 解码器

语音识别建模目标是：

$$
P(W|X)=\frac{P(X|W)P(W)}{P(x)}\propto P(X|W)P(W)
$$

其中，$P(X|W)$为声学模型（Acoustic Model，AM），$P(W)$为语言模型（Language Model，LM）。化整为零，各个击破，因此大多数研究将语音识别分为声学模型和语言模型两部分，分别求$P(X|W)$和$P(W)$。端到端（End-to-End）方法则直接计算$P(W|X)$，将声学模型和语言模型融为一体。

语音识别的最终目标是选择能使$P(W|X)=P(X|W)P(W)$最大的$W$，因此解码本质是一个搜索问题，并可借助加权有限状态转录机（Weighted Finite State Transducer，WFST）统一进行最优路径搜索。WFST由状态节点和边组成，且边上有对应的输入、输出符号及权重，形式为$x:y/w$，表示该边的输入符号为$x$，输出符号为$y$，权重为$w$，权重可以定义为概率（此时越大越好）、或者惩罚（此时越小越好）等，从起始到结束状态上的所有权重通常累加起来，记作该路径的分数，一条完整的路径必须从起始状态到结束状态。

![](attachments/Pasted%20image%2020220511004832.png)

上图中，粗圆表示开始，双线圆表示结束，其余圆表示中间状态。如上图所示，定义输入、输出均为词的WFST为G，定义输入为Phone、输出为词的WFST为L，定义输入为Triphone、输出为Phone的WFST为C，定义输入为Senone、输出为Triphone的WFST为H，至此得到4个WFST，也就是HCLG：

| WFST | 转换对象   | 输入       | 输出     |
| ---- | ---------- | ---------- | -------- |
| H    | HMM        | Senone序列 | 三音素   |
| C    | 上下文关系 | 三音素序列 | 音素序列 |
| L    | 发音词典   | 音素序列   | 词       |
| G    | 语言模型   | 词序列     | 词序列   |

上表中，“输入”、“输出”表示走完一条完整路径后整个WFST的输入、输出，而不是一条边上的输入、输出，可见前者的输出是后者的输入，因此可以将它们融合（Composition）成一条WFST，实现了Senone到Triphone（H）、Triphone到Phone（C）、Phone到Word（L）、Word到Sentence（G），这就是解码图（Decoding Graph）。

WFST的融合一般从大到小，也就是先将G和L融合，再依次融合C、H，每次融合都将进行确定化（Determination）和最小化（Minimisation）操作。WFST的确定化是指，确保给定某个输入符号，对应输出符号是惟一的；WFST的最小化是指，将WFST转换为一个状态节点和边更少的等价WFST。H、C、L、G的融合，常见过程为：

$$
HCLG={\rm min}({\rm det}(H\circ {\rm min}({\rm det}(C\circ {\rm min}({\rm det}(L\circ G))))))
$$

其中HCLG为最终的解码图WFST，$\circ$表示Composition，${\rm det}$表示确定化（Determination），${\rm min}$表示最小化（Minimisation）。OpenFST等工具实现了这些操作。

最终解码时，由于HMM已经在解码图中，因此只需要GMM或者DNN就可以利用HCLG进行解码了。给定语音特征序列$X$，可以通过GMM或DNN计算出$P(x_i|s_j)$，即HMM的发射概率，借助于HCLG，$P(W|X)\propto P(X|W)P(W)$的计算将变得简单：

$$
{\rm log}P(W|X)\propto {\rm log}P(X|W)+{\rm log}P(W)
$$

假设路径上的权重定义为惩罚，将W路径上的权重相加，再减去各状态针对输入的发射概率得到最终得分，该分数越小，则说明该语音X转录为W的可能性越大。由于HCLG中的权重是固定的，不同的$P(x_i|s_j)$将使得HCLG中相同的W路径有不同的得分。通过比较不同路径的得分，可以选择最优路径，该路径对应的W即为最终的解码结果。

由于HCLG搜索空间巨大，通常采用束搜索（Beam Search）方法，也就是每一步根据当前得分仅保留指定数目的最优路径，也就是N-best，直至走到终点，选择一条最优路径。

> [语音识别基本法](http://cslt.riit.tsinghua.edu.cn/mediawiki/images/2/25/Speech_book.pdf)
> [Decoders used in the Kaldi toolkit](https://kaldi-asr.org/doc/decoders.html)
> [faster-decoder.h](https://kaldi-asr.org/doc/faster-decoder_8h_source.html#l00035)
> [京东：基于WeNet的端到端语音识别优化方案与落地](https://bbs.csdn.net/topics/600623643)
> [LiJian-kaldi搭建在线语音识别系统/P4](https://www.bilibili.com/video/BV19a4y1h7cB?p=4)
> [WeNet训练初体验](https://note.abeffect.com/articles/2021/09/18/1631965447015.html)
> [语音识别中的WFST和语言模型](https://zhuanlan.zhihu.com/p/339718203)
> [如何通俗的理解beam search？](https://zhuanlan.zhihu.com/p/82829880)
