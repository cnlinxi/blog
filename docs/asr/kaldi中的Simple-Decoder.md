# kaldi中的Simple-Decoder

## HCLG

HCLG是WSFT格式的语音任务相关的表示，对于语音识别中用到的隐马尔科夫模型（H）、发音词典（L）、语言模型（G）以及上下文（C），都可以用 一个加权有限状态接收器（WFST）来表示。

### 准备L

发音词典。

#### 文件准备

准备文件，放到dict目录下：

- lexicon.txt。发音词典。格式：文本\s音素\n，比如：

```

// lexicon.txt

<SIL> SIL
今天 j in1 t ian1
是 sh ix4
```

- phones.txt。音素集合。格式：音素\n，比如：

```

// phones.txt
SIL
j
in1
...
```

- silence_phones.txt。静音音素集合。格式：音素\n，比如：

```

// silence_phones.txt
SIL  // 静音
SPN  // 有声音但无法识别
NSN  // 非口语噪声
LAU  // 笑声
```

- nonsilence_phones.txt。非静音音素集合。格式：音素\n，比如：

```

// nonsilence_phones.txt
j
in1
...
```

- optional_silence.txt。填充词之间的静音音素。格式：音素\n，比如：

```

// optional_silence.txt
SIL  // 词间静音
```

![](attachments/Pasted%20image%2020220508212602.png)

#### 工具构建

使用`egs/wsj/s5`中的`utils/prepare_lang.sh`：

```

Usage: utils/prepare_lang.sh <dict-src-dir> <oov-dict-entry> <tmp-dir> <lang-dir>
```

比如：

```shell


utils/prepare_lang.sh <YOUR_DIR>/dict '<SIL>' <YOUR_DIR>/tmp <YOUR_DIR>/lang
```

产物：

- L.fst。本步骤产生的L（Lexicon）有限状态转录机。将所有的音素转成了对应关系（解码图）。该图的输入：音素组合，输出：词。是后续训练H、C、L解码图的一种，G训练的时候还用不到。
- L_disambig.fst。消除歧义，消除同音字导致的歧义，`#+数字`区分同一个音对应的不同字。
- oov.txt。集外词（Out of Vocabulary Words)，oov.int是对应的id。
- topo。拓扑结构。
- phone.txt存放了所有的音素及其编号，形如：`<phone> <phone id>`。

![](attachments/Pasted%20image%2020220508231132.png)

![](attachments/Pasted%20image%2020220508231153.png)

### 准备G

语言模型。

#### 准备语料

text文件。需要根据`L`中的lexicon.txt分词，例如：

```

// text文件
今天 几 号
今天 是 几 号
```

#### 工具构建

1. 准备语言模型

例如，

```shell

// order：一阶
ngram-count -text text -order 1 -w gram.count.1
// 创建语言模型，产物LM_1，采用kndiscount平滑
ngram-count -read gram.count.1 -order 1 -lm LM_1 -interpolate -kndiscount
```

产物：

- gram.count.1
- LM_1。语言模型。

2. 产生G

使用`egs/wsj/s5`中的`utils/format_lm_sri.sh`：

```

Usage: utils/format_lm_sri.sh [options] <lang-dir> <arpa-LM> [<lexicon>] <out-dir>
```

比如上例中：

```

utils/format_lm_sri.sh lang/ gram/LM_1 out_lang/
```

产物：

- G.fst。本步骤产生的G有限状态转录机。
- L.fst。
...

![](attachments/Pasted%20image%2020220508222542.png)

### 准备H

HMM。声学模型。

#### 文件准备

- wav.scp。音频和对应路径。格式：音频序号\s路径\n。
- text。语音标注文件，比如text：

```

// text文件
001 今天 几 号
002 今天 是 几 号
```

#### 工具构建

1. 声学参数准备

提声学特征，使用`egs/wsj/s5`中的`steps/make_mfcc.sh`：

```

Usage: steps/make_mfcc.sh [options] <data-dir> [<log-dir> <mfcc-dir>]
Note: <log-dir> defaults to <data-dir>/log
	  <mfcc-dir> defaults to <data-dir>/data
Options:
--mfcc-config <config-file>           # config passed to compute-mfcc-feats.
--nj <nj>                             # number of parallel jobs.
--cmd <run.pl|queue.pl <queue opts>>  # how to run jobs.
--write-utt2num-frames <true|false>   # if true, write utt2num_frames file
--write-utt2dur <true|false>          # if true, write utt2dur file.
```

根据上述描述，实际只需要指定数据路径data-dir即可。`make_mfcc.sh`应该是根据`wav.scp`计算声学参数了。

之后需要计算平均值和方差，使用`egs/wsj/s5`中的`steps/compute_cmvn_stats.sh`：

```

Usage: steps/compute_cmvn_stats.sh [options] <data-dir> [<log-dir> <cmvn-dir>]
Note: <log-dir> defaults to <data-dir>/log
      <cmvn-dir> defaults to <data-dir>/data
```

同样地，`compute_cmvn_stats.sh`应该也是通过`wav.scp`计算声学参数的平均值和方差，以便进行归一化，主要目的是提高声学特征对说话人、录音设备、环境、音量等因素的鲁棒性。

产物：

- `feats.scp`。声学特征scp文件。
- `cmvn.scp`。声学特征均值方差scp文件。
- `utt2num_frames`。`make_mfcc.sh`时可选文件，记录每条音频的帧数。
...

2. 训练声学模型

训练声学模型，使用`egs/wsj/s5`中的`steps/train_mono.sh`：

```

Usage: steps/train_mono.sh [options] <data-dir> <lang-dir> <exp-dir>
```

其中，`<data-dir>`是放置feats.scp、cmvn.scp、text等数据的目录，`<lang-dir>`是上述准备`L`时的产物目录，`<exp-dir>`是声学模型输出目录。

产物：

- `final.mdl`。声学模型。
- `ali.1.gz`。对齐。
- `phones.txt`。音素集合。

### 获得HCLG

#### 构图

所需文件：

- `lang`。准备L阶段的输出目录。单词音素。
- `exp/mono`。准备H阶段的输出目录。声学模型。

使用`egs/wsj/s5`中的`utils/mkgraph.sh`：

```

utils/mkgraph.sh [options] <lang-dir> <model-dir> <graphdir>
```

承接上述例子：

```

utils/mkgraph.sh lang/ exp/mono/ exp/mono/graph
```

产物：

- HCLG.fst。*HCLG*有限状态转录机。

#### HCLG可视化

使用kaldi中的`fstdraw`工具，该工具在kaldi安装后是全局工具。比如：

```shell

fstdraw HCLG.fst > HCLG.dot
dot -Tjpg -Gdpi=800 HCLG.dot > HCLG.jpg
```

#### 缩减HCLG

主要思路是缩减L，单词音素，减少音素对应的状态数：

```

// 默认音素状态数太多，导致最终的HCLG.fst较为复杂
utils/prepare_lang.sh <YOUR_DIR>/dict '<SIL>' <YOUR_DIR>/tmp <YOUR_DIR>/lang
// 缩减音素的状态数，不需要音素B（开始）/I（中间）/E（结束）/S（静音）的区分
utils/prepare_lang.sh --num-sil-states 1 --num-nonsil-states 1 --position-dependent-phones false <YOUR_DIR>/dict '<SIL>' <YOUR_DIR>/tmp <YOUR_DIR>/lang
```

### 小结

#### 所需文件

| 文件    | 作用                          | 内容模式                                                        | 示例                                              |
| ------- | ----------------------------- | --------------------------------------------------------------- | ------------------------------------------------- |
| wav.scp | 语音文件id到语音wav文件的映射 | `<uterranceID> <full_path_to_audio_file>`                       | sentense_id path_to_wavfile.wav                   |
| text    | 语音文件id到语音内容的映射    | `<uterranceID> <text_transcription>`                             | sentense_id 今天天气很好                          |
| utt2spk | 语音文件id到说话人的id映射    | `<uterranceID> <speakerID>`                                     | sentense_id speaker_id                            |
| spk2utt | 说话人的id 到语音文件id的映射 | `<speakerID> <uterranceID_1> <uterranceID_2> <uterranceID_3> …` | speaker_id sentense_id1 sentense_id2 sentense_id3 |

#### 步骤总结

![](attachments/Pasted%20image%2020220508230547.png)

## 解码

### 所需文件

1. HCLG
2. 声学模型。构建G.fst时产生的transition文件。 

使用`gmm-decode-simple`，这是kaldi安装之后的全局工具：

```

gmm-decode-simple

Decode features using GMM-based model.
Viterbi decoding, only produces linear sequence; any lattice produced is linear

Usage: gmm-decode-simple [options] <model-in> <fst-in> <features-rspecifier> <words-wspecifier> [<alignments-wspecifier>] [<lattice-wspecifier>]
```

tricks：

- vimdiff。文本对比工具，`vimdiff <file1> <file2>`
- ll -lh。查看文件详细信息，包括软连接。
- ln -s <实际文件路径> <软链接路径>

> [【Kaldi解码原理】按行分析Simple-Decoder](https://www.bilibili.com/video/BV1KU4y1K7gz)
> [LiJian-kaldi搭建在线语音识别系统 资料汇总](https://blog.csdn.net/weixin_42217661/article/details/117715318)
> [LiJian-kaldi搭建在线语音识别系统](https://www.bilibili.com/video/BV19a4y1h7cB)
> [语音资料分享（更新ing）](https://www.bilibili.com/read/cv7110144)

