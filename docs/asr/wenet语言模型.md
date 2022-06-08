# wenet语言模型

WeNet中选择基于n-gram的统计语言模型（Language Model，LM），结合加权有限状态机（Weighted Finite State Transducer，WFST）和传统语音识别解码方法，以支持语言模型。

## 系统结构

![](attachments/Pasted%20image%2020220519232423.png)

如上图，WeNet采用U2（Unified Two Pass）方案以实现流式解码：

1. First Pass使用CTC解码结果作为中间结果，使用CTC prefix beam search或CTC WFST search算法产生N-best，其中CTC WFST search支持语言模型生成N-best。WFST search 是依靠传统解码图的解码方式。
2. Second Pass利用Attention Decoder对CTC产生的候选结果N-best进行重打分。

![](attachments/Pasted%20image%2020220519233035.png)

如上图所示，CTC WFST search支持语言模型。采用语言模型进行解码的过程包括解码图的构建和解码器的解码。

### 解码图的构建

端到端语音识别模型通常使用TLG解码图，构建TLG解码图就是将建模单元T，词典L和语言模型G三个层次的信息组合进一个解码图TLG中，其中：

- T为端到端语音识别模型的建模单元，一般来说，对中文的建模单元为汉字，英文为英文字母或BPE单元。
- L为词典，该词典比较简单，直接将单词拆分为对应的建模单元序列即可。比如，在中文中，将单词”我们“拆分为”我“、”们“两个建模单元，则词典中的一个词条类似于：`我们 我 们`，在英文中，将单词”we“拆分为”w“、”e“两个英文字母，则词典中的一个词条类似于：`we w e`。可以看到，该词典没有传统词典中音素的概念，无需人为设计发音序列。
- G为语言模型，也就是将n-gram语言模型转换为WFST形式。

因此CTC WFST Beam Search在解码时，使用构建的TLG解码图，实际已经利用了语言模型。

### 解码器的解码

解码器解码和传统语音识别中解码器一致，使用标准Viterbi Beam Search算法进行解码。

## 工程实现

直接采用Kaldi中的`LatticeFasterDecoder`解码器和相关工具实现基于TLG的集成，并实现了blank frame skipping算法，以加速解码。对于引用的Kaldi代码，保持了其原有的目录结构，并利用`glog`替换原始的日志系统:

```cpp

#define KALDI_WARN \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_WARNING).stream()
#define KALDI_ERR \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_ERROR).stream()
#define KALDI_INFO \
  google::LogMessage(__FILE__, __LINE__, google::GLOG_INFO).stream()
#define KALDI_VLOG(v) VLOG(v)

#define KALDI_ASSERT(condition) CHECK(condition)
```

参见`wenet/runtime/core/decoder/ctc_wfst_beam_search.cc`相关代码。

## WeNet中语言模型的使用方法

`wenet/examples/aishell/s0/run.sh`中给出了准备词典、训练语言模型、构建解码图和利用TLG解码的示例：

```shell

# wenet/examples/aishell/s0/run.sh
# 7.1 Prepare dict
unit_file=$dict
mkdir -p data/local/dict
cp $unit_file data/local/dict/units.txt
# tools/fst/prepare_dict.py usage:
# sys.argv[1]: e2e model unit file(lang_char.txt)
# sys.argv[2]: raw lexicon file
# sys.argv[3]: output lexicon file
# sys.argv[4]: bpemodel
tools/fst/prepare_dict.py $unit_file ${data}/resource_aishell/lexicon.txt \
    data/local/dict/lexicon.txt
# 7.2 Train lm
lm=data/local/lm
mkdir -p $lm
tools/filter_scp.pl data/train/text \
     $data/data_aishell/transcript/aishell_transcript_v0.8.txt > $lm/text
# train LM by SRILM tools
# This script takes no arguments.
# It takes as input the files
# data/local/lm/text
# data/local/dict/lexicon.txt
local/aishell_train_lms.sh
# 7.3 Build decoding TLG
# usage: tools/fst/compile_lexicon_token_fst.sh <dict-src-dir> <tmp-dir> <lang-dir>
tools/fst/compile_lexicon_token_fst.sh \
    data/local/dict data/local/tmp data/local/lang
# usage: make_tlg.sh <lm-dir> <src-lang> <tgt-lang>
tools/fst/make_tlg.sh data/local/lm data/local/lang data/lang_test || exit 1;
# 7.4 Decoding with runtime
# usage: $0 [options] <wav.scp> <label_file> <model_file> <dict_file> <output_dir>
./tools/decode.sh --nj 16 \
    --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
    --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
    --fst_path data/lang_test/TLG.fst \
    data/test/wav.scp data/test/text $dir/final.zip \
    data/lang_test/words.txt $dir/lm_with_runtime
```

## 语言模型参与最终打分

对于解码的候选结果进行打分，除了有声学模型的分数外，还可能会有额外的语言模型和长度惩罚分。假设$W$为解码结果，$X$为输入语音，$\alpha,\beta$分别为超参数,则最终得分的计算公式如下：

$$
score=P_{am}(W|X)\cdot P_{lm}(W)^{\alpha}\cdot |W|^{\beta}
$$


> [WeNet 更新：支持语言模型](https://mp.weixin.qq.com/s/uMYzoRbJocRWWo9MVzXceA)
> [github/wenet/kaldi](https://github.com/wenet-e2e/wenet/tree/main/runtime/core/kaldi)
> [github/wenet/lm_doc](https://github.com/wenet-e2e/wenet/blob/main/docs/lm.md)
> [LiJian-kaldi搭建在线语音识别系统/P4](https://www.bilibili.com/video/BV19a4y1h7cB?p=4)
> 洪《语音识别》-P185~P191
> 陈《语音识别实战》-P123（transition-id）