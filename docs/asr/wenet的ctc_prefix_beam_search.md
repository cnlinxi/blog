# wenet的ctc_prefix_beam_search

## CTC字符串和规整字符串

- CTC字符串：模型在每个时刻输出字符组成的字符串。
- 规整字符串：`CTC字符串`去除连续重复和blank之后的字符串。

## CTC常见的解码方式

1. greedy decode。每帧输出最大值，然后规整。
![](attachments/Pasted%20image%2020220529214340.png)

2. 在`CTC字符串`上做beam search（也称`CTC Beam Search`），输出n个结果之后规整，合并相同序列，然后再应用语言模型，也即SecondPass LM。
3. 在`规整字符串`上做beam search（也称`CTC Prefix Beam Search`），可额外应用语言模型。可以在解码过程中直接应用语言模型，也即FirstPass LM。
4. 使用FST静态解码，可引入语言模型和字典模型。

`CTC字符串`上beam search（`CTC Beam Search`）和`规整字符串`上beam search（`CTC Prefix Beam Search`）的主要区别在于，同样的beam size下，前者丢弃的CTC路径更多，因此效果更差一些：

1. `CTC Beam Search`在解码过程中产生的CTC候选路径有可能产生重复项，而这些重复项在`CTC Beam Search`的计算过程中是各自独立的，占用了beam数。
2. `CTC Prefix Beam Search`在解码过程中合并重复项的概率，从而提升了解码的多样性和鲁棒性。

如下图，直接在`CTC字符串`上做beam search，当beam size=3时，每个时刻只有三条路径：

![](attachments/Pasted%20image%2020220515223121.png)

在`规整字符串`上做beam search，当beam size=3时，每个时刻则可以保留更多的路径：

![](attachments/Pasted%20image%2020220515223409.png)

CTC Prefix Beam Search仍然会丢失一些CTC序列的概率，比如上例中，当时刻$t$的beam中只有a而没有ab时，在$t+1$时刻计算ab，只使用了$t$时刻a的扩展，此时会丢失来自$t$时刻ab的得分。

## CTC Prefix Beam Search原理

利用神经网络和CTC建模序列问题时，神经网络输出一个$T\times M$的矩阵，其中$T$表示音频帧数，$M$表示词典的大小。CTC Prefix Beam Search算法则在该矩阵的基础上，找出概率最高的N条路径。

CTC Prefix Beam Search的过程中，每个时刻均执行：

1. 扩展。根据前缀串和当前时刻的输出，计算所有当前可能输出的`规整字符串`概率。
2. 规约。将规约得到的相同串对应的候选概率相加。
3. 裁剪。仅保留top k个最优的序列做下一步扩展。上图中$k=3$，绿色表示保留，红色表示裁剪。

对于$t$时刻处在beam中的每个规整字符串，更新其对应的$t+1$时刻规整字符串的概率值，此时并不能直接使用$t$时刻规整字符串的概率乘上$t+1$时刻输出字符的概率以得到$t+1$时刻规整字符串的概率，主要原因是根据CTC规则，使用相同的$t$时刻规整字符串，相同的$t+1$时刻输出字符能够得到不同的$t+1$时刻规整字符串。比如$t$时刻CTC字符串为aa和a-（“-”表示blank），对应的规整字符串均为a，当$t+1$时刻输出字符a时，得到的$t+1$时刻规整字符串分别为a和aa。因此需要区分对待blank和非blank结尾的CTC字符串规整概率：

- $p_b(L)$表示所有以blank结尾且规整后字符串是L的各个CTC字符串概率之和。
- $p_{nb}(L)$表示所有以非blank结尾且规整后字符串是L的各个CTC字符串概率之和。

假设$t=3$时规整字符串为a，则：

$$
p_b(a)=p(aa-)+p(-a-)+p(a--)
$$

$$
p_{nb}(a)=p(aaa)+p(-aa)+p(--a)
$$

若$t=4$时输出字符a，产生规整字符串有如下四种情况：

1. 当$t+1$输出是blank时，产生规整字符串a。
2. 当$t+1$输出是a时，可以产生规整字符串a（$t$时刻输出为非blank）。
3. 当$t+1$输出是a时，也可以产生规整字符串aa（$t$时刻输出为blank）。
4. 当$t+1$输出为b时，产生规整字符串ab。

四种情况需要更新的统计量如下：

$$
p_b^{t+1}(a)+=[p_b^t(a)+p_{nb}^t(a)]p_{ctc}^{t+1}(-)
$$

$$
p_{nb}^{t+1}(a)+=p_{nb}^t(a)p_{ctc}^{t+1}(a)
$$

$$
p_{nb}^{t+1}(aa)+=p_{b}^t(a)p_{ctc}^{t+1}(a)
$$

$$
p_{nb}^{t+1}(ab)+=[p_b^t(a)+p_{nb}^t(a)]p_{ctc}^{t+1}(b)
$$

如上式，在情况1和4中，可以通过2种途径获得所需要的$t$时刻规整字符串a，因此需要遵循概率论中的加法原理，将两个概率值加起来；然后遵循乘法原理，乘第$t+1$时刻的概率值。

总之，如果当前时刻输出字符（不含blank）和规整字符串最后一个字符不同，则不需要区分blank和非blank规整概率，两者加起来之后乘当前字符概率即可；如果当前时刻输出字符（不含blank）和规整字符串最后一个字符相同，此时有可能出现相同规整字符串和当前输出字符，产生当前时刻不同规整字符串的情况，因此需要区分blank和非blank规整概率。

![](attachments/Pasted%20image%2020220513005416.png)

如上图所示，假设左上角为神经网络输出的$T\times M$矩阵，其中$T=3,M=3$。以上图$T=2$，绿框$a$扩展到$T=3$的过程为例。

在步骤1中：

- 假设$a$扩展到$\epsilon$，查阅表格可知，该过程的概率为0.10，因此计算得到的概率为$0.3875\times 0.10=0.03875$，注意到乘数为0.3875，而非0.2275或者0.16，这是因为下一个字符为$\epsilon$，与当前字符a不同，应该使用候选概率的加和值。
- 假设$a$扩展到$a$，查阅表格可知，该过程的概率为0.50，由于当前字符和下一个字符相同，因此乘数应采用各自的候选概率，分别是$0.2275\times 0.50=0.11375$和$0.16\times 0.50=0.08000$。
- 假设$a$扩展到$b$，查阅表格可知，该过程的概率为0.40，并且当前字符和下一个字符不同，乘数应采用候选概率的加和值，因此计算得到的概率为$0.3875\times 0.40=0.1150$。

以此类推，计算出其它的概率值。注意到，通过不同的路径规约之后可以得到相同串，比如$a\to a,a\epsilon \to a$，因此根据步骤2将规约得到的相同串对应的候选概率相加，得到a对应的概率加和值为$0.11375+0.03875=0.1525$。

在步骤3中，由于$k=3$，则保留概率最高的3组最优序列，上图中以绿色标注。

## 具体步骤和Python实现

在[First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs](https://arxiv.org/abs/1408.2873)中给出CTC Prefix Beam Search的具体步骤如下：

![](attachments/Pasted%20image%2020220515224208.png)

如上算法流程图中，除了`else if c = space then`，分别对应上述情况1~4。

对应的Python实现：

```python

"""
Author: Awni Hannun

This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.

The algorithm is a prefix beam search for a model trained
with the CTC loss function.

For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873

"""

import numpy as np
import math
import collections

NEG_INF = -float("inf")

def decode(probs, beam_size=10, blank=0):
    """
    Performs inference for the given output probabilities.

    Arguments:
      probs: The output probabilities (e.g. log post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.

    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T): # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S): # Loop over vocab
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam: # Loop over beam

				# 情况1
                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
	              # n_p_b：当前时刻blank概率，n_p_nb：当前时刻非blank概率
                  n_p_b, n_p_nb = next_beam[prefix]
                  # logsumexp(n_p_b, p_b + p, p_nb + p): 
                  # n_p_b x ((p_b + p) + (p_nb + p))
                  n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                  next_beam[prefix] = (n_p_b, n_p_nb)
                  continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:  # 情况4
                  n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:  # 情况2和3
                  # We don't include the previous probability of not ending
                  # in blank (p_nb) if s is repeated at the end. The CTC
                  # algorithm merges characters not separated by a blank.
                  n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:  # 情况2，规整字符串不变，但规整概率需要更新
                  n_p_b, n_p_nb = next_beam[prefix]
                  n_p_nb = logsumexp(n_p_nb, p_nb + p)
                  next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(),
                key=lambda x : logsumexp(*x[1]),
                reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])
```

## WeNet中获取字级别的时间戳

在语音识别的一些任务中，字级别的时间戳和N-best扮演着重要的作用，比如视频应用中，语音识别结合字级别的时间戳可以在精确的时间显示字幕，在会议场景中，字级别的时间戳可以标定与会者说某句话某个字的精确时间。

每个前缀串可以由多个串规约而成。WeNet 使用前缀串的被规约串中最优的一条路径，即viterbi路径来记录时间信息，viterbi路径记录了每个字峰值的时间。

![](attachments/Pasted%20image%2020220512013922.png)

如上图所示，解码后一共得到三个解码结果：a，ab和ba：

- 对于解码结果a来说，考虑到剪枝策略，因此规约前的串只可能是$\epsilon a\epsilon$，$\epsilon a a$或者$aaa$。
	- viterbi分数较高的是$aaa$，为$0.40\times 0.35\times 0.50=0.07$。
	- a的峰值在$T=3$，对应的概率为$0.50$，对应的时间戳$T=[3]$。
- 对于解码结果ab来说，规约前的串可能是aab或者$a\epsilon b$。
	- viterbi分数较高的是$a\epsilon b$，为$0.40\times 0.40\times 0.40=0.064$。
	- a和b对应的时间戳为$T=[1,3]$。
- 对于解码结果ba来说，规约前的串可能是$\epsilon ba,b\epsilon a,ba\epsilon ,baa$或者$bba$。
	- viterbi分数较高的是$b\epsilon a$，为$0.35\times 0.40\times 0.50=0.07$。
	- b和a的时间戳为$T=[1,3]$。

通常一个字的时间戳信息应该包括起始时间和终止时间，而使用上述算法，只能获取该字峰值所在的时间，因此在WeNet实现中，考虑到延迟等因素，将峰值所在的时间当做该字的终止时间，上一个字峰值所在的时间当做起始时间。

## 代码实现

WeNet使用HashMap保存解码过程中产生的规整字符串（字符对应的ID序列）及其对应的分数信息：

```cpp

// wenet/runtime/core/decoder/ctc_prefix_beam_search.cc
std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;

// wenet/runtime/core/decoder/ctc_prefix_beam_search.h
struct PrefixScore {
  float s = -kFloatMax;               // 以blank结尾的分数
  float ns = -kFloatMax;              // 以非blank结尾的分数
  float v_s = -kFloatMax;             // 以blank结尾的viterbi分数
  float v_ns = -kFloatMax;            // 以非blank结尾的viterbi分数
  float cur_token_prob = -kFloatMax;  // 当前token峰值的概率
  std::vector<int> times_s;           // 以blank结尾viterbi路径的时间戳
  std::vector<int> times_ns;          // 以非blank结尾viterbi路径的时间戳

  // blank和非blank规整概率之和
  float score() const { return LogAdd(s, ns); }
  // viterbi分数为blank和非blank维特比分数的最大值
  float viterbi_score() const { return v_s > v_ns ? v_s : v_ns; }
  // 根据viterbi分数选择规整串的时间戳
  const std::vector<int>& times() const {
    return v_s > v_ns ? times_s : times_ns;
  }

  ...

  // 语言模型和e2e声学模型分数之和
  float total_score() const { return score() + context_score; }
};
```

在解码过程中，通过`for`循环遍历每个时刻，获取每个时刻的输出，然后执行CTC Prefix Beam Search。

### 第一次剪枝

声学模型每个时刻输出`vocab_size`个概率值，第一次剪枝时，选取前`opts_.first_beam_size`（默认值10）个字符和对应概率值。

```cpp

// wenet/runtime/core/decoder/ctc_prefix_beam_search.cc
// 1. First beam prune, only select topk candidates
std::tuple<Tensor, Tensor> topk = logp_t.topk(opts_.first_beam_size);
Tensor topk_score = std::get<0>(topk);
Tensor topk_index = std::get<1>(topk);
```

### 令牌传播

令牌传播（Token Passing）中遍历当前时刻的`opts_.first_beam_size`个输出，对每一个输出执行CTC Prefix Beam Search。

```cpp

// 2. Token Passing
// next_hyps 记录下一个时刻的规约字符串，避免更新当前时刻的规约字符串
std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
...
for (int i = 0; i < topk_index.size(); ++i) { // 遍历当前beam_size个输出
      int id = topk_index[i];
      auto prob = topk_score[i];
      for (const auto& it : cur_hyps_) {
        const std::vector<int>& prefix = it.first;
        const PrefixScore& prefix_score = it.second;
	    // 如果 prefix 不在 next_hyps 中, next_hyps[prefix] 则会插入默认的分数信息
	    // 也即PrefixScore(-inf, -inf)
	    
	    // 代码中利用对数，将乘法转化为加法：LogAdd()
	    
	    // 为叙述方便，下述将当前时刻新的规整字符串称作“新串”
	    // 前一时刻的规整字符串称作“前缀串”
	
        if (id == opts_.blank) {
          // 情况1：当前时刻输出blank，则新串和前缀串相同
	      // Case 0: *a + ε => *a
          PrefixScore& next_score = next_hyps[prefix];
          // 新串以 ε 结尾的分数 next_score.s 为两者之和：
          //  1. 新串以 ε 结尾的分数 next_score.s
          //  2. 前缀串的分数 prefix_score.score() 与当前输出的概率的对数 prob 之和
          next_score.s = LogAdd(next_score.s, prefix_score.score() + prob);
          // 新串以 ε 结尾的 viterbi 分数 next_score.v_s 为：
          // 前缀串的 viterbi 分数 prefix_score.viterbi_score() 与当前输出的概率 prob 的对数和
          next_score.v_s = prefix_score.viterbi_score() + prob;
          // 新串以 ε 结尾的 viterbi 路径的时间戳 next_score.times_s 等于：
          // 前缀串的时间戳 prefix_score.times()
          next_score.times_s = prefix_score.times();
          // 上下文偏置（context biasing），通过偏置短语列表使模型倾向于识别特定词
          // Prefix not changed, copy the context from prefix.
          if (context_graph_ && !next_score.has_context) {
            next_score.CopyContext(prefix_score);
            next_score.has_context = true;
          }
        } else if (!prefix.empty() && id == prefix.back()) {
	      // 情况2：当前时刻的输出字符（不含blank），与CTC字符串最后一个字符相同（因此与前缀串最后一个字符也相同）
	      // 则新串和前缀串相同
          // Case 2: *a + a => *a
          PrefixScore& next_score1 = next_hyps[prefix];
          // 新串以非 ε 结尾的分数 next_score1.ns 为两者之和：
          //  1. 新串以非 ε 结尾的分数 next_score1.ns
          //  2. 前缀串以非 ε 结尾的分数 prefix_score.ns 和当前输出的概率 prob 的对数和
          next_score1.ns = LogAdd(next_score1.ns, prefix_score.ns + prob);
          // 判断是否需要更新以非 ε 结尾的新串 viterbi 分数 next_score1.v_ns
          if (next_score1.v_ns < prefix_score.v_ns + prob) {
            next_score1.v_ns = prefix_score.v_ns + prob;
            // 判断是否需要更新新串中最后一个字峰值的概率 next_score1.cur_token_prob
            if (next_score1.cur_token_prob < prob) {
              next_score1.cur_token_prob = prob;
              // 新串以非 ε 结尾的 viterbi 路径的时间戳 next_score1.times_ns 等于：
              // 前缀串以非 ε 结尾的 viterbi 路径的时间戳 prefix_score.times_ns
              next_score1.times_ns = prefix_score.times_ns;
              CHECK_GT(next_score1.times_ns.size(), 0);
              // 更新新串中最后一个字峰值的位置 next_score1.times_ns.back()
              next_score1.times_ns.back() = abs_time_step_;
            }
          }
          // 上下文偏置
          if (context_graph_ && !next_score1.has_context) {
            next_score1.CopyContext(prefix_score);
            next_score1.has_context = true;
          }

          // 情况3：当前时刻的输出字符（不含blank），与前缀串最后一个字符相同，但CTC字符串最后一个字符为blank
          // 则新串在前缀串的基础上加上当前时刻的输出字符
          // Case 3: *aε + a => *aa
          std::vector<int> new_prefix(prefix);
          // 将当前时刻的输出拼接到前缀串上，得到新串
          new_prefix.emplace_back(id);
          PrefixScore& next_score2 = next_hyps[new_prefix];
          // 新串以非 ε 结尾的分数 next_score2.ns 为两者之和：
          //  1. 新串以非 ε 结尾的分数 next_score2.ns
          //  2. 前缀串以 ε 结尾的分数 prefix_score.s 和当前输出的概率 prob 的对数和
          next_score2.ns = LogAdd(next_score2.ns, prefix_score.s + prob);
          // 判断是否需要更新新串以非 ε 结尾的 viterbi 分数 next_score2.v_ns
          if (next_score2.v_ns < prefix_score.v_s + prob) {
            next_score2.v_ns = prefix_score.v_s + prob;
            next_score2.cur_token_prob = prob;
            // 新串以非 ε 结尾的 viterbi 路径的时间戳 next_score2.times_ns 等于：
            // 前缀串以 ε 结尾的 viterbi 路径的时间戳 prefix_score.times_s，拼接上当前时间步 abs_time_step_
            next_score2.times_ns = prefix_score.times_s;
            next_score2.times_ns.emplace_back(abs_time_step_);
          }
          if (context_graph_ && !next_score2.has_context) {
            // 上下文偏置
            // Prefix changed, calculate the context score.
            next_score2.UpdateContext(context_graph_, prefix_score, id,
                                      prefix.size());
            next_score2.has_context = true;
          }
        } else {
          // 情况4：当前时刻的输出字符（不含blank），与CTC字符串最后一个字符不同（因此与前缀串最后一个字符也不同）
          // 则新串在前缀串的基础上加上当前时刻的输出字符
          // Case 4: *a + b => *ab, *aε + b => *ab
          std::vector<int> new_prefix(prefix);
          // 将当前时刻的输出拼接到前缀串上，得到新串
          new_prefix.emplace_back(id);
          PrefixScore& next_score = next_hyps[new_prefix];
          // 新串以非 ε 结尾的分数 next_score.ns 为两者之和：
          //  1. 新串以非 ε 结尾的分数 next_score.ns
          //  2. 前缀串的分数 prefix_score.score() 和当前输出的概率 prob 的对数和
          next_score.ns = LogAdd(next_score.ns, prefix_score.score() + prob);
          // 判断是否需要更新新串以非 ε 结尾的 viterbi 分数 next_score.v_ns
          if (next_score.v_ns < prefix_score.viterbi_score() + prob) {
            next_score.v_ns = prefix_score.viterbi_score() + prob;
            // 更新前缀串最后一个字峰值的概率
            next_score.cur_token_prob = prob;
            // 新串以非 ε 结尾的 viterbi 路径的时间戳 next_score.times_ns 等于：
            // 前缀串 viterbi 路径的时间戳 prefix_score.times()，拼接上当前时间步 abs_time_step_
            next_score.times_ns = prefix_score.times();
            next_score.times_ns.emplace_back(abs_time_step_);
          }
          if (context_graph_ && !next_score.has_context) {
            // 上下文偏置
            // Calculate the context score.
            next_score.UpdateContext(context_graph_, prefix_score, id,
                                     prefix.size());
            next_score.has_context = true;
          }
        }
      }
    }
```

### 第二次剪枝

 第二次剪枝时只保留分数最高的`opts_.second_beam_size`条路径（N-Best），便于后续的重打分。

```cpp

// 3. Second beam prune, only keep top n best paths
std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(),
														  next_hyps.end());
int second_beam_size =
	std::min(static_cast<int>(arr.size()), opts_.second_beam_size);
std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(),
				 PrefixScoreCompare);
arr.resize(second_beam_size);
std::sort(arr.begin(), arr.end(), PrefixScoreCompare);
```

### 更新规整字符串

```cpp

cur_hyps_.clear();
outputs_.clear();
hypotheses_.clear();
likelihood_.clear();
viterbi_likelihood_.clear();
times_.clear();
for (auto& item : hpys) {
  // 更新前缀串
  cur_hyps_[item.first] = item.second;
  // 更新语言模型的Context Graph
  UpdateOutputs(item);
  // 更新解码结果
  hypotheses_.emplace_back(std::move(item.first));
  // 更新每个解码结果的分数
  likelihood_.emplace_back(item.second.total_score());
  // 更新每个解码结果的viterbi分数
  viterbi_likelihood_.emplace_back(item.second.viterbi_score());
  // 更新每个解码结果的时间戳信息
  times_.emplace_back(item.second.times());
}
```

> [CTC的Decode算法-Prefix Beam Search](http://placebokkk.github.io/asr/2020/02/01/asr-ctc-decoder.html)
> [First-Pass Large Vocabulary Continuous Speech Recognition using Bi-Directional Recurrent DNNs](https://arxiv.org/abs/1408.2873)
> [Distill-Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
> [WeNet 更新：支持字级时间戳和 N-best](https://mp.weixin.qq.com/s/YD2s4Wd9K4T_Bjm_Uha96Q)