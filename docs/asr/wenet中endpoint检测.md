# wenet中endpoint检测

断点（Endpoint）检测的任务是确定用户何时结束讲话，这对于实时长语音转写和语音搜索等交互式语音应用十分重要。

## EndPoint原理

Endpoint检测的实现主要有两种思路，一是端到端直接在声学模型中对Endpoint进行建模；二是制定规则，检测到持续静音则认为是Endpoint。

### kaldi-Endpoint原理

kaldi通过规则检测Endpoint，满足以下5条中任意一条，则认为是检测到了Endpoint：

1. 识别出文字之前，检测到5秒的静音
2. 识别出文字之后，检测到2秒的静音
3. 解码出概率较小的final state，且检测到1秒的静音
4. 解码出概率较大的final state，且检测到0.5秒的静音
5. 已经解码了20秒的音频

### WeNet-Endpoint原理

WeNet基于CTC结构实现了Endpoint检测，这种实现易于控制Endpoint超参数，并且不需要训练额外的模型。WeNet将连续的长blank标签看作非语音区域，非语音区域满足一定条件，即可认为检测到Endpoint。WeNet中满足以下3条规则中任意一条，则认为检测到了Endpoint：

1. 识别出文字之前，检测到5秒的静音
2. 识别出文字之后，检测到1秒的静音
3. 已经解码了20秒的音频

## WeNet中Endpoint检测的实现

主要实现位于`wenet/runtime/core/decoder/ctc_endpoint.h`和`wenet/runtime/core/decoder/ctc_endpoint.cc`。

### 参数说明

```cpp

// wenet/runtime/core/decoder/ctc_endpoint.h

struct CtcEndpointRule {
  // `must_decoded_sth`用来区分识别出文字之前/之后两个场景
  // `must_decoded_sth`==false表示适用于识别出文字之前
  bool must_decoded_sth;
  // 最小静音段长度，单位ms
  int min_trailing_silence;
  // 最小句子长度，单位ms
  int min_utterance_length;

...
};

struct CtcEndpointConfig {
  /// 在Endpoint检测中，将解码中的blank统一认为是静音silence
  int blank = 0;                // blank id
  float blank_threshold = 0.8;  // blank threshold to be silence
  /// 可以手动添加更多规则，或者修改规则中的时间
  /// 如果希望取消某一条规则，直接将CtcEndpointRule构造函数中的`min_trailing_silence`
  /// 设置为非常大的值即可
  /// 规则1：识别出文字之前，检测到5秒的静音
  CtcEndpointRule rule1;
  /// 规则2：识别出文字之后，检测到1秒的静音
  CtcEndpointRule rule2;
  /// 规则3：已经解码了20秒的音频
  CtcEndpointRule rule3;

  CtcEndpointConfig()
      : rule1(false, 5000, 0), rule2(true, 1000, 0), rule3(false, 0, 20000) {}
};
```

### 实现

```cpp

static bool RuleActivated(const CtcEndpointRule& rule,
                          const std::string& rule_name, bool decoded_sth,
                          int trailing_silence, int utterance_length) {
  // `decoded_sth`表示是否解码出文字，`decoded_sth`==true表示已经解码出文字
  // 通过`(decoded_sth || !rule.must_decoded_sth)`指示
  // 当前解码状态`decoded_sth`是否适用于该规则
  bool ans = (decoded_sth || !rule.must_decoded_sth) &&
             trailing_silence >= rule.min_trailing_silence &&
             utterance_length >= rule.min_utterance_length;
...
  return ans;
}

bool CtcEndpoint::IsEndpoint(const torch::Tensor& ctc_log_probs,
                             bool decoded_something) {
  // 遍历每一个时间步
  for (int t = 0; t < ctc_log_probs.size(0); ++t) {
    torch::Tensor logp_t = ctc_log_probs[t];
    // 获取当前时间步的blank标签概率
    float blank_prob = expf(logp_t[config_.blank].item<float>());

    // 解码帧数加一
    num_frames_decoded_++;
    // 判断blank标签的概率是否大于阈值，默认0.8
    if (blank_prob > config_.blank_threshold) {
      // 可以认定为Endpoint中所谓的“静音帧”，则尾部blank标签的帧数加一
      num_frames_trailing_blank_++;
    } else {
      // 不是所谓的“静音帧”，尾部blank标签的帧数置0
      num_frames_trailing_blank_ = 0;
    }
  }
...
  if (RuleActivated(config_.rule1, "rule1", decoded_something, trailing_silence,
                    utterance_length))
    return true;
...
  return false;
}
```

## WeNet实时长语音转写

大多数端到端语音识别都假设输入音频已经被适当地切分为短音频，该假设不适用于长语音转写。使用Endpoint检测，在进行实时长语音转写时，检测到Endpoint时就可以对当前候选结果进行重打分，并重置解码状态。然后继续转写后续内容，重复以上步骤。

WeNet使用实时长语音转写，只需要在启动客户端的时候，加上参数`--continuous_decoding=true`即可。

> [WeNet 更新：支持 Endpoint 检测](https://mp.weixin.qq.com/s/Y5c2GzdAO9RT5Es6b7fhaQ)
> [kaldi-endpoint](https://github.com/kaldi-asr/kaldi/blob/6260b27d146e466c7e1e5c60858e8da9fd9c78ae/src/online2/online-endpoint.h#L132-L150)
> [End-to-End Automatic Speech Recognition Integrated with CTC-Based Voice Activity Detection](https://arxiv.org/pdf/2002.00551.pdf)