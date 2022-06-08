# WeNet runtime知识点

## 头文件中的define

```cpp

#ifndef DECODER_PARAMS_H_
#define DECODER_PARAMS_H_

#include <memory>
...
#endif  // DECODER_PARAMS_H_
```

这是C++项目中头文件的惯常做法，`#ifndef/#define/#endif`防止重复定义错误。

`#define`预处理指令用于创建宏。指令的一般形式是：

```cpp

#define macro-name replacement-text 
```

所有头文件都应该有`#define`保护来防止头文件被多重包含，并且为了为保证唯一性，命名格式当是：`<PROJECT>_<PATH>_<FILE>_H_` 。当然WeNet没有完全遵从C++代码规范，WeNet采用的头文件命名格式为`<PATH>_H_`。

> [C++风格指南-#define保护](https://zh-google-styleguide.readthedocs.io/en/latest/google-cpp-styleguide/headers/#define)
> [C++语法基础](https://oi-wiki.org/lang/basic/)

## GEMM

```cpp

DEFINE_int32(num_threads,1,"num threads for GEMM");
```

`GEMM`应为Kaldi中矩阵乘法加速库。

> [kaldi Namespace Reference](https://kaldi-asr.org/doc/namespacekaldi.html)

## TLG.fst

|            | 输入                | 输出     |
| ---------- | ------------------- | -------- |
| T(token)   | 帧级别的CTC标签序列 | lexicon建模单元 |
| L(lexicon) | lexicon建模单元            | 词       |
| G(grammer) | 词                  | 词       |

### Token

将帧级别的CTC标签映射到单个lexicon单元。比如，有如下几个标签序列：

- $A\ A\ A\ A\ A$
- $\phi\ \phi\ A\ A\ \phi$
- $\phi\ A\ A\ A\ \phi$

token WFST将这几个标签序列均映射到单个lexicon单元，也就是`A`。搜索图$S$可以表示为：

$$
S=T\circ{\rm min}({\rm det}(L\circ G))
$$

其中，$\circ$表示合并，${\rm min}$表示最小化，${\rm det}$表示确定化。搜索图的输入是帧级别的CTC标签序列，输出是lexicon建模单元。

### Lexicon

将lexicon建模单元序列（字符）映射为词，对于端到端声学模型而言，lexicon WFST就是把lexicon unit（字符）拼成word（词）。

![](attachments/Pasted%20image%2020220510110816.png)

### Grammer

将单词组合成可能的词序列，也就是词级别的语言模型。

![](attachments/Pasted%20image%2020220510110740.png)

> [小白的WFST之路](https://blog.csdn.net/qq_39354864/article/details/117453497)
> [EESEN: End-to-End Speech Recognition using Deep RNN Models and WFST-based Decoding](https://arxiv.org/pdf/1507.08240)

## 解码选项

### `chunk_size`

结构体`DecodeOptions`中的`chunk_size`选项指的是经过下采样之后，解码帧的大小。

### `final_score`

$$
final\_score = rescoring\_weight \times rescoring\_score + ctc\_weight \times ctc\_score
$$

其中，

$$
rescoring\_score = left\_to\_right\_score \times (1 - reverse\_weight) + right\_to\_left\_score \times reverse\_weight
$$

`ctc_score`在下面两种搜索方法中是不同的：

- `CtcPrefixBeamSearch`: $ctc\_score={\rm sum}(prefix)$
- `CtcWfstBeamSearch`: $ctc\_score={\rm max}(viterbi\_path)$

因此要根据搜索方法设置`ctc_weight`。

## WeNet中的哈希函数

在CTC Prefix Beam Search算法中，采用HashMap保存解码过程中产生的前缀串及其对应的分数信息，并利用BK&DR算法求哈希值以减少碰撞。

```cpp

// 利用BK&DR算法求哈希值以减少碰撞
// wenet/runtime/core/decoder/ctc_prefix_beam_search.h
struct PrefixHash {
  size_t operator()(const std::vector<int>& prefix) const {
    size_t hash_code = 0;
    // 此处注释有误，应为BK&DR hash code
    // here we use KB&DR hash code
    for (int id : prefix) {
      hash_code = id + 31 * hash_code;
    }
    return hash_code;
  }
};

// 使用PrefixHash
// wenet/runtime/core/decoder/ctc_prefix_beam_search.cc
std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;
```

> [哈希表之bkdrhash算法解析及扩展](https://blog.csdn.net/wanglx_/article/details/40400693)
> [BKDRhash.c](https://github.com/TempoJiJi/BKDR-hash/blob/master/BKDRhash.c)

## cpp纯虚函数和抽象类

在WeNet中，解码搜索算法都继承自抽象类`SearchInterface`，向外提供统一接口。抽象类`SearchInterface`定义为：

```cpp

class SearchInterface {
 public:
  virtual ~SearchInterface() {}
  virtual void Search(const torch::Tensor& logp) = 0;
  virtual void Reset() = 0;
  virtual void FinalizeSearch() = 0;
  ...
};
```

在C++中，可以将虚函数声明为纯虚函数，语法格式为：

```cpp

virtual 返回值类型 函数名 (函数参数) = 0;
```

包含纯虚函数的类称为抽象类（Abstract Class）。抽象类通常是作为基类，让派生类去实现纯虚函数。**派生类必须实现纯虚函数才能被实例化。**

定义虚函数只是为了允许基类指针调用子类的这个函数。

> [C++纯虚函数和抽象类详解](http://c.biancheng.net/view/2299.html)

## cpp类的特殊成员函数及default和delete特性

### 类的特殊成员函数

C++类有四种特殊成员函数，分别是：

- 默认构造函数：创建新对象。
- 析构函数：销毁对象。
- 拷贝构造函数：创建新对象并利用类对象初始化。
- 拷贝赋值函数：利用类对象对一个已存在的对象赋值。

```cpp

// sample class
class Test {
private:
  int _val;

public:
  Test(int val);                     // 构造函数
  ~Test();                           // 析构函数
  Test(const Test &test);            // 拷贝构造函数
  Test &operator=(const Test &test); // 拷贝赋值函数
};
```

这四种特殊成员函数如果没有显式为一个类定义，而又需要用到该特殊成员函数时，则编译器会隐式为这个类生成一个默认的特殊成员函数，并且隐式创建比显式创建的执行效率高。

#### 构造函数和拷贝构造函数

构造函数和拷贝构造函数都是用来进行类实例化，C++类进行初始化有如下两种方式：

```cpp

Test t1(2);  // 调用实参匹配的构造函数
Test t2 = t1; // 调用拷贝构造函数，利用t1对t2进行初始化
```

其中，调用拷贝构造函数的作用主要是复制对象，调用拷贝构造函数的情况有：

1. 复制对象，并通过函数返回复制后的对象。
2. 一个对象以`值传递`的方式传入函数，此时必定调用拷贝构造函数。
3. 一个对象通过另一个对象进行初始化，比如`Test t2 = t1;`

拷贝时尤其需要注意指针和动态分配的资源，否则很容易造成拷贝变成**浅拷贝**，也就是复制前后的对象都指向同一块内存区域。因此如果类的成员变量中包含指针类型，或者对象存在构造函数中分配的其它资源，此时必须显式定义拷贝构造函数。

#### 拷贝赋值函数

通过定义`拷贝赋值运算符`实现类对象之间的拷贝运算。定义拷贝赋值运算符的通常形式为：

```cpp

classname& operator=(const classname& a)
```

`拷贝构造函数`和`拷贝赋值函数`的不同之处如下：

```cpp

Test t2 = t1; // 拷贝构造函数，即创建时进行初始化
Test t3;
t3 = t1;      // 拷贝赋值函数，因为并不是实例化类对象时进行初始化
```

### C++11特性之default和delete

只需在函数声明后加上`=default;`就可以将该函数声明为`default`函数，编译器将为显式声明`=default;`的函数自动生成函数体，以获得更高的执行效率。

但有些时候，可以通过`=delete;`禁用某些函数，将该函数变得不可调用。`=delete;`不仅可以禁用类内的特殊成员函数，也可以禁用一般函数。

```cpp

// sample class
class Test {
private:
  int _val;

public:
  Test() = default;                  // 定义默认构造函数
  Test(int val);                     // 构造函数
  ~Test();                           // 析构函数
  Test(const Test &test) = delete;   // 不可调用拷贝构造函数，Test t2=t1;将不可用
  Test &operator=(const Test &test); // 拷贝赋值函数
};
```

在C++11标准之前，为了阻止拷贝构造函数和拷贝赋值运算符，可将其声明为`private`来阻止拷贝。如下，由于拷贝构造函数和拷贝赋值运算符都被定义为`private`，这两个函数无法被外部访问，因此也就无法进行类之间的拷贝了：

```cpp

// sample class
class Test {
private:
  int _val;
  Test(const Test &test);            // 拷贝构造函数
  Test &operator=(const Test &test); // 拷贝赋值函数

public:
  Test(int val);                     // 构造函数
  ~Test();                           // 析构函数
};
```

当然，如果使用C++11标准，**希望阻止拷贝的类**推荐使用`=delete`定义自己的拷贝构造函数和拷贝赋值运算符，而不是将其声明为`private`。

> [C++类的特殊成员函数及default/delete特性](https://zhuanlan.zhihu.com/p/77806109)
> [C++:73---C++11标准（类的删除函数：=delete关键字、阻止构造、阻止拷贝、private阻止拷贝控制）](https://blog.csdn.net/qq_41453285/article/details/100606913)

## cpp的智能指针

传统C++中需要使用`new`和`delete`手动申请和释放资源，C++11中引入智能指针，使用引用计数，可“自动”管理资源。智能指针包括`std::unique_ptr`/`std::shared_ptr`/`std::weak_ptr`，使用时需要包括头文件`<memory>`。

### unique_ptr

独占指针，由`unique_ptr`管理的内存，只能被一个对象持有。

unique_ptr只有移动构造函数，因此只能移动（转移内部对象所有权，或称浅拷贝），不能拷贝（深拷贝）。因此`unique_ptr`不支持复制和赋值，只支持移动。

```cpp

auto w = std::make_unique<MyClass>();
auto w2 = w; // 编译错误
auto w2 = std::move(w); // 使用std::move转移对象所有权，执行后w变为nullptr
```

`shared_ptr`需要额外维护引用计数，因此内存占用较高；并且引用计数必须是原子操作，而原子操作性能较低。当符合移动语义时，可以采用`std::move`转移所有权，避免复制，从而提高性能。

### shared_ptr

共享指针，`shared_ptr`能够记录有多少个共享指针指向一个对象，`shared_ptr`内部使用引用计数实现内存的自动管理，每当复制一个`shared_ptr`，引用计数就会加一，当引用计数变为零后就会将对象自动删除。可以通过`use_count()`查看一个对象的引用计数，`get()`获取原始指针，`reset()`减少一个引用计数。

其中，调用`reset()`会使引用计数减1。如果向`reset()`传入类对象，比如`reset(new xxx())`时，智能指针首先生成新对象，然后将就对象的引用计数减1，如果发现引用计数为0时，则析构旧对象，最后将新对象的指针交给智能指针保管。

```cpp

auto w = std::make_shared<MyClass>();
{
    auto w2 = w; // shared_pt允许复制
    cout << w.use_count() << endl;  // 此时，对象w的引用计数为2
}
// 离开作用域，引用计数减一，此时输出为1，当引用计数为0时，delete内存
cout << w.use_count() << endl;
// shared_ptr支持移动，执行后w为nullptr，w3.use_count()等于1
auto w3 = std::move(w);
```

### weak_ptr

`weak_ptr`用于解决`shared_ptr`循环引用的问题。`weak_ptr`不会增加引用计数，因此可以打破`shared_ptr`的循环引用。一般父类持有子类的`shared_ptr`，子类持有父类的`weak_ptr`。

```cpp

class B;
struct A{
    shared_ptr<B> b;
};
struct B{
    weak_ptr<A> a; // 不可以采用shared<A> a; 否则会造成循环引用
};
auto pa = make_shared<A>();
auto pb = make_shared<B>();
// 循环引用时，pa和pb都无法正常释放
pa->b = pb;
pb->a = pa;
```

> [C++ 智能指针的正确使用方式](https://www.cyhone.com/articles/right-way-to-use-cpp-smart-pointer/)
> [第 5 章 智能指针与内存管理](https://changkun.de/modern-cpp/zh-cn/05-pointers/)

## cpp关键字explicit

在C++中，`explicit`关键字用来修饰类的构造函数，阻止隐式类型转换。

```cpp

class MyClass {
public:
  int size_;
  char p_;
  // 构造函数一个参数，且没有使用explicit修饰，默认支持隐式类型转换
  MyClass(int size):size_(size){}
  // 构造函数两个参数，不管是否使用explicit，都无法进行隐式类型转换
  MyClass(int size, const char *p):size_(size),p_(p) {}
};

class MyClass2 {
public:
  int size_;
  // 构造函数一个参数，并使用explicit修饰，阻止隐式类型转换
  MyClass2(int size):size_(size){}
};

// 使用
MyClass c1(10); // 编译通过，显式调用
MyClass c2 = 10; // 编译通过，隐式类型转换
MyClass2 c3 = 10; // 编译失败，explicit阻止MyClass2进行隐式类型转换
```

> [C++ explicit的作用](https://www.cnblogs.com/this-543273659/archive/2011/08/02/2124596.html)
> [详解 c++ 关键字 explicit](https://blog.csdn.net/xiezhongyuan07/article/details/80257420)

## torch.jit加载模型

```cpp

// wenet/runtime/core/decoder/torch_asr_model.cc
torch::jit::script::Module model = torch::jit::load(model_path);
```

## WeNet推断时每次一个样本

WeNet runtime每次输入一个样本进行推理：

```cpp

// wenet/runtime/core/decoder/torch_asr_model.cc
void TorchAsrModel::ForwardEncoderFunc(
    const std::vector<std::vector<float>>& chunk_feats,
    std::vector<std::vector<float>> *out_prob) {
  // 执行编码器chunk级别的一次推断，每次输入一条音频的若干帧组成的chunk
  // 输入参数chunk_feats：[frames,feature_dim]，和缓存cached_feature_共同组成输入
  // 输出参数out_prob：[num_frames,vocab_size]，log softmax之后CTC之前的编码器输出
  
  // 1. Prepare libtorch required data, splice cached_feature_ and chunk_feats
  // The first dimension is for batchsize, which is 1.
  int num_frames = cached_feature_.size() + chunk_feats.size();
  const int feature_dim = chunk_feats[0].size();
  torch::Tensor feats =
      torch::zeros({1, num_frames, feature_dim}, torch::kFloat);
...
}
```

## WeNet对CTC输出N-best进行重打分

```cpp

// wenet/wenet/transformer/decoder.py
def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        r_ys_in_pad: torch.Tensor,
        reverse_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
                olens: (batch, )
        """
        ...

// wenet/runtime/core/decoder/torch_asr_model.cc
float TorchAsrModel::ComputeAttentionScore(const torch::Tensor& prob,
                                           const std::vector<int>& hyp,
                                           int eos) {
  // 对hyp表示的一条解码路径进行打分
  // 分数为Attention解码器输出的，解码路径对应的log softmax概率之和
  // 输入参数prob：[max_text_len,vocab_size]，Attention解码器输出分数
  // 输入参数hyp：[text_len,]，CTC解码结果
  // 输出：Attention解码器对hyp表示的解码路径的打分
  float score = 0.0f;
  auto accessor = prob.accessor<float, 2>();
  for (size_t j = 0; j < hyp.size(); ++j) {
    score += accessor[j][hyp[j]];
  }
  score += accessor[hyp.size()][eos];
  return score;
}

// wenet/runtime/core/decoder/torch_asr_model.cc
void TorchAsrModel::AttentionRescoring(
    const std::vector<std::vector<int>>& hyps,
    float reverse_weight,
    std::vector<float>* rescoring_score) {
    // 对CTC输出的N-best解码结果进行重打分
    // 输入参数hyps: [num_best,text_len]，CTC输出的解码结果，存储token id
    // 输入参数reverse_weight：scaler，如果采用双向解码器才有效，表示逆向解码器权重
    // 输出参数rescoring_score：[num_best,]，对CTC输出N-best的打分
    ...
}
```

## WeNet runtime数据队列

WeNet runtime使用`FeaturePipeline`计算梅尔频谱，并将其送入模型。通常情况下，两个线程执行`FeaturePipeline`：线程A调用`AcceptWaveform()`添加原始音频数据并通过`set_input_finished()`通知输入音频的结束；线程B（解码器线程）调用`Read()`消费声学特征。因此`BlockingQueue`用来确保类`FeaturePipeline`的线程安全。

当队列`feature_queue_`中没有声学特征并且输入尚未结束时，`Read()`将会阻塞。


```cpp

// wenet/runtime/core/frontend/feature_pipline.h
class FeaturePipeline {
 public:
  explicit FeaturePipeline(const FeaturePipelineConfig& config);

  // The feature extraction is done in AcceptWaveform().
  void AcceptWaveform(const float* pcm, const int size);
  void AcceptWaveform(const int16_t* pcm, const int size);

...

  // The caller should call this method when speech input is end.
  // Never call AcceptWaveform() after calling set_input_finished() !
  void set_input_finished();
  bool input_finished() const { return input_finished_; }

...

  // Read #num_frames frame features.
  // Return False if less then #num_frames features are read and the
  // input is finished.
  // Return True if #num_frames features are read.
  // This function is a blocking method when there is no feature
  // in feature_queue_ and the input is not finished.
  bool Read(int num_frames, std::vector<std::vector<float>>* feats);

...
};
```

## WeNet数据队列并发逻辑

```cpp

// wenet/runtime/core/frontend/feature_pipeline.cc
bool FeaturePipeline::ReadOne(std::vector<float>* feat) {
  if (!feature_queue_.Empty()) {
    // 最普遍的情形，数据队列不为空，直接从队列中取出一帧声学特征返回
    *feat = std::move(feature_queue_.Pop());
    return true;
  } else {
    // 数据队列为空，加锁挂起本线程，等待数据存入队列
    std::unique_lock<std::mutex> lock(mutex_);
    while (!input_finished_) {
      // 此时输入尚未结束，释放互斥锁，等待condition_variable通知并唤醒本线程
      // This will release the lock and wait for notify_one()
      // from AcceptWaveform() or set_input_finished()
      finish_condition_.wait(lock);
      // 本线程被唤醒，查看此时数据队列是否为空
      if (!feature_queue_.Empty()) {
        // 不为空则从队列中取出一帧声学特征并返回
        *feat = std::move(feature_queue_.Pop());
        return true;
      }
    }
    CHECK(input_finished_);
    // 原始代码如下：
    
    // CHECK(feature_queue_.Empty());
    // return false;
    
    // 上述代码在如下时序时会发生问题：
    // 1. 读线程判断feature_queue_.Empty()为true，进入else，读线程挂起
    // 2. 写线程执行AcceptWaveform并设置input_finished_为true
    // 3. 读线程开始执行else部分，此时input_finished_为true，因此不会执行while循环体，
    // 此时CHECK(feature_queue_.Empty())失败
    // 也就是写线程写入最后一帧，并将input_finished_设置为true，此时队列实际还有数据，
    // 但读线程根据input_finished_判断此时已经结束，直接无法进入while
    // 因此不能CHECK(feature_queue_.Empty())，并且double check数据队列是否为空
    // Double check queue.empty, see issue#893 for detailed discussions.
    // https://github.com/wenet-e2e/wenet/issues/893
    if (!feature_queue_.Empty()) {
      *feat = std::move(feature_queue_.Pop());
      return true;
    } else {
      return false;
    }
  }
}
```

## 解码配置项

```cpp

struct DecodeOptions {
  // chunk_size为经过下采样之后的解码帧帧数，而非原始音频帧
  int chunk_size = 16;
  // 左侧全视野
  int num_left_chunks = -1;

  // final_score = rescoring_weight * rescoring_score + ctc_weight * ctc_score;
  // rescoring_score = left_to_right_score * (1 - reverse_weight) +
  // right_to_left_score * reverse_weight
  // ctc_score在不同的搜索方式中含义不同：
  // CtcPrefixBeamSearch: ctc_score = sum(prefix) score + context score
  // CtcWfstBeamSearch: ctc_score = a max(viterbi) path score + context score
  // 因此需要根据搜索方式设置ctc_weight
  float ctc_weight = 0.5;
  float rescoring_weight = 1.0;
  float reverse_weight = 0.0;
  CtcEndpointConfig ctc_endpoint_config;
  CtcPrefixBeamSearchOptions ctc_prefix_search_opts;
  CtcWfstBeamSearchOptions ctc_wfst_search_opts;
};
```

## wenet词表

```cpp

class AsrDecoder {
...
 private:
   // 输出词表
  // output symbol table
  std::shared_ptr<fst::SymbolTable> symbol_table_;
  // 端到端声学模型词表
  // e2e unit symbol table
  std::shared_ptr<fst::SymbolTable> unit_table_ = nullptr;
...
};
```

## wenet的上下文偏置（context biasing）

在ASR的实际应用中，常用词的识别效果较好，但对于一些特殊的词，识别精度可能会降低。上下文偏差（Context Biasing）是指在推理过程中将先验知识注入ASR，例如用户喜欢的歌曲、联系人、应用程序或位置。传统的ASR通过从偏置短语列表中构建一个n-gram有限状态转录机（Finite State Transducer，FST）来进行上下文偏置，该偏置短语列表在解码过程中与解码图动态组合，这有助于将识别结果偏向于上下文有限状态转录机中包含的n-gram，从而提高特定场景中的识别准确性。

无论是CTC Prefix Beam Search还是CTC WFST Beam Search都可以引入上下文偏置（context biasing），以便适用特殊场景：

```cpp

// wenet/runtime/core/decoder/asr_decoder.cc
if (nullptr == fst_) {
searcher_.reset(new CtcPrefixBeamSearch(opts.ctc_prefix_search_opts,
										resource->context_graph));
} else {
searcher_.reset(new CtcWfstBeamSearch(*fst_, opts.ctc_wfst_search_opts,
									  resource->context_graph));
```

注意，上下文偏置不同于语言模型，只不过偏置短语列表和语言模型都是用有限状态转录机（FST）实现，语言模型只适用于CTC WFST Beam Search。

> [wenet/context.md at main · wenet-e2e/wenet · GitHub](https://github.com/wenet-e2e/wenet/blob/main/docs/context.md)

## WeNet时间戳的后处理策略（”wenet的ctc_prefix_beam_search“补充，待合并）

为了避免输出的单词时间戳都黏连在一起，引入最小单词间隔，两个词时间戳之间的间隔至少大于该值，默认100ms。

```cpp

// wenet/runtime/core/decoder/asr_decoder.h
const int time_stamp_gap_ = 100;  // timestamp gap between words in a sentence

// wenet/runtime/core/decoder/asr_decoder.cc
// 时间戳仅在输入完毕，准备输出最终结果时产生
// 采用声学模型解码时产生的时间戳，同时此处也需要e2e模型的词表
// TimeStamp is only supported in final result
// TimeStamp of the output of CtcWfstBeamSearch may be inaccurate due to
// various FST operations when building the decoding graph. So here we use
// time stamp of the input(e2e model unit), which is more accurate, and it
// requires the symbol table of the e2e model used in training.
if (unit_table_ != nullptr && finish) {
  const std::vector<int>& input = inputs[i];
  const std::vector<int>& time_stamp = times[i];
  CHECK_EQ(input.size(), time_stamp.size());
  for (size_t j = 0; j < input.size(); j++) {
	std::string word = unit_table_->Find(input[j]);
	int start = time_stamp[j] * frame_shift_in_ms() - time_stamp_gap_ > 0
					? time_stamp[j] * frame_shift_in_ms() - time_stamp_gap_
					: 0;
	if (j > 0) {
	  // 如果本时刻单词与上一个时刻单词的时间间隔小于“最小单词间隔”（time_stamp_gap_），
	  // 则该单词的开始时刻start取上一个单词和本时刻单词的中间时刻
	  start = (time_stamp[j] - time_stamp[j - 1]) * frame_shift_in_ms() <
					  time_stamp_gap_
				  ? (time_stamp[j - 1] + time_stamp[j]) / 2 *
						frame_shift_in_ms()
				  : start;
	}
	int end = time_stamp[j] * frame_shift_in_ms();
	if (j < input.size() - 1) {
	  // 如果本时刻单词与下一时刻单词的时间间隔小于“最小单词间隔”（time_stamp_gap_），
	  // 则该单词的结束时刻end取下一个单词和本时刻单词的中间时刻
	  end = (time_stamp[j + 1] - time_stamp[j]) * frame_shift_in_ms() <
					time_stamp_gap_
				? (time_stamp[j + 1] + time_stamp[j]) / 2 *
					  frame_shift_in_ms()
				: end;
	}
	WordPiece word_piece(word, offset + start, offset + end);
	path.word_pieces.emplace_back(word_piece);
  }
}
```

## AsrModel和AsrDecoder的调用关系

- `AsrModel`管理声学模型、语言模型的资源、配置等，调用编解码器实现推理。
- `AsrDecoder`通过`FeaturePipeline`输入数据，调用`AsrModel`产生解码结果。计算推理耗时，产生最终的时间戳，后处理解码结果等。

`CtcPrefixBeamSearch/CtcPrefixBeamSearch`->`TorchAsrModel/OnnxAsrModel`(`AsrModel`的子类)->`AsrDecoder`

## WeNet对空格的处理

1. 无语言模型的解码。训练时，`_`表示空格，因此直接拼接输出单元并将`_`替换为空格，等同于：`detokenized = ''.join(pieces).replace('_', ' ')`。
2. 有语言模型的解码。此时输出中没有`_`，输出单位为字，因此用空格拼接输出单元，等同于`detokenized = ' '.join(pieces)`。 

最后，WeNet通过后处理策略`PostProcessor`统一去除不需要的空格。

```cpp

// wenet/runtime/core/decoder/asr_decoder.cc
for (size_t j = 0; j < hypothesis.size(); j++) {
  std::string word = symbol_table_->Find(hypothesis[j]);
  // A detailed explanation of this if-else branch can be found in
  // https://github.com/wenet-e2e/wenet/issues/583#issuecomment-907994058
  if (searcher_->Type() == kWfstBeamSearch) {
	path.sentence += (' ' + word);
  } else {
	path.sentence += (word);
  }
}
...
// 后处理策略，目前主要功能是去除不需要的空格。
// example1:  “我 爱 你”==> “我爱你”
// example2: “ i love wenet” ==> “i love wenet”
// example3: “我 爱 wenet very much” ==> “我爱wenet very much”
// example4: “aa ää xx yy” ==> “aa ää xx yy”
if (post_processor_ != nullptr) {
  path.sentence = post_processor_->Process(path.sentence, finish);
}
```

> [Runtime: words containing non-ASCII characters are concatenated without space · Issue #583 · wenet-e2e/wenet · GitHub](https://github.com/wenet-e2e/wenet/issues/583#issuecomment-907994058)
