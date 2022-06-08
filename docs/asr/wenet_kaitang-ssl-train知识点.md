# wenet_kaitang-ssl-train知识点

## Python 相关环境变量设置

### PYTHONIOENCODING

设置Python输入输出的编码格式为UTF-8，防止在一些情况下发生字符编解码失败。
```shell

# use utf-8 in python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
```

### PYTHONPATH
```shell

export PATHONPATH=../../../:$PATHONPATH
```

设置Python查找模块的路径。一般来说，有两种途径：

1. PYTHONPATH
```shell

export PATHONPATH=...:$PATHONPATH
```

该方法设置的Python模块路径，在窗口关闭之后失效。

2. sys.path.append('...')

```python

import sys;sys.path.append('...')
```

该方法设置的模块路径，在Python解释器退出之后失效。相比上一种，生命周期更少一些。

> https://blog.csdn.net/qq_40586364/article/details/103466499

## shell中的逻辑运算符

```shell
. ./path.sh || exit 1;
```

### |运算符
```shell
command1 | command2
```
把第一个命令command1执行的结果作为command2的输入传给command2。
例如：
```shell
ls -s|sort -nr
```

### &&运算符

```shell

command1  && command2
```

&&左边的命令（命令1）返回真(即返回0，成功被执行）后，&&右边的命令（命令2）才能够被执行；换句话说，“如果这个命令执行成功&&那么执行这个命令”。
### ||运算符
```shell

command1 || command2
```
||则与&&相反。如果||左边的命令（command1）未执行成功，那么就执行||右边的命令（command2）；或者换句话说，“如果这个命令执行失败了||那么就执行这个命令。

> https://www.cnblogs.com/aaronlinux/p/8340281.html
> https://www.runoob.com/linux/linux-shell-basic-operators.html

## shell删除文本文件中的空格
!!! TODO

```shell
# Remove the space in text

paste -d " " <(cut -f 1 -d" " data/${x}/text.org) <(cut -f 2- -d" " data/${x}/text.org | tr -d " ") \
> data/${x}/text
```

## shell命令作为变量
```shell

mkdir -p $(dirname $dict)
```

利用`$(...)`可以执行命令，并将命令执行结果作为变量。

## shell获取绝对地址
```shell

tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
--num_threads 16 data/$x/wav.scp data/$x/text \
$(realpath data/$x/shards) data/$x/data.list
```
可以通过`realpath`获取绝对地址。
> https://baike.baidu.com/item/realpath/2895213

## shell查找符号链接指向位置
```shell
init_method=file://$(readlink -f $INIT_FILE)
```

`readlink`找出符号链接所指向的位置。选项`-f`表示一直跟随符号链接，直到直到非符号链接的文件位置，限制是最后必须存在一个非符号链接的文件。

## 文本处理awk
```shell
tools/text2token.py -s 1 -n 1 data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
...
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
...
etc.
```

- NF列数
- NR行数
```shell
[root@localhost test]# awk '{print "字段数：" NF}' test
字段数：4
字段数：4
字段数：3
字段数：4
字段数：4
[root@localhost test]# cat test
a aa aaa 1
b bb bbb 2
c cc ccc
d dd ddd 4
e ee eee 5

[root@localhost test]# awk '{print "行号为：" NR}' test
行号为：1
行号为：2
行号为：3
行号为：4
行号为：5
[root@localhost test]# cat test
a aa aaa 1
b bb bbb 2
c cc ccc
d dd ddd 4
e ee eee 5
```

- FNR 读取行号，从1开始，与`NR`不同，新的文件重新从1开始计数
- -F制定分隔符
```shell

awk -F ':|,' '{print $2, $4, $6}' log
```
上述命令中，指定冒号(:)和逗号(,)同时作为分隔符。

> https://blog.csdn.net/tabactivity/article/details/111474085
> https://blog.csdn.net/weixin_36213081/article/details/112127488

## shell字符串操作
```shell

${bpemodel:+--bpe_model ${bpemodel}.model}
```

`:+`如果非空，则返回`:+`后面的值：
```shell
${file:+my.file.txt} ：若$file为非空值，则使用my.file.txt作传回值。
```
在上述的例子中，如果`bpemodel=`则该变量为空，如果`bpemodel=xxx`，则该变量为`--bpe_model xxx.model`。
> [shell中 ${}, ##, %%, :-，:+, ? 的使用](https://blog.csdn.net/XFH1207/article/details/107358803/)

## shell中的脚本配置选项`set -euxo pipefail`
1. `set -e`：脚本出现错误时马上退出，后续命令不再执行。
2. `set -u`：所有未赋值的变量均视为错误。
3. `set -o pipefail`：默认情况下Bash只会检查管道（pipeline）操作最后一个命令的返回值，假如最右边的命令成功那么它就认为这个语句正确。`set -o pipefail`表示在管道连接的命令中，只要有任何一个命令失败（返回值非0），则整个管道操作被视为失败。只有管道中所有命令都成功执行了这个管道才算成功执行。
4. `set -x`可以让Bash把每个命令在执行之前先打印出来，可以认为这是Bash的Debug开关。

> [Bash 脚本中的 set -euxo pipefail](https://zhuanlan.zhihu.com/p/107135290)

## wenet训练主函数wenet/bin/train.py配置项
1. `data_type`，可选项为`raw`、`shard`，`raw`为存储原始数据，适用于数据量较少，音频时长低于一千小时的场景，`shard`相当于数据打包；适用于数据量较大，音频时长高于一千小时的场景。
2. `gpu`，本地GPU序号
3. `ddp.rank`，分布式训练的全局GPU序号
4. `ddp.world_size`，分布式训练中GPU/进程总数
5. `ddp.dist_backend`，可选项为`nccl`、`gloo`，分布式训练后端
6. `ddp.init_method`，分布式训练的初始化方式，可以为tcp、file等方式
7. `num_workers`，用于数据读取的子进程数
8. `pin_memory`，指定即为`true`，使用数据读取的固定内存缓存区，可加速数据读取，但会增大显存占用
9. `use_amp`，指定即为`true`，启动自动混合精度训练
10. `fp16_grad_sync`，指定即为`true`，启用fp16分布式训练梯度同步
11. `symbol_table`，词典
12. `prefetch`，读取数据预取个数，默认100
13. `enc_init`，初始化编码器的预训练模型
14. `enc_init_mods`，用预训练模型初始化的编码器模块，用逗号`,`隔开

## wenet支持非语言识别
wenet支持非语言标签识别，比如噪音、停顿、笑声等。
1. 这些标签的词典路径由参数`--non_lang_syms`指定，每行一个标签。类似于：
```

{NOISE}\n
{BRK}\n
...
```
2. 标签格式为`{xxx}`或`<xxx>`或`[xxx]`。比如：
- `[xxx]` for swithboard and fisher
- `<xxx>` for wsj and chime4

> https://github.com/wenet-e2e/wenet/pull/819

相关代码：

```python

# wenet/bin/train.py
non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

# wenet/utils/file_utils.py
def read_non_lang_symbols(non_lang_sym_path):
    """read non-linguistic symbol from file.

    The file format is like below:

    {NOISE}\n
    {BRK}\n
    ...


    Args:
        non_lang_sym_path: non-linguistic symbol file path, None means no any
        syms.

    """
    if non_lang_sym_path is None:
        return None
    else:
        syms = read_lists(non_lang_sym_path)
        non_lang_syms_pattern = re.compile(r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        for sym in syms:
            if non_lang_syms_pattern.fullmatch(sym) is None:
                class BadSymbolFormat(Exception):
                    pass
                raise BadSymbolFormat(
                    "Non-linguistic symbols should be "
                    "formatted in {xxx}/<xxx>/[xxx], consider"
                    " modify '%s' to meet the requirment. "
                    "More details can be found in discussions here : "
                    "https://github.com/wenet-e2e/wenet/pull/819" % (sym))
        return syms
```

## wenet中的数据加载
### 分布式采样器

```python

# wenet/dataset/dataset.py

class DistributedSampler:
    def __init__(self, shuffle=True, partition=True):
        self.epoch = -1
        self.update()
        self.shuffle = shuffle
        self.partition = partition
```

上述初始化函数中，
- `shuffle`：每一轮采样之前是否打乱数据，打乱顺序的随机数种子为`self.epoch`
- `partition`：是否分片。如果设置为`True`，则每个GPU分配到不同数据；设置为`False`，则每台机器分配到不同数据：

```python

if self.partition:
	if self.shuffle:
		random.Random(self.epoch).shuffle(data)
	data = data[self.rank::self.world_size]
data = data[self.worker_id::self.num_workers]
```

## torch.Tensor和torch.tensor之间的区别

`torch.Tensor`是所有张量类型的**父类**，所有的张量均是`torch.Tensor`的实例，当执行`torch.Tensor()`时将会返回一个没有任何`data`的空`Tensor`。

而`torch.tensor`是一个利用必须利用`data`构建没有梯度回传历史的张量的**函数**：

```python

torch.tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False) → Tensor
```

> https://pytorch.org/docs/master/tensors.html#torch.Tensor
> https://pytorch.org/docs/master/generated/torch.tensor.html#torch.tensor

## 相对位置编码和相对多头注意力

相比于RNN，Transformer无法从网络结构上对位置信息进行处理。原始Transformer采用绝对位置编码，并且仅仅将位置信息加入到输入层。而相对位置编码解决这一缺陷的方法是**将相对位置编码加入到self-attention内部。**

相对位置编码在计算第$i$和第$j$个元素之间注意力的key和value时，加入$i$和$j$之间的位置编码，并且加入的是$i$和$j$之间的相对位置关系。相对位置编码在自注意力的计算中加入了两个可学习变量$a_{i,j}^V$和$a_{i,j}^K$。

### 原始注意力

原始自注意力上下文向量$z_i$：

$$
z_i=\sum_{j=1}^{n}\alpha_{i,j}(x_jW^V)
$$

其中，$W^V$是可学习张量，权重张量$\alpha_{i,j}$通过softmax计算得到：

$$
\alpha_{i,j}=\frac{{\rm exp}e_{i,j}}{\sum_{k=1}^n{\rm exp}e_{i,k}}
$$

其中，$e_{i,j}$则是利用query和key计算得到：

$$
e_{i,j}=\frac{(x_iW^Q)(x_jW^K)^T}{\sqrt{d_k}}
$$

其中，$W^Q,W^K$为可学习张量。

### 相对注意力

相对注意力上下文向量$z_i$的计算方法改为：

$$
z_i=\sum_{j=1}^n\alpha_{i,j}(x_jW^V+a_{i,j}^V)
$$

注意到上式中，新增的$a_{i,j}^V$为加入的相位位置信息。权重张量$\alpha_{i,j}$的计算方法不变，而$e_{i,j}$的计算方法改为：

$$
e_{i,j}=\frac{(x_iW^Q)(x_jW^K+a_{i,j}^K)^T}{\sqrt{d_k}}
$$

上式中，新增的$a_{i,j}^K$同样是加入的相对位置信息。$a_{i,j}^K$和$a_{i,j}^V$的计算方法相同，也就是在$[-k,k]$的范围内计算相对距离，超出范围的用0或者$k$进行截断：

$$
a_{i,j}^K=w_{{\rm clip}(j-i,k)}^K
$$

$$
a_{i,j}^V=w_{{\rm clip}(j-i,k)}^V
$$

$$
{\rm clip}(x,k)={\rm max}(-k,{\rm min}(k,x))
$$


> [详解Transformer-XL](https://zhuanlan.zhihu.com/p/271984518)
> [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
> [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)

## Depthwise和Pointwise卷积

- Depthwise卷积：一个卷积核负责一个通道，一个通道只被一个卷积核卷积。
- Pointwise卷积：卷积核的尺寸为`1×1×M`，`M`为上一层的通道数。所以这里的卷积运算会将上一步的feature map在深度方向上进行加权组合，生成新的Feature map。

> [Depthwise卷积与Pointwise卷积](https://zhuanlan.zhihu.com/p/80041030)

## 梯度缩放

```python

# wenet/wav2vec/grad_multiply.py
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

# wenet/wav2vec/wav2vec2_model.py
if self.encoder_grad_mult !=1.0:
	encoder_out=GradMultiply.apply(encoder_out,self.encoder_grad_mult)
	encoder_mask=GradMultiply.apply(encoder_mask,self.encoder_grad_mult)
```

对于多头注意力的解码器，可以对编码器的梯度进行缩放，从而稳定训练。

> https://github.com/pytorch/fairseq/issues/13
> [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf)

## wenet解码

一共有4个负责推断的函数：

- `recognize`：attention解码器+beam search（集束搜索）
- `ctc_greedy_search`：ctc+greedy search（贪婪搜索）。greedy search可以认为是`beam size=1`的beam search特例。
- `ctc_prefix_beam_search`：ctc+beam search。
- `attention_rescoring`：注意力重排序解码，首先使用ctc+beam search获得n-best的结果，之后利用attention解码器对n-best进行重排序，以获得更准确的结果。

除此之外，包括若干以`_`开头的内部函数，以及`@torch.jit.export`的C++调用导出接口。

```python

# wenet/wav2vec/wav2vec2_model.py
def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> torch.Tensor:
    pass

def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
    pass

def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:
    pass

def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
    ) -> List[int]:
    pass
```











