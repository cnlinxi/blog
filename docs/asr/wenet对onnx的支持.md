# wenet对onnx的支持

## Pytorch转ONNX

Pytorch转ONNX实际上是将模型的每一个op转化为ONNX定义的某一个算子，比如对于Pytorch中的`nn.Upsample()`和`F.interpolate()`，在转换为ONNX后最终都会成为ONNX的`Resize`算子。通过修改继承自`torch.autograd.Function`算子的`symbolic`方法，可以改变该算子映射到ONNX算子的行为。

> [模型部署入门教程（二）：解决模型部署中的难题 - 知乎](https://zhuanlan.zhihu.com/p/479290520)
> [torch.onnx — PyTorch master documentation](https://pytorch.org/docs/master/onnx.html)

Pytorch转ONNX格式的`torch.onnx.export()`函数需要`torch.jit.ScriptModule`，而不是`torch.nn.Module`，如果传入的模型不是`ScriptModule`形式，该函数会利用tracing方式，追踪流入tensor的流向，来记录模型运算时的所有操作并转为ScriptModule：

![](attachments/Pasted%20image%2020220602155510.png)

跟踪法只能通过实际运行一遍模型的方法导出模型的静态图，无法识别出模型中的控制流（如循环）和运行时的动态变化；记录法则能通过解析模型来正确记录所有的控制流。

> [模型部署入门教程（三）：PyTorch 转 ONNX 详解 - 知乎](https://zhuanlan.zhihu.com/p/498425043)

tracing方式的转换会导致模型无法对动态的操作进行捕获，比如对torch.tensor的动态切片操作会被当做固定的长度切片，一旦切片的长度发生变化就会触发错误。为了对这些动态操作进行保存，可以使用scripting的方式，直接将动态操作流改写为ScriptModule。

## 导出ONNX

### torch.onnx.export

```python

# wenet/wenet/bin/export_onnx_cpu.py
encoder = asr_model.encoder
...
inputs = (chunk, offset, required_cache_size,
              att_cache, cnn_cache, att_mask)
...
encoder_outpath = os.path.join(args['output_dir'], 'encoder.onnx')
...
dynamic_axes = {
	'chunk': {1: 'T'}, # chunk张量在axis=1上是可变的，该axis=1维度名为T
	'att_cache': {2: 'T_CACHE'},
	'att_mask': {2: 'T_ADD_T_CACHE'},
	'output': {1: 'T'},
	'r_att_cache': {2: 'T_CACHE'},
}
...
torch.onnx.export(
	encoder, # model (torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction)
	inputs,# 模型输入参数。导出时可以构建随机等大的张量作为输入参数，用于模型的张量跟踪
	encoder_outpath,  # 导出onnx文件的路径
	opset_version=13,  # 转换时参考哪个ONNX算子集版本
	export_params=True, # 是否导出模型的参数
	do_constant_folding=True, # 常数折叠优化，对仅输出常数的op直接用常数代替
	input_names=[
		'chunk', 'offset', 'required_cache_size',
		'att_cache', 'cnn_cache', 'att_mask'
	], # 分配给计算图输入节点的名称，需要和`inputs`顺序一致
	output_names=['output', 'r_att_cache', 'r_cnn_cache'], # 分配给计算图输出节点的名称，有序
	dynamic_axes=dynamic_axes, # 指定张量的可变维度
	verbose=False) # 打印输出模型的描述
```

`torch.onnx.export()`中`dynamic_axes`参数可以指定一些张量的可变维度，形式为`dict<string, dict<python:int, string>> or dict<string, list(int)>, default empty dict`。默认情况下，导出的模型将所有输入输出张量的大小均设置为给定张量的大小，为了指定张量的一些维度是动态可变的，可以设置`dynamic_axes`，其中：

- `KEY(str)`：大小可变的输入/输出张量名，张量名需要在`input_names`和`output_names`中。
- `VALUE (dict or list)`：如果是`dict`，`key`是可变大小对应的维度，`value`是对应维度名；如果是`list`，每个元素表示可变大小对应的维度。

参数`opset_version`指定的ONNX算子集版本可参考[onnx/Operators.md at main · onnx/onnx · GitHub](https://github.com/onnx/onnx/blob/main/docs/Operators.md)。在Pytorch中，和ONNX有关的定义存放在[pytorch/torch/onnx at master · pytorch/pytorch · GitHub](https://github.com/pytorch/pytorch/tree/master/torch/onnx)：

![](attachments/Pasted%20image%2020220602161512.png)

其中，`symbolic_opset{n}.py`（符号表文件）表示 Pytorch 在支持第`n`版 ONNX 算子集时新加入的内容，可以在该目录下查找Pytorch到ONNX算子的映射。

在实际应用时可以在`torch.onnx.export()`的`opset_version`中先预设一个版本号，碰到问题就去对应的Pytorch符号表文件里去查。如果某算子确实不存在，或者算子的映射关系不满足要求，就可能需要利用其它算子绕过去，或者自定义算子。

> [onnx/Operators.md at main · onnx/onnx · GitHub](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
> [模型部署入门教程（三）：PyTorch 转 ONNX 详解 - 知乎](https://zhuanlan.zhihu.com/p/498425043)

### NoneType类型参数

在流式解码时，刚开始的若干chunk中编码器的cache为空。ONNX转写的模型不支持NoneType输入，但Torch和ONNX均可以接受维度中存在0的tensor，且可以对这种tensor进行常规的切片和拼接操作，比如：

```python

a = torch.ones((1, 2, 0, 4))  # 维度中存在0的tensor
b = torch.ones((1, 2, 3, 4))
c = torch.cat((a, b), dim=2)
torch.equal(b, c)        # True
d = torch.split(a, 2, dim=-1)
torch.equal(d[0], d[1])  # True
```

因此可以引入长度为0、元素值为0的dummy张量代替NoneType。

### cache中的动态变化

`torch.onnx.export()`利用tracing方式追踪tensor流向时，无法捕获动态操作。在编码器更新缓存时，需要利用`next_cache_start`对上一时刻的计算产物进行切片：

```python

# wenet/wenet/transformer/encoder.py
# required_cache_size < 0（16 chunksize / -1 leftchunks）
if required_cache_size < 0:
    # 该分支下，next_cache_start始终为0
	next_cache_start = 0
# required_cache_size == 0（16 chunksize / 0 leftchunks）
elif required_cache_size == 0:
    # 该分支下，next_cache_start始终为attention_key_size
    # 而attention_key_size是超参数计算出来的定值
	next_cache_start = attention_key_size
# required_cache_size > 0（16 chunksize / 4 leftchunks）
else:
    # 该分支下，next_cache_start动态变化
	next_cache_start = max(attention_key_size - required_cache_size, 0)
```

在16/-1和16/0的解码配置下，不会产生动态操作。但是在16/4的解码配置下，如果对第一个chunk送入长度为0的cache，那么前4个chunk的`next_cache_start`均为0，而对第5个及其之后的chunk，由于`next_cache_start`将变为`attention_key_size - required_cache_size`，计算得到的`next_cache_start`不再是0，这就是所谓的动态变化。

`att_cache`缓存多头注意力的key和value，`next_cache_start`表示下一个`att_cache`在时间维度上起始点：

```python

# wenet/wenet/transformer/encoder.py

# new_att_cache是计算完成多头注意力但尚未利用new_cache_start进行切片并更新的注意力缓存
# 所谓的注意力缓存`att_cache`实际上就是缓存上一个chunk中的多头注意力的key和value
# new_att_cache是对key和value进行concat: new_att_cache=torch.cat((k, v),dim=-1)
# 因此new_att_cache的最后一个维度需要乘以2
# shape(new_att_cache) is (1, head, attention_key_size, d_k * 2)
# attention_key_size = cache_t1 + chunk_size
# cache_t1 = required_cache_size = chunk_size * num_decoding_left_chunks

# So, shape(new_att_cache[:, :, next_cache_start:, :]) in 16/4
# always be (1, head, chunk_size * num_decoding_left_chunks, d_k * 2)
r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
```

为了保证在16/4配置下`next_cache_start`的值在推理的全过程中维持不变，对第一个chunk 送入长度为`required_cache_size`而非长度为0的cache。换句话说，从第一个 chunk 开始就送入“真实”的 cache，只是该cache的元素值均为0，并利用`att_mask`指示该部分的cache为False，此时`next_cache_start == attention_key_size - required_cache_size`恒成立，也即对于第一个及之后的chunk，`next_cache_start == chunk_size`恒成立。

### 动态切片

如果使用tracing方式转写成ONNX，对`torch.tensor`的切片只能是静态切片，比如`data[：3] = new_data`，这里的3只能是固定值3，不能是传入的tensor，比如`data[:data.shape[0]]`在ONNX的opset<13时是不支持的。

可以依靠传入的`torch.tensor`作为index，实现tracing方式下对张量的动态切片，比如`data[torch.tensor([1,2])]`。WeNet流式解码时，每个时刻都需要编码器对输入的cache进行切片，每次均传入切片index会将模型变得复杂。此时将需要动态切片的操作通过scripting方式直接改写为ScriptModule是更优策略，比如：

```python

@torch.jit.script
def slice_helper(x, offset):
    return x[:, -offset: , : ]

chunk = x.size(1) - output_cache.size(1)

# x_q = x[:, -chunk:, :]
# residual = residual[:, -chunk:, :]
# mask = mask[:, -chunk:, :]
# 更改为：
x_q = slice_helper(x, chunk)
residual = slice_helper(residual, chunk)
mask = slice_helper(mask, chunk)
```

需要注意的是，如果将`torch.nn.Module`转为`torch.jit.ScriptModule`，模型无法进行训练，此时可以将训练代码和转写代码分为两部分，实际上也可以简单地在使用到scripting的模块中，添加bool属性onnx_mode，在训练时设置为False，转写时设置为True即可：

```python

@torch.jit.script
def slice_helper(x, offset):
    return x[:, -offset: , : ]

chunk = x.size(1) - output_cache.size(1)

if onnx_mode:
    x_q = slice_helper(x, chunk)
    residual = slice_helper(residual, chunk)
    mask = slice_helper(mask, chunk)
else:
    x_q = x[:, -chunk:, :]
    residual = residual[:, -chunk:, :]
    mask = mask[:, -chunk:, :]
```

当然，opset>=13 时，ONNX已经直接支持上述的动态切片操作：

```python

# 在导出时，将opset设置为13，即可直接支持动态切片，无需任何代码层面的改动
torch.onnx.export(
	encoder, inputs, encoder_outpath, opset_version=13,
	export_params=True, do_constant_folding=True,
	input_names=[
		'chunk', 'offset', 'required_cache_size',
		'att_cache', 'cnn_cache', 'att_mask'
	],
	output_names=['output', 'r_att_cache', 'r_cnn_cache'],
	dynamic_axes=dynamic_axes, verbose=False)
```

### tracing只追踪tensor

tracing方式只能通过追踪`tensor`流向来定位参与的运算，而无法追踪其它类型比如`List[tensor]`。因此encoder模块中的`forward_chunk()`函数各个层的cache不能使用`list`来保存，而必须通过`torch.cat()`函数合并成tensor，否则在调用ONNX模型时，对模型输出的索引将会出错。比如：

```python

r_conformer_cnn_cache.append(new_cnn_cache)
```

输出对应索引位置的值，不是`r_conformer_cnn_cache`，而是`r_conformer_cnn_cache[0]`。因此应改为：

```python

r_conformer_cnn_cache = torch.cat((r_conformer_cnn_cache, new_cnn_cache.unsqueeze(0)), 0)
```

### ONNX不支持pad_sequence

重新设计了一个与`pad_sequence()`等价且能被ONNX感知到shape变化的函数。

```python

# https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/asr_model.py#L683-L721
# `pad_sequence` is not supported by ONNX, it is used
#   in `reverse_pad_list` thus we have to refine the below code.
#   Issue: https://github.com/wenet-e2e/wenet/issues/1113
# Equal to:
#   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
#   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
max_len = torch.max(r_hyps_lens)
index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
seq_len_expand = r_hyps_lens.unsqueeze(1)
seq_mask = seq_len_expand > index_range  # (beam, max_len)
#   >>> seq_mask
#   >>> tensor([[ True,  True,  True],
#   >>>         [ True,  True,  True],
#   >>>         [ True, False, False]])
index = (seq_len_expand - 1) - index_range  # (beam, max_len)
#   >>> index
#   >>> tensor([[ 2,  1,  0],
#   >>>         [ 2,  1,  0],
#   >>>         [ 0, -1, -2]])
index = index * seq_mask
#   >>> index
#   >>> tensor([[2, 1, 0],
#   >>>         [2, 1, 0],
#   >>>         [0, 0, 0]])
r_hyps = torch.gather(r_hyps, 1, index)
#   >>> r_hyps
#   >>> tensor([[3, 2, 1],
#   >>>         [4, 8, 9],
#   >>>         [2, 2, 2]])
r_hyps = torch.where(seq_mask, r_hyps, self.eos)
#   >>> r_hyps
#   >>> tensor([[3, 2, 1],
#   >>>         [4, 8, 9],
#   >>>         [2, eos, eos]])
r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
#   >>> r_hyps
#   >>> tensor([[sos, 3, 2, 1],
#   >>>         [sos, 4, 8, 9],
#   >>>         [sos, 2, eos, eos]])
```

### 超参数读写

通过ONNX的metadata接口，实现将超参数全部存入ONNX模型。

```python

# wenet/wenet/bin/export_onnx_cpu.py
# 写入（python版本）
onnx_encoder = onnx.load(encoder_outpath)
for (k, v) in args.items():
    meta = onnx_encoder.metadata_props.add()
    meta.key, meta.value = str(k), str(v)
onnx.save(onnx_encoder, encoder_outpath)

# python版本的读取
ort_session = onnxruntime.InferenceSession(encoder_outpath)
meta = ort_session.get_modelmeta()
print("\t\tcustom_metadata_map={}".format(meta.custom_metadata_map))

// cpp版本的读取
// wenet/runtime/core/decoder/onnx_asr_model.cc
auto model_metadata = encoder_session_->GetModelMetadata();
Ort::AllocatorWithDefaultOptions allocator;
encoder_output_size_ = std::move(
  atoi(model_metadata.LookupCustomMetadataMap("output_size", allocator)));
```

### 其它

- 通过tracing追踪模型，如果模型传入的tensor没有被使用，导出的模型将会认为不会输入该参数，如果后续输入该参数将会导致报错。
- ONNX不支持tensor转bool的操作，训练时python脚本中大量的assert将无法使用。

## ONNX推理

[ONNX Runtime](https://onnxruntime.ai/docs/)是由微软维护的跨平台机器学习推理加速器，也即”推理引擎“，可实现Pytorch->ONNX->ONNX Runtime这条部署流水线。WeNet ONNX推理流程为：加载模型和超参数->初始化cache->encoder推理->CTC推理->attention rescoring推理。

`onnx_asr_model`和`torch_asr_model`均继承自`asr_model`，`asr_model`中定义了`Reset()`、`ForwardEncoderFunc()`和`AttentionRescoring()`三个纯虚函数：

- `Reset()`实现了`offset_`、`att_cache_`等cache的初始化。
- `ForwardEncoderFunc()`包含了encoder和CTC推理。
- `AttentionRescoring()`对识别结果做重打分。

[C++ ONNX Runtime APIs]均定义在`Ort`命名空间下。

### ONNX线程数的配置

ONNX默认采用多核加速解码，设置ONNX线程数的代码为：

```cpp

// wenet/runtime/core/decoder/onnx_asr_model.cc
Ort::SessionOptions OnnxAsrModel::session_options_ = Ort::SessionOptions();
...
// 会话线程数的配置
session_options_.SetIntraOpNumThreads(num_threads);
session_options_.SetInterOpNumThreads(num_threads);
...
// 使用配置启动会话
encoder_session_ = std::make_shared<Ort::Session>(
	env_, encoder_onnx_path.c_str(), session_options_);
```

### 编码器的入参个数不确定

由于导出ONNX时，存在不同的`chunk_size/num_decoding_left_chunks`配置，此时ONNX会自动优化掉无用参数，这将导致模型`encoder.onnx`的入参不一样。具体来说，当使用`16/-1`、`-1/-1`和`16/0`时，`next_cache_start`将会被ONNX硬编码为0或`chunk_size`，因此不再需要`required_cache_size`和`att_mask`，这两个参数也将会被ONNX自动移除。

由于编码器的入参会发生变化，对于`encoder.onnx`，会先获取输入参数名列表，在准备编码器的输入时，根据参数名列表，挑选相应变量作为输入。而对于编码器的输出、CTC和解码器的输入和输出，也全部采用从模型读取参数名列表的方式，避免手工定义参数名列表：

```cpp

// wenet/runtime/core/decoder/onnx_asr_model.cc
//根据encoder_in_names_准备输入
std::vector<Ort::Value> inputs;
for (auto name : encoder_in_names_) {
if (!strcmp(name, "chunk")) {
  inputs.emplace_back(std::move(feats_ort));
} else if (!strcmp(name, "offset")) {
  inputs.emplace_back(std::move(offset_ort));
} else if (!strcmp(name, "required_cache_size")) {
  inputs.emplace_back(std::move(required_cache_size_ort));
} else if (!strcmp(name, "att_cache")) {
  inputs.emplace_back(std::move(att_cache_ort_));
} else if (!strcmp(name, "cnn_cache")) {
  inputs.emplace_back(std::move(cnn_cache_ort_));
} else if (!strcmp(name, "att_mask")) {
  inputs.emplace_back(std::move(att_mask_ort));
}
}
```

### int类型参数

在runtime阶段，构造int类型的张量需要进行特殊处理。创建张量的`CreateTensor()`函数签名为：

```

static Value Ort::Value::CreateTensor(const OrtMemoryInfo * info,
									T * p_data,
									size_t 	p_data_element_count,
									const int64_t * shape,
									size_t 	shape_len)
```

其中：

- `info`：用户缓冲区所在的内存描述，比如CPU或GPU。
- `p_data`：指向用户提供的缓冲区指针。
- `p_data_element_count`：用户缓冲区的元素个数。
- `shape`：用户缓冲区的张量大小。
- `shape_len`：张量大小`shape`的维度个数。

在构造int类型的张量时，`CreateTensor()`函数里`shape`和`shape_len`两个形参应分别传入空指针和0：

```cpp

// wenet/runtime/core/decoder/onnx_asr_model.cc
// 一般张量的构造
// chunk
const int64_t feats_shape[3] = {1, num_frames, feature_dim};
  Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
      memory_info, feats.data(), feats.size(), feats_shape, 3);
// int类型张量的构造
// offset
int64_t offset_int64 = static_cast<int64_t>(offset_);
Ort::Value offset_ort = Ort::Value::CreateTensor<int64_t>(
  memory_info, &offset_int64, 1, std::vector<int64_t>{}.data(), 0);
```


### Ort环境变量

Ort环境变量保存着其它对象使用的日志状态，必须在使用ONNXRuntime的其它函数之前创建好环境变量，跨线程共享环境，并且应将其设置为全局变量。

```cpp

// sessions
// NOTE(Mddct): The Env holds the logging state used by all other objects.
//  One Env must be created before using any other Onnxruntime functionality.
static Ort::Env env_;  // shared environment across threads.
static Ort::SessionOptions session_options_;
std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
std::shared_ptr<Ort::Session> rescore_session_ = nullptr;
std::shared_ptr<Ort::Session> ctc_session_ = nullptr;
```

### 全局变量和局部变量

用于构造`att_cache_ort_`的`att_cache_`应设置为全局变量，这是因为ONNXRuntime在构建张量`att_cache_ort_`时不会对`att_cache_`里面的数据进行拷贝，而只是维护了指向`att_cache_`的指针，如果在`Reset()`函数中将`att_cache_`声明为局部变量，并用于构造`att_cache_ort_`，在识别时会出现运行时突然崩溃的现象，主要原因是`att_cache_`作为局部变量，内存会被系统回收，`cnn_cache_`声明为全局变量的原因类似。

而`att_mask_ort`需要设置成局部变量主要有三个原因：
- `att_mask_ort`需根据`offset_`动态设置元素的值。
- 构造编码器的输入时会通过`std::move`把`att_mask_ort`清空。
- `Reset()`函数不需对`att_mask_ort`进行初始化。

### 常用函数

1.  `Ort::AllocatorWithDefaultOptions`

内存分配接口，可用于用户自定义内存分配器。在销毁内存分配器之前，必须确保使用该分配器的对象已经全部被销毁。

> [OnnxRuntime: OrtAllocator Struct Reference](https://onnxruntime.ai/docs/api/c/struct_ort_allocator.html)
> [OnnxRuntime: Ort::AllocatorWithDefaultOptions Struct Reference](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_allocator_with_default_options.html)

2.  `session->GetOutputName(i, allocator)`

获取模型输出节点名称。

- 第一个入参`i`类型为int，表示输出节点的序号。
- 第二个入参`allocator`定义内存分配器，可用于用户自定义内存分配器。

3. `std::make_shared<Ort::Session>(env_, encoder_onnx_path.c_str(), session_options_)`

创建会话对象，和`Tensorflow 1.x`类似，只有会话对象才可以执行模型推理。

- `env_`类型为`Ort::Env`，持有所有对象的日志记录状态，在使用任何ONNXRuntime之前必须先创建一个`Ort::Env`。
- `encoder_onnx_path.c_str()`类型为`const char *`，模型路径。
- `session_options_`类型为`Ort::SessionOptions`，用于创建`Session`对象的`Options`对象。

> [OnnxRuntime: Ort::Session Struct Reference](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session.html)

4. `encoder_session_->Run(Ort::RunOptions{nullptr}, encoder_in_names_.data(), inputs.data(), inputs.size(), encoder_out_names_.data(), encoder_out_names_.size())`

会话对象执行模型推理。

- `Ort::RunOptions{nullptr}`，运行配置。
- `encoder_in_names_.data()`类型为`const char *const *`，C风格字符串数组，输入节点名称。
- `inputs.data()`类型为`const T *`，输入数据。
- `encoder_out_names_.data()`类型为`const char *const *`，C风格字符串数组，输出节点名称。
- `encoder_out_names_.size()`类型为`size_t`，输出节点名称的个数。

> [OnnxRuntime: Ort::Session Struct Reference](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session.html#ae1149a662d2a2e218e5740672bbf0ebe)

5. `ctc_ort_outputs[0].GetTensorMutableData<float>();`

获取张量内部原始数据的指针，用于直接读取、写入、修改张量的数据，返回的指针在张量销毁前均有效。

### Attention Rescore原理

将CTC解码结果作为目标值，送入解码器中进行计算，解码器输出正向和逆向的softmax得分，作为正向和逆向解码器的`AttentionScore`，计算得到最终的`rescoring_score`。

```python

# wenet/wenet/bin/export_onnx_cpu.py::export_decoder
# 将解码器的forward()函数替换为对应torch.jit版本的forward_attention_decoder()
decoder.forward = decoder.forward_attention_decoder

# wenet/wenet/transformer/asr_model.py::forward_attention_decoder()
# 将解码器的log_softmax结果作为score输出
decoder_out, r_decoder_out, _ = self.decoder(
	encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
	reverse_weight)  # (num_hyps, max_hyps_len, vocab_size)
decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
# right to left decoder may be not used during decoding process,
# which depends on reverse_weight param.
# r_dccoder_out will be 0.0, if reverse_weight is 0.0
r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
return decoder_out, r_decoder_out

# wenet/wenet/transformer/asr_model.py::BiTransformerDecoder::forward()
# 第三个入参实际是已填充的目标文本序列
def forward(
	self,
	memory: torch.Tensor,
	memory_mask: torch.Tensor,
	ys_in_pad: torch.Tensor,
	ys_in_lens: torch.Tensor,
	r_ys_in_pad: torch.Tensor,
	reverse_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
...
```

runtime调用：

```cpp

// wenet/runtime/core/decoder/onnx_asr_model.cc::OnnxAsrModel::AttentionRescoring()

// 送入解码器作为target的张量实际就是CTC解码结果
Ort::Value hyps_pad_tensor_ = Ort::Value::CreateTensor<int64_t>(
  memory_info, hyps_pad.data(), hyps_pad.size(), hyps_pad_shape, 2);
...
rescore_inputs.emplace_back(std::move(hyps_lens_tensor_));
...
// 启动会话，执行解码器推理出正向和逆向的attention score
std::vector<Ort::Value> rescore_outputs = rescore_session_->Run(
  Ort::RunOptions{nullptr}, rescore_in_names_.data(), rescore_inputs.data(),
  rescore_inputs.size(), rescore_out_names_.data(),
  rescore_out_names_.size());

float* decoder_outs_data = rescore_outputs[0].GetTensorMutableData<float>();
float* r_decoder_outs_data = rescore_outputs[1].GetTensorMutableData<float>();
...
// 利用CTC解码结果和Attention解码器计算出rescoring_score
// combined left-to-right and right-to-left score
(*rescoring_score)[i] =
	score * (1 - reverse_weight) + r_score * reverse_weight;
```


> [作业帮：基于 WeNet + ONNX 的端到端语音识别方案](https://mp.weixin.qq.com/s?__biz=MzU2NjUwMTgxOQ==&mid=2247484139&idx=1&sn=0018045eff55fee866045c42b6af0351&chksm=fcaaca3fcbdd43295b1d2352a0400d0fec3294980ed795551465001cbceb859702c997f9be94&scene=21#wechat_redirect)
> [虎牙在 WeNet 中开源 ONNX 推理支持](https://mp.weixin.qq.com/s?__biz=MzU2NjUwMTgxOQ==&mid=2247484403&idx=1&sn=8dad3f84663f068da939654a28f07d75&chksm=fcaacb27cbdd42317fa38f625c1cc7cf784a8c6f4c9d03195eb0db6e4458d65205f615cbb90b&scene=21#wechat_redirect)
> [论如何优雅地在 WeNet 中支持 ONNX 导出](https://mp.weixin.qq.com/s?__biz=MzU2NjUwMTgxOQ==&mid=2247484423&idx=1&sn=fa909210fe2a275daa1ee059ff6623d6&chksm=fcaaccd3cbdd45c5b087ddfa19a1dafda3ab42f4a77cf127a27cc0b60a2fdce95067aec2c273&scene=90&subscene=93&sessionid=1653659101&clicktime=1653659126&enterid=1653659126&ascene=56&devicetype=android-31&version=28001557&nettype=WIFI&abtest_cookie=AAACAA==&lang=zh_CN&session_us=gh_c55e18da77f8&exportkey=A7yv9afYHfMRum2fauZL86A=&pass_ticket=yHnhkUy5n4qUebqXa2RbI9/noIV9wSEkBdKGWZk8aoa8M0Fu6g+okdDBCkM5ZFgq&wx_header=3)
> [模型部署入门教程（五）：ONNX 模型的修改与调试 - 知乎](https://zhuanlan.zhihu.com/p/516920606)
> [torch.onnx — PyTorch master documentation](https://pytorch.org/docs/master/onnx.html)
> [模型部署入门教程（三）：PyTorch 转 ONNX 详解 - 知乎](https://zhuanlan.zhihu.com/p/498425043)
> [PyTorch (可选）将模型从 PyTorch 导出到 ONNX 并使用 ONNX Runtime 运行_w3cschool](https://www.w3cschool.cn/pytorch/pytorch-fs5q3bsv.html)

tag:: #TODO 