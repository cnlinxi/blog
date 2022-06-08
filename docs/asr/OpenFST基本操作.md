# OpenFST基本操作

## 概述

OpenFST是C++模板库，需要包含头文件`<fst/fstlib.h>`，动态链接库`libfst.so`位于安装库目录（installation library directory），所有类和函数位于命名空间`fst`。

OpenFST同样提供了shell接口，位于安装`bin`目录下，可对FST文件进行操作。

## FST示例

下图定义了一个有限状态机（Finite State Transducer，FST）：

![](attachments/Pasted%20image%2020220530101444.png)

上图中，仅有一个初始状态，初始状态为标签0；结束状态为标签2，权重为3.5，任意具有非无穷大权重的状态就是最终状态。上图中状态0到1的弧（`arc`/`transition`）输入标签为`a`，输出标签为`x`，权重为0.5。该FST可以将字符序列“ac”映射为“xz”，对应的权重为$0.5+2.5+3.5=6.5$。

## 创建FST

可以用C++或shell命令创建上述FST示例。如果将标签表示为非负整数，则可以省略符号表文件（symbol table files），这也是该示例FST的内部表示：

![](attachments/Pasted%20image%2020220530102634.png)

### cpp创建FST

```cpp

// 通用可变FST
// A vector FST is a general mutable FST 
StdVectorFst fst;

// 将状态0添加到最初为空的FST，使其成为初始状态
// Adds state 0 to the initially empty FST and make it the start state. 
fst.AddState();   // 1st state will be state 0 (returned by AddState) 
fst.SetStart(0);  // arg is state ID

// 添加状态0的两条弧
// Adds two arcs exiting state 0.
// Arc constructor args: ilabel, olabel, weight, dest state ID. 
fst.AddArc(0, StdArc(1, 1, 0.5, 1));  // 1st arg is src state ID 
fst.AddArc(0, StdArc(2, 2, 1.5, 1)); 

// 添加状态1和它的一条弧
// Adds state 1 and its arc. 
fst.AddState();
fst.AddArc(1, StdArc(3, 3, 2.5, 2));

// 添加状态2并设置最终权重
// Adds state 2 and set its final weight. 
fst.AddState();
fst.SetFinal(2, 3.5);  // 1st arg is state ID, 2nd arg weight

// save this FST to a file
fst.Write("binary.fst");
```

### shell创建FST

为示例FST创建文本文件：

```shell

# arc format: src dest ilabel olabel [weight]
# final state format: state [weight]
# lines may occur in any order except initial state must be first line
# unspecified weights default to 0.0 (for the library-default Weight type) 
$ cat >text.fst <<EOF
0 1 a x .5
0 1 b y 1.5
1 2 c z 2.5
2 3.5
EOF
```

FST的内部表示为整数，因此必须提供符号到整数的映射文件：

```shell

$ cat >isyms.txt <<EOF
<eps> 0
a 1
b 2
c 3
EOF

$ cat >osyms.txt <<EOF
<eps> 0
x 1
y 2
z 3
EOF
```

可以使用任意字符串作为标签，也可以使用任意非负整数作为标签ID。输出符号表中`<eps> 0`表示标签$\epsilon$的标签ID为0，在示例FST中并未用到。

将文本文件转换为FST文件，以便OpenFST的使用：

```shell

# Creates binary Fst from text file. 
# The symbolic labels will be converted into integers using the symbol table files. 
$ fstcompile --isymbols=isyms.txt --osymbols=osyms.txt text.fst binary.fst

# As above but the symbol tables are stored with the FST. 
$ fstcompile --isymbols=isyms.txt --osymbols=osyms.txt --keep_isymbols --keep_osymbols text.fst binary.fst
```

> 关于FST I/O的详细信息参见[FstInputOutput](https://www.openfst.org/twiki/bin/view/FST/FstAdvancedUsage#FstInputOutput)

## 读取FST

### 使用cpp读取FST

一条弧的标准表示：

```cpp

struct StdArc {
 typedef int Label;
 typedef TropicalWeight Weight;  // see "FST Weights" below 
 typedef int StateId; 
 
 Label ilabel;
 Label olabel;
 Weight weight;
 StateId nextstate;
};
```

读取示例：

```cpp

typedef StdArc::StateId StateId;

# Gets the initial state; if == kNoState => empty FST. 
StateId initial_state = fst.Start();

# Get state i's final weight; if == Weight::Zero() => non-final. 
Weight weight = fst.Final(i);

# 迭代FST状态
# Iterates over the FSTs states. 
for (StateIterator<StdFst> siter(fst); !siter.Done(); siter.Next()) 
  StateId state_id = siter.Value();

# 迭代状态i的弧
# Iterates over state i's arcs. 
for (ArcIterator<StdFst> aiter(fst, i); !aiter.Done(); aiter.Next())
  const StdArc &arc = aiter.Value();

# 迭代状态i中输入标签为l的弧
# Iterates over state i's arcs that have input label l (FST must support this
# in the simplest cases,  true when the input labels are sorted). 
Matcher<StdFst> matcher(fst, MATCH_INPUT);
matcher.SetState(i);
if (matcher.Find(l)) 
  for (; !matcher.Done(); matcher.Next())
     const StdArc &arc = matcher.Value();
```

### 通过shell打印、绘制、查看FST

```shell

# 打印FST
# Print FST using symbol table files.
# 如果没有指定符号表，且符号表没有存储到FST，则FST将打印数字标签
$ fstprint --isymbols=isyms.txt --osymbols=osyms.txt binary.fst text.fst

# 绘制FST
# Draw FST using symbol table files and Graphviz dot: 
$ fstdraw --isymbols=isyms.txt --osymbols=osyms.txt binary.fst binary.dot
$ dot -Tps binary.dot >binary.ps

# 查看FST的详细信息
$ fstinfo binary.fst
```

## 操作FST

### cpp操作FST

以下抽象类模板构成FST接口：

- [`Fst<Arc>`](https://www.openfst.org/twiki/bin/view/FST/FstAdvancedUsage#BaseFsts)：支持上述读取操作。
- [`ExpandedFst<Arc>`](https://www.openfst.org/twiki/bin/view/FST/FstAdvancedUsage#ExpandedFsts)：额外支持`NumStates()`的`Fst`。
- [`MutableFst<Arc>`](https://www.openfst.org/twiki/bin/view/FST/FstAdvancedUsage#MutableFsts)：支持各种修改操作比如`AddStates()`和`SetStart()`的`ExpandedFst`。

特别地，非抽象FST包括以下类模板：

- `VectorFst<Arc>`：通用可变FST。
- `ConstFst<Arc>`：通用扩展不可变FST。
- `ComposeFst<Arc>`：两个FST的非扩展、延迟复合（delayed composition）。

在OpenFST中，`StdFst`是`Fst<StdArc>`的typedef，上述类模板均存在类似的typedef。在迭代时，指定具体的FST类作为迭代器模板参数，可以获得更好的性能，比如迭代时优先使用`ArcIterator<StdVectorFst>`，而不是`ArcIterator<StdFst>`。

### shell操作FST

shell操作FST时，通常读取一个或多个FST文件，调用内部cpp操作，最后写入FST文件。如果省略输出文件，将使用标准输出；如果输入文件也省略或者是`-`，将使用标准输入：

```shell

# 一元操作
fstunaryop in.fst out.fst
fstunaryop <in.fst >out.fst
# 二元操作
fstbinaryop in1.fst in2.fst out.fst
fstbinaryop - in2.fst <in1.fst >out.fst
```

## 应用示例

以两个FST复合（Composition）操作为例。

### 使用cpp复合

```cpp

#include <fst/fstlib.h>

namespace fst {
  // Reads in an input FST. 
  StdVectorFst *input = StdVectorFst::Read("input.fst");

  // Reads in the transduction model. 
  StdVectorFst *model = StdVectorFst::Read("model.fst");

  // FST需要按照复合的维度进行排序
  // The FSTs must be sorted along the dimensions they will be joined.
  // In fact, only one needs to be so sorted.
  // This could have instead been done for "model.fst" when it was created. 
  ArcSort(input, StdOLabelCompare());
  ArcSort(model, StdILabelCompare());

  // Container for composition result. 
  StdVectorFst result;

  // Creates the composed FST. 
  Compose(*input, *model, &result);

  // Just keeps the output labels. 
  Project(&result, PROJECT_OUTPUT);

  // Writes the result FST to a file.
  result.Write("result.fst");
}
```

### 使用shell复合

```shell

# FST需要按照复合的维度进行排序
# The FSTs must be sorted along the dimensions they will be joined.
# In fact, only one needs to be so sorted.
# This could have instead been done for "model.fst" when it was created. 
$ fstarcsort --sort_type=olabel input.fst input_sorted.fst
$ fstarcsort --sort_type=ilabel model.fst model_sorted.fst

# Creates the composed FST. 
$ fstcompose input_sorted.fst model_sorted.fst comp.fst

# Just keeps the output label 
$ fstproject --project_output comp.fst result.fst

# Do it all in a single command line. 
$ fstarcsort --sort_type=ilabel model.fst | fstcompose input.fst - | fstproject --project_output result.fst
```

## Compose

复合（Composition）操作将不同的有限状态机合并成一个。

```shell

fstcompose [--opts] a.fst b.fst out.fst
  --connect: Trim output (def: true)
```

`fsttablecompose`与`fstcompose`类似，但是前者速度更快。

> [ComposeDoc < FST < TWiki](https://www.openfst.org/twiki/bin/view/FST/ComposeDoc)

## Determine

确定化（Determinization）操作确保每个状态对应每个输入有唯一输出。

```shell

fstdeterminize a.fst out.fst
```

`stdeterminizestar`与`fstdeterminize`类似，但前者包含去除空转移处理。

> [DeterminizeDoc < FST < TWiki](https://www.openfst.org/twiki/bin/view/FST/DeterminizeDoc)

## Minimize

最小化（Minimization）操作对有限状态机进行精简以得到最少的状态和转移弧。

```shell

# 如果有两个输出参数，第二个参数表示新输出标签到旧输出标签的映射，也即是in=out1\circ out2
fstminimize  in.fst [out1.fst [out2.fst]]
```

`fstminimizeencoded`与`fstminimize`类似，但前者没有进行权重推移。

> [MinimizeDoc < FST < TWiki](https://www.openfst.org/twiki/bin/view/FST/MinimizeDoc)

## ArcSort

按状态对FST中的弧进行排序。

```shell

# 可以选择按照弧的输入标签或输出标签进行排序
fstarcsort [--opts] a.fst out.fst
  --sort_type: ilabel (def) | olabel
```

## Replace

若干子有限状态机（SubFST）替换根有限状态机（Root FST）中的槽位。

```shell

fstreplace [--epsilon_on_replace] root.fst rootlabel [subfst1.fst label1 ....] [out.fst]
```

实际不指定`rootlabel`功能也可以实现的，但在`fstreplace`源码中，操作对象是label id与对应fst的 pair，因此为了便于处理，需要指定`rootlabel`，便于拓展的root fst。

> [ReplaceDoc < FST < TWiki](https://www.openfst.org/twiki/bin/view/FST/ReplaceDoc)

## 示例

```shell

# 从文本文件中创建FST，输出$lang/address_slot.fst
fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
    $lang/address_slot.txt $lang/address_slot.fst

# 替换操作，使用$lang/address_slot.fst替换$lang/g_with_slot.fst中的槽位
# 输出$lang/g.fst
fstreplace --epsilon_on_replace $lang/g_with_slot.fst $root_label \
  $lang/address_slot.fst  $address_slot_label $lang/g.fst

# 复合->确定化->最小化->弧排序，输出$lang/lg.fst
fsttablecompose $lang/L.fst $lang/g.fst | fstdeterminizestar --use-log=true | fstminimizeencoded | fstarcsort --sort_type=ilabel > $lang/lg.fst || exit 1;
# 复合操作，输出$lang/tlg.fst
fsttablecompose $lang/T.fst $lang/lg.fst > $lang/tlg.fst || exit 1;
```

> [FstQuickTour < FST < TWiki](https://www.openfst.org/twiki/bin/view/FST/FstQuickTour)
> [Available FST Operations](https://www.openfst.org/twiki/bin/view/FST/FstQuickTour#Available%20FST%20Operations)
> [AIBigKaldi（七）| Kaldi的解码图构造（上）（源码解析） - 知乎](https://zhuanlan.zhihu.com/p/340140838)
> [FstAdvancedUsage < FST < TWiki](https://www.openfst.org/twiki/bin/view/FST/FstAdvancedUsage)
> [飞桨AI Studio - 个性化语音识别](https://aistudio.baidu.com/aistudio/projectdetail/4123501)