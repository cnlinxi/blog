# GCC的常用编译选项

编译选项`-O`是优化标志（Optimization flags）的合集，比如指定编译选项`-Og`就等同于打开了`-fauto-inc-dec,-fbranch-count-reg，...,-fbranch-count-reg,...`标志。GCC编译选项示例如下：

```

CFLAGS := -Wall -Wundef -Wshadow -Wconversion -Werror=return-type \
    -Werror=implicit-function-declaration -Werror=unused-variable \
    -fno-strict-aliasing -fno-common -fshort-wchar -fno-PIE \
    -Wno-trigraphs -Os
```

## 调试选项

指定`-g`标志，如果为了提升调试程序性能，可以配合使用针对调试的优化选项`-Og`。

## 优化选项

常用的是`-O2`和`-Os`。

- `-O0`/`-O`：默认选项，不执行优化。
- `-O1`：执行级别1的优化，尝试减少代码大小和提高性能，比如 -fdce（移除不可能执行到的代码），-fif-conversion（尝试简化if语句），-fmerge-constants（尝试合并相同的常量）。但是不包括需要花费大量编译时间的优化选项；
- `-Og`：调试选项，启用`-O1`的优化执指令，同时获取更多调试信息。
- `-O2`：执行`-O1`所有优化选项，同时额外执行几乎全部不需要在空间和性能之间平衡的优化选项。比如 -fgcse（优化全局公共表达式、常量的传递），-fcode-hoisting（将所有分支都需要执行的表达式尽早执行），-finline-functions（考虑将所有函数变成内联函数）；
- `-Os`：专门用于**优化代码大小**的优化级别，执行`-O2`所有优化选项，同时排除那些可能导致程序大小增加的优化选项；
- `-O3`：最高优化等级，该优化级别较高，执行的优化不会很直观，可能也会出现一些问题。

## 警告选项

一般启用特定类型警告的格式为`-Wxxx`，排除特定类型的警告的格式为`-Wno-xxx`。比如使用`-Wall -Wno-unused-variable`可以从`-Wall`中排除`-Wunused-variable`。

- `-Wall`：常用编译选项，对代码进行基本检查；
- `-Wextra`：`-Wall`基础上的补充警告；
- `-Werror`：所有警告视作错误。

![](attachments/Pasted%20image%2020220523104702.png)

> [GCC的常用编译选项](https://zhuanlan.zhihu.com/p/483854632)
> [Optimize Options (Using the GNU Compiler Collection (GCC))](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)
> [Warning Options (Using the GNU Compiler Collection (GCC))](https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html)