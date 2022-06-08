# cpp调试技巧

- 编译时，需要带上`-g`参数：
```shell

g++ -g test.c -o test
// -g可配合-Og使用，-Og可提升调试程序的性能
g++ -g -Og test.c -o test
```

- 进入调试：
```shell

gdb test.out
```

## 参考资料

> [掌握gdb调试从入门到进阶（一）](https://zhuanlan.zhihu.com/p/100385553)
> [掌握gdb调试从入门到进阶（二）](https://zhuanlan.zhihu.com/p/100403438)
> [gdb 相关命令](http://blog.letow.top/2017/11/23/gdb-commands/)
> [《100个gdb小技巧》](https://wizardforcel.gitbooks.io/100-gdb-tips/content)
> [在文件行号上打断点 | 100个gdb小技巧](https://wizardforcel.gitbooks.io/100-gdb-tips/content/break-on-linenum.html)


## GDB常用命令

### 启动调试

- `run`。一直执行程序，直到执行到断点处或结束。
- `start`。执行到`main()`函数起始位置。

### 断点

缩写`b`
- 跟行号。`b <line_num>`或者`b <file_path>:<line_name>`，比如`b a/file.c:6`。
- 跟函数名。`b <func_name>`或者`b <file_path>:<func_name>`。
- 条件断点。`break <...> if condition`，中间`<...>`表示上述命令。比如`b <line_num> if <condition>`。
- 查看断点。`info break`，缩写`i b`。
- 删除断点。
	- `delete <break_num>`，缩写`d <break_num>`。其中，`<break_num>`为`i b`查询到的断点序号。
	- `clear <line_num>`。其中，`<line_num>`为行号，可以为`line_num/file:line_num`等形式。

### 单步命令

- `next`，缩写`n`。执行下一句，调用其它函数时不进入。
- `step`，缩写`s`。执行下一句，调用其它函数时进入，step into。
- `continue`，缩写`c`。停止后，继续执行，直到遇到下一个断点或执行结束。
- `finish`，缩写`f`。跳出本层函数，也即一直执行，直至当前函数完成，打印当前的堆栈信息和返回值。
- `util`，缩写`u`。循环体内一直执行，直至退出循环体。
- `stepi`和`nexti`，缩写`si`和`ni`。机器指令的单步命令。

### 查看

- `frame`，缩写`f`。显示当前所在的行及相关信息。
- `list`，缩写`l`。显示当前程序运行位置附近的相关代码。
- `print`，缩写`p`。
	- 动态数组，比如
	```cpp
	
	int *array = (int *)malloc(len * sizeof (int))
	```
	查看该动态数组的值：`p *array@len`
	更多参见：
	
	- [打印STL容器中的内容](https://wizardforcel.gitbooks.io/100-gdb-tips/content/print-STL-container.html)
	- [gdb pretty print](https://papaux.github.io/til/html/cpp/gdb-pretty-print.html)

- `watch`，缩写`w`。观察的变量在变化时，就停止程序。

## 在macOS上使用GDB

> https://zhuanlan.zhihu.com/p/68398728

## 其它技巧

### GDB调试时传入可执行文件的参数

使用`--args`参数：

```shell

gdb --args <executablename> <arg1> <arg2> <arg3>
```

参见：[How do I run a program with commandline arguments using GDB within a Bash script?](https://stackoverflow.com/questions/6121094/how-do-i-run-a-program-with-commandline-arguments-using-gdb-within-a-bash-script)

### 批量杀死进程

```shell

ps aux | grep <进程的关键字> | awk '{print $2}' | xargs kill -9
```

> https://www.coder4.com/archives/1334
> https://stackoverflow.com/questions/3510673/find-and-kill-a-process-in-one-line-using-bash-and-regex

### macOS进入GDB之后进程挂起

例如：
```shell

Starting program: /Users/fluzzlesnuff/Documents/C++/a.out
[New Thread 0x2a03 of process 2389]
```

解决方法：

```shell

sudo DevToolsSecurity -enable
```

> https://apple.stackexchange.com/questions/420492/gdb-hangs-after-new-thread-on-macos

或者，将下述命令写入`~/.gdbinit`：
```

set startup-with-shell off
```
并在执行时指定该配置文件：

```shell

gdb -x ~/.gdbinit <program>
```

macOS上使用GDB参考文档：

- [在macOS10.14上使用GDB的教程](https://zhuanlan.zhihu.com/p/68398728)
- [Setup gdb on macOS in 2020](https://dev.to/jasonelwood/setup-gdb-on-macos-in-2020-489k)