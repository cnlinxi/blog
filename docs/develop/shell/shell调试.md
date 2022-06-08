# shell调试

## Shell脚本调试选项

Shell本身提供一些调试方法选项：

```
-n，读一遍脚本中的命令但不执行，用于检查脚本中的语法错误。
-v，一边执行脚本，一边将执行过的脚本命令打印到标准输出。
-x，提供跟踪执行信息，将执行的每一条命令和结果依次打印出来。
使用这些选项有三种方法(注意:避免几种调试选项混用)
```

1. 在命令行提供参数：`$sh -x script.sh`
2. 脚本开头提供参数：`#!/bin/sh -x`
3. 在脚本中用set命令启用or禁用参数：其中`set -x`表示启用，`set +x`表示禁用。

set命令的详细说明：
- http://man.linuxde.net/set
- https://www.runoob.com/linux/linux-comm-set.html

> https://www.cnblogs.com/anliven/p/6032081.html