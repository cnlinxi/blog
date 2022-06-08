# shell语法速查手册

## 简介

- `sh`是Unix最初使用的shell，Linux默认为`bash`，是`sh`的扩展。

- `source <FileName>`：在当前bash环境下，读取并执行`FileName`中的命令，该`FileName`可以无执行权限，该命令可以用命令`.`代替；

- `sh <FileName>`或者`bash <FileName>`：打开子shell来读取并执行`FileName`中的命令，该`FileName`可以无执行权限；

- `./<FileName>`：打开子shell来读取并执行`FileName`中的命令，该`FileName`需要执行权限。

## shell变量

1. 定义变量：`var_name="linxi"`

2. 使用变量：`echo $var_name`或者`echo ${var_name}`，后者使用变量时加入了花括号`{}`，这是可选的，加不加都行，但是*推荐*加上，以便帮助shell界定变量名的边界。比如`echo ${var_name}Script`，去掉花括号，shell将认为变量名为`var_nameScript`，而非`var_name`。

3. 只读变量：`readonly`。比如：

```shell
var_name="my var"
readonly var_name
```

4. 删除变量：`unset`，`unset <var_name>`。

5. shell字符串

    - 单引号：单引号的任何字符串都会原样输出，单引号内的变量名是无效的；

    - 双引号：双引号内可以有变量，双引号里可以出现转义字符。

    - 获取字符串长度：

    ```shell
    string="12"
    echo ${#string}
    ```

    - 提取子字符串：`echo ${string:1:4}`

6. shell数组：用括号表示数组，数组元素用**空格**隔开，一般形式：

    ```shell
    arr_name=(val1 val2 val3)
    ```

    或者

    ```shell
    arr_name=(val1
    val2
    val3)
    ```

    - 读取数组：`${var_name[index]}`

    - 获取数组中所有元素：`${arr_name[@]}`

    - 获取数组长度：`${#arr_name[@]}`或者`${#arr_name[#]}`

## shell传递参数

向shell脚本内传递参数，脚本内获取参数的格式为：`$n`，其中`n`表示以一个数字，`1`是执行脚本的第一个参数，`2`是第二个参数……`0`是执行的脚本名。

- 在向shell脚本传递的参数中如果包含空格，应该使用单引号或者双引号将该参数括起来，以便脚本将该参数作为整体接收；

- shell重点中括号（包括单中括号`[]`和双中括号`[[]]`）可用于一些条件的测试：

    - 算法比较，比如检查一个变量是否是0，应写作`[ $var_name -eq 0 ]`

    - 文件属性，比如确定一个文件是否存在，应写作`[ -e $file ]`；一个目录是否存在，应写作`[ -d $dir ]`;

    - 字符串比较，比如比较两个字符串是否相同：`[[ "$string1" = "$string2" ]]`。

## shell基本运算符

原生shell不支持简单的数学运算，但可以通过其他命令实现，例如`awk`和`expr`，其中`expr`最常用，比如两数字相加：

```shell
val=`expr 2 + 2`  # 注意，表达式和运算符之间必须要有空格，例如“2+2”是不正确的，应为“2 + 2”
val=$(expr 2 + 2)  # 完整的表达式要被``包裹
```

1. 算术运算符`+`, `-`, `\*`, `/`, `%`, `=`, `==`, `!=`。例如：

```shell
if [ $a != $b ]; then
    echo "a不等于b"
fi
```

2. 关系运算符`-eq`(equal), `-ne`(negative equal), `-gt`(greater than), `lt`(less than), `-ge`(greater equal), `-le`(less equal)

```shell
if [ $a -le $b ]; then
    echo "$a -le $b: a小于等于b"
else
    echo "$a -gt $b: a大于b"
fi
```

3. 布尔运算符

- `!`（非运算）

```shell
# echo "!false"
flag=false
if [ !${flag} ]; then
    echo "!${flag}"
else
    echo "NO!"
fi
```

- `-o`（或运算）

```shell
# echo "1 -lt 2 -o 2 -lt 2"
a=1
b=2
if [ $a -lt 2 -o $b -lt 2 ]; then
    echo "$a -lt 2 -o $b -lt 2"
else
    echo "NO!"
fi
```

- `-a`（与运算）

```shell
# echo "1 -le 2 -o 2 -le 2"
a=1
b=2
if [ $a -le 2 -a $b -le 2 ]; then
    echo "1 -le 2 -a 2 -le 2"
else
    echo "NO!"
fi
```

3. 逻辑运算符，`&&`逻辑AND，`||`逻辑OR

4. 字符串运算符

- `==`检查两个字符串是否相等，相等返回true

- `!=`检测字符串不相等

- `-z`检测字符串的长度是否*是0*，长度是0就返回true

- `-n`检测字符串的长度是否*不是0*，长度不是0就返回true

5. 文件测试运算符

- `-d <dir>`检查`<dir>`是否是目录，如果是目录则返回true

```shell
# echo "/home/my_dir is dir"
dir_path=/home/my_dir  # this is the path of one directory
if [[ -d ${dir_path} ]]; then
    echo "${dir_path} is dir"
else
    echo "NO!"
fi
```

- `-f <file>`检查`<file>`是否是文件，如果是文件则返回true

- `-e <file>`检查文件或者目录`<file>`是否存在，如果存在返回true

- `-x <file>`检查文件是否可执行

- `-s <file>`检查文件是否为空

**Tricks**

- 在判断条件中推荐使用`[[]]`而非`[]`可以避免脚本中的逻辑错误，比如`&&`和`||`，以及操作符`>`和`<`都存在于`[[]]`中，而不能存在于`[]`中；

- `[]`表达式：在`[]`表达式中，常见的`>`，`<`需要加转义字符；

- `[[]]`表达式：支持`<`，`>`且不需要转义字符，并且支持`||`，`&&`逻辑运算符，在`[[]]`中不适用`-a`，`-o`。

## 字符串输出

`echo`输出字符串。

- `>`重定向输出至某位置，没有则新建，清空原有内容；

- `>>`重定向追加至某位置，没有则新建，追加内容。

- `2 >`重定向错误输出

- `2 >>`重定向错误，追加输出到文件结尾

- `& >`混合错误和正确输出

## shell流程控制

1. `if else`

```shell

if condition1; then
    command1
elif condition2; then
    command2
elif condition3; then
    command3
else
    command4
fi
```

例：

```shell
a=10
b=20
if [ $a == $b ]; then
    echo "$a == $b"
elif [ $a -gt $b ]; then
    echo "$a greater than $b"
fi
```

2. `for`

```shell
for var in item1 item2 itemN; do
    command
done
```

或者

```shell
for((assignment;condition;next)); do
    command
done
```

3. `while`

```shell
while condition; do
    command
done
```

例：

```shell
i=1
while(($i<=5)); do
    echo "i is $i"
    let "i++"
done
```

无限循环：

```shell
while
do
    command
done
```

或者

```shell
while true:
do
    command
done
```

或者

```shell
for((;;))
```

4. `until`循环：与`while`的condition相反

```shell
until condition
do
    command
done
```

5. `case`：多选择语句

6. `break`：跳出循环

7. `continue`：跳出本步循环

## 输入/输出重定向

如果希望将`stdout`和`stderr`合并后重定向到`file`，则可以这样写：

```shell
command > file 2>&1
```

或者

```shell
command >> file 2>&1
```

`2>1`表示将`stderr`重定向到当前路径下文件名为`1`的*普通文件*中；而`2>&1`表示将`stderr`重定向到文件描述符为`1`的*文件（也即/dev/stdout）*中，这个文件其实是`stdout`在文件系统中的映射。

例如：

```shell
find /etc -names "*.txt" > list 2>&1
```

上例中，命令从左向右执行，执行至*list*时，此时标准输出`stdout`为*list*；而执行到`2>&1`时，则表示`stderr`也重定向到`stdout`，在本例中即是*list*文件；

又由于`find`命令的参数应为`-name`而非`-names`，因此会发生错误，错误信息`2>&1`，重定向至标准输出，也就是`list`文件。此时屏幕不会出现错误信息，而全部打印到`list`文件中。

例：

```shell
# 屏蔽所有输出
command > /dev/null 2>&1
```