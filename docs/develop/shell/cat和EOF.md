# cat和EOF

在shell上向一个文件中写入多行内容，并且自定义文件结束符，就可以使用`cat > file << EOF`来完成：

```shell

# 以下命令向g_with_slot.txt中写入如下内容：
# 0 1 打 打
# 1 2 <ADDRESS_SLOT> <ADDRESS_SLOT>
# 2
cat > g_with_slot.txt <<EOF
0 1 打 打
1 2 <ADDRESS_SLOT> <ADDRESS_SLOT>
2
EOF
```

`cat > file << EOF`中，`cat > file`表示创建文件并将标准输入设备上的内容输出重定向到`file`文件中，当然可以用`>>`代替`>`追加内容而非覆盖内容。`<< EOF`显示输入提示符`>`，并以`EOF`判定文件内容的输入结束。`EOF`并非固定的文件结束符，也可以写作`cat > file << EOF2`，此时则以`EOF2`判定文件内容的输入结束。

`<< EOF`和`> file`的位置不固定，两者可交换。

> [Linux小技巧：cat > file 和 EOF 的妙用_曾经去过跨越一个小时的地方的博客-CSDN博客](https://blog.csdn.net/u012814856/article/details/87972688)