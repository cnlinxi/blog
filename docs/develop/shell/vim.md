# vim

| 按键                    | 说明                                                                                           |
| ----------------------- | ---------------------------------------------------------------------------------------------- |
| 0或功能键`Home`         | 移动到一行的最前面字符处                                                                       |
| $或功能键`End`          | 移动到一行的最后面字符处                                                                       |
| G                       | 移动到最后一行                                                                                 |
| nG                      | n为数字，移动到这个档案的第n行                                                                 |
| gg                      | 移动到第一行，相当于1G                                                                         |
| /word                   | 向光标之下寻找一个名称为word的字符串                                                           |
| n                       | 重复前一个搜索动作                                                                             |
| N                       | 反向重复前一个搜索动作                                                                         |
| `:n1,n2s/word1/word2/g` | 在第 n1 与 n2 行之间寻找 word1 这个字符串，并将该字符串取代为 word2                            |
| `:1,$s/word1/word2/g`   | 从第一行到最后一行寻找 word1 字符串，并将该字符串取代为 word2                                  |
| `:1,$s/word1/word2/gc`  | 从第一行到最后一行寻找 word1 字符串，并将该字符串取代为 word2，在替换前要求用户确认（confirm） |
| dd                      | 剪切游标所在的一整行，用p/P粘贴                                                                |
| ndd                     | n 为数字，剪切光标所在的向下 n 行，用p/P粘贴                                                   |
| yy                      | 复制游标所在的一行                                                                             |
| u                       | 复原前一个动作                                                                                 |
| `[Ctrl]+r`              | 重做前一个动作                                                                                 |
| `:set nu`               | 显示行号                                                                                       |
| `:set nonu`             | 取消行号                                                                                       |

> [Linux vi/vim | 菜鸟教程](https://www.runoob.com/linux/linux-vim.html)