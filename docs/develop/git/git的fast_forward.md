# Git的Fast Forward

## fast-forward含义

Git可以采用fast-forward和no fast-forward两种方式进行合并（merge），fast-forward合并时直接将目标分支的指针指向源分支的位置，不会保留源分支的提交记录。默认情况下，Git以fast-forward的方式进行合并（merge），可以通过：

```shell

# no fast-forward提交
git merge <source-branch> --no-ff
```

修改默认行为，进行no fast-forward方式提交。

## 示例

如下，希望将`feature/add-page`分支合并到`master`分支，此时`master`分支落后`feature/add-page`两个提交：

![](attachments/Pasted%20image%2020220522225003.png)

如果采用默认的fast-forward进行合并（merge），则会将`feature/add-page`分支的提交记录合并到`master`上，直接将`master`指针指向`feature/add-page`即可：

![](attachments/Pasted%20image%2020220522225248.png)

如果采用no fast-forward进行合并（merge），则会保留原始分支的提交记录，并新增一个提交`merge branch 'feature/add-page' into master`：

![](attachments/Pasted%20image%2020220522225538.png)

## fast-forward优缺点

### 优点

提交记录清晰，合并不会出现多个小岔路。

### 缺点

不保留每一个分支的提交记录。

> [連葉子都秒懂的 Fast Forward](https://tzuhui.github.io/2019/06/20/Git/fast-forward/)