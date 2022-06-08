# git修改提交信息

1. 如果提交信息`commit message`写错了且这次提交`commit`还没有推`push`, 可以通过下面的方法来修改提交信息`commit message`：

```shell

# 打开默认编辑器，修改提交信息
git commit --amend --only
# 或者一次性完成，直接修改提交信息
git commit --amend --only -m <commit message>
```

2. 如果已经推送`push`，可以修改这次提交`commit`然后强推`force push`，但这种方法不推荐。

> [git commit --amend 修改git提交记录用法详解](https://zhuanlan.zhihu.com/p/100243017)
> [45 个 Git 经典操作场景，专治不会合代码](https://zhuanlan.zhihu.com/p/485010145)