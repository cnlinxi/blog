# git转移部分代码cherry-pick

利用Git将代码从一个分支转移到另一个分支上：

- 转移另一个分支的*所有代码变动*：使用合并`git merge`
- 转移另一个分支的*部分代码变动*（比如某几个提交）：可以采用`git cherry-pick`

## 基本用法

```shell

git cherry-pick <commitHash>
```

上述命令将指定的提交`commitHash`，应用于当前分支，这会在当前分支产生一个新的提交。比如代码仓库中有`feature`和`master`两个分支：

```

a - b - c - d   Master
	 \
	   e - f - g Feature
```

现在希望将提交`f`转移到`master`分支上，则可以：

```

# 切换到 master 分支
$ git checkout master

# Cherry pick 操作
$ git cherry-pick f
```

完成后，代码库就变为了：

```

a - b - c - d - f   Master
	 \
	   e - f - g Feature
```

可见，`master`分支的末尾增加了一个提交`f`。

> [git cherry-pick 教程](https://www.ruanyifeng.com/blog/2020/04/git-cherry-pick.html)