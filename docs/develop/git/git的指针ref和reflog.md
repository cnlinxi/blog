# git的指针ref和reflog

## Git指针ref概念

`ref`是一种指向一次提交的非直接方式，可以看做是提交哈希（commit hash）的别名，用于表示Git的分支和标签。`ref`作为普通文件存储于`.git/refs`目录下：

```shell

$ ls .git/refs
heads remotes tags
```

其中，

- `heads`目录定义了仓库中所有的本地分支，`heads`目录下的每个文件的文件名对应分支名，文件内容为该分支最后一次提交对应的哈希值。在`main`分支上进行一次提交，Git实际就是修改`refs/heads/main`的内容，新建一个分支，实际就是将提交哈希写入一个新文件中。
- `tags`目录与`heads`作用类似，不同之处在于保存`git tags`而非分支信息。
- `remote`目录存放着所有远程仓库，每个远程仓库对应一个子目录，该子目录下又存放着获取（fetch）到存储库中的所有远程分支：

	```shell
	
	$ ls .git/refs/remotes 
	origin
	$ ls .git/refs/remotes/origin 
	HEAD                        fix_db7f3d3                 master
	Tacotron2-iter-260K-824c091 fix_server                  refactor
	dependabot                  ljspeech-tacotron-iter-185K
	dev                         main
	```

因此，在一些需要指定具体提交哈希的场景中，除了指定哈希值之外，还可以直接指定`ref`，比如`git show .git/refs/heads/HEAD`。

对于大型仓库，Git会周期性执行垃圾回收，以减少不必要的对象并压缩`ref`，以提高性能。当然，也可以使用`git gc`手动执行垃圾回收，该命令会将上述分支和tag对应的`ref`文件压缩到`.git`目录下的`packed-refs`文件中。

## 特殊ref的指向

- `HEAD`：最近检出（check out）的提交（commit）或分支。
- `FETCH_HEAD`：最近从远程仓库中获取（fetch）的分支。
- `ORIG_HEAD`：在较大变更前`HEAD`的备份。
- `MERGE_HEAD`：使用`git merge`合并到当前分支的提交（commit）。
- `CHERRY_PICK_HEAD`：使用`git cherry-pick`摘取部分内容到当前分支的提交（commit）。

## 远程仓库的默认名origin

`origin`是远程仓库链接的别名，`origin`对应的链接太长，使用起来较为麻烦。因此`origin`的本质是指向远程仓库的一个指针名，而master/main则属于仓库中的一个部分。

可以通过`git remote -v`或者仓库配置文件`.git/config`查看远程仓库的具体链接，可以通过`git remote add <remote-repo-name> <url>`添加新的远程仓库别名`<remote-repo-name>`和对应的链接`<url>`。

`git push`默认创建`origin`和`main`，因此推送时可以省略，完整形式应是`git push origin main`。如果希望Git记录推送到远程分支的默认值，可以加上`-u`参数，也就是`git push -u <remote> <local>`，这样当下次还想要继续推送到该远程分支时，推送命令就可以简写为`git push`。

## 远程分支和本地分支的映射refspecs

`refspecs`将本地仓库的分支映射到远程仓库的分支，以便使用本地Git命令管理远程分支，并配置一些特殊的`git push`和`git fetch`行为。

`refspecs`定义如下：

```shell

# definition
[+]<src>:<dst>
# example
+refs/heads/*:refs/remotes/origin/*
```

其中，

- `+`表示强制远程仓库执行非快进更新（non-fast-forward update）。
- `<src>`表示本地仓库的源分支。
- `<dst>`表示远程仓库的目标分支。

`refspecs`可以与`git push`一起使用，为远程分支指定不同的名称。比如以下命令将`main`本地分支推送到`origin`远程仓库，但使用`qa-main`作为远程仓库`origin`中对应分支的名称：

```shell

git push origin main:refs/heads/qa-main
```

同样可以利用`refspecs`删除远程分支，类似地，此时将本地分支的名称置为空即可：

```shell

git push origin :qa-main
# 该删除分支的操作等同于
git push origin --delete qa:main
```

通过修改Git的配置文件`.git/config`，可以使用`refspecs`修改`git fetch`的默认行为。默认情况下，`git fetch`获取远程仓库的所有分支，因此`.git/config`文件相关内容如下：

```

[remote "origin"]
        url = https://github.com/wenet-e2e/wenet.git
        fetch = +refs/heads/*:refs/remotes/origin/*
```

上述配置要求Git下载远程仓库`origin`的所有分支，如果只需要`main`分支，可以更新相关内容如下：

```

[remote "origin"]
        url = https://github.com/wenet-e2e/wenet.git
        fetch = +refs/heads/main:refs/remotes/origin/main
```

类似地，可以修改`git push`的默认行为。比如，如果想一直将本地分支`main`推送到远程仓库`origin`的`qa-main`，可以更新相关内容如下：

```

[remote "origin"]
        url = https://github.com/wenet-e2e/wenet.git
        fetch = +refs/heads/main:refs/remotes/origin/main
        push = refs/heads/main:refs/heads/qa-main
```

## 相对ref

- `HEAD~1`：等同于`HEAD~`或`HEAD^1`或`HEAD^`，后退至`HEAD`之前的提交。
- `HEAD^2`：后退至当前分支的第二个父提交。
- `HEAD~1^2`：后退至`HEAD`之前的提交，再后退到当前分支的第二个父提交上，如果`HEAD`没有合并分支，则非法。
- `HEAD@{2}`：指向`git reflog`记录的整体操作的第三条操作（`git reflog`记录的整体操作从0开始）。

## reflog

`reflog`记录了在当前仓库上进行的所有Git操作，可以通过`git reflog`查看该Git日志，比如：

```shell

$ git reflog
02d7064 (HEAD -> main, origin/main, origin/HEAD) HEAD@{0}: pull: Fast-forward
25897e0 HEAD@{1}: checkout: moving from 8a1cb96795b38ba9aeb23c4005242acd0d9829e8 to main
8a1cb96 (origin/kaitang-ssl-train) HEAD@{2}: pull https://github.com/cnlinxi/wenet.git kaitang-ssl-train: Fast-forward
422c89e HEAD@{3}: checkout: moving from main to remotes/origin/kaitang-ssl-train
88a652a HEAD@{4}: commit: delete set_audio_backend
a9bae0c HEAD@{5}: commit: update
f0fe211 HEAD@{6}: commit: update(examples/librispeech):为kaggle修改数据路径
```

其中，`HEAD{<num>}`用于标识在`reflog`中的提交（commit），可以配合其他命令对指定提交（commit）进行操作，比如删除最后一次提交时，可以执行`git reset --soft HEAD@{1}`。

> [Refs and the Reflog](https://www.atlassian.com/git/tutorials/refs-and-the-reflog)
> [Git 里面的 origin 到底代表啥意思?](https://www.zhihu.com/question/27712995)
> [git push 的 -u 参数含义](https://blog.csdn.net/Lakers2015/article/details/111318801)
> [这才是真正的GIT——GIT实用技巧](https://www.lzane.com/tech/git-tips/)
> [这才是真正的GIT——分支合并](https://www.lzane.com/tech/git-merge/)