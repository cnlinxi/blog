# git中的HEAD

## HEAD简介

HEAD在`git`版本控制中表示头节点，也就是**当前分支的最后一次提交**。同时它是`<repo>/.git/HEAD`文件，内容一般是：`ref: refs/heads/main`，本质是指上一次提交的hash值，比如`se11d9be5cc7007995b607fb12285a43cd03154b`。

## HEAD~和HEAD^

在`HEAD`后加`^`和`~`是指以`HEAD`为基准，表示之前的版本。因为`HEAD`是当前分支的最新版本，因此`HEAD~`和`HEAD^`是指次新版本，也就是倒数第二个版本，`HEAD~~`和`HEAD^^`是指次次新版本，也就是倒数第三个版本。

`HEAD~`、`HEAD^`是`HEAD~1`、`HEAD^1`的简略形式。

## HEAD~和HEAD^后加大于1的数字

最新提交之前的最近提交称之为`父提交`，一个分支可能有两个父提交，两个分支合并在一起时，这两个分支的原`HEAD`都会成为合并后的最新提交的`父提交`。

当`HEAD~<num>`表示在第一个父提交上后退`<num>`步，`HEAD^<num>`表示后退到第`<num>`个父提交上。比如`HEAD~2`表示后退两步，且每一步后退均在第一个父提交上，而`HEAD^2`表示后退一步，这一步后退到第二个父提交上，参见下例。

## 示例

- `HEAD~1`：等同于`HEAD~`或`HEAD^1`或`HEAD^`，后退至`HEAD`之前的提交。
- `HEAD^2`：后退至当前分支的第二个父提交。
- `HEAD~1^2`：后退至`HEAD`之前的提交，再后退到当前分支的第二个父提交上，如果`HEAD`没有合并分支，则非法。
- `HEAD@{2}`：指向`git reflog`记录的整体操作的第三条操作（`git reflog`记录的整体操作从0开始）。
- `HEAD~~`：早于`HEAD`的2个提交。
- `HEAD^^`：早于`HEAD`的2个提交。

![](attachments/Pasted%20image%2020220521010848.png)

> [git在回退版本时HEAD~和HEAD^的作用和区别](https://blog.csdn.net/albertsh/article/details/106448035)
> [HEAD~ vs HEAD^ vs HEAD@{} also known as tilde vs caret vs at sign](https://stackoverflow.com/questions/26785118/head-vs-head-vs-head-also-known-as-tilde-vs-caret-vs-at-sign)