# Linux操作基础备忘录

| 命令    | 简要说明                                             | 示例                                  | 示例说明                             |
| ------- | ---------------------------------------------------- | ------------------------------------- | ------------------------------------ |
| man     | 命令文档                                             | man find                              |                                      |
| which   | 查看程序的binary文件所在路径                         |                                       |                                      |
| whereis | 查看程序搜索路径，当安装同一工具多个版本时，会很有用 |                                       |                                      |
| find    | 实时文件查找                                         | `find /home -name "*.txt"`            |                                      |
| locate  | 文件系统数据库上查找                                 | `locate myfile`                       |                                      |
| ln -s   | 软链接                                               | `ln -s <真实文件> <快捷方式>`         |                                      |
| &&      | 前面成功则执行后面                                   |                                       |                                      |
| \|\|    | 前面失败则执行后面                                   |                                       |                                      |
| ps -ef  | 查询正在运行的进程信息                               |                                       |                                      |
| kill    | 杀死进程                                             | `kill -9 <PID>`                       |                                      |
| netstat | 显示网络信息                                         | `netstat -a`                          | 列出所有端口，包括监听或未监听的端口 |
| rsync   | 复制                                                 | `rsync -avzP <待复制文件> <目标文件>` | 将待复制文件复制到目标文件           |
| scp     | 远程下载/上传                                        | `scp -r <localpath> <ID@host:path>`   | 上传，将localpath上传到host的path    |
| tar     | 压缩/解压缩                                          | `tar -xvf <demo.tar>`                                      |                                      |

> [Linux基础](https://linuxtools-rst.readthedocs.io/zh_CN/latest/base)
> [Linux命令搜索](https://wangchujiang.com/linux-command/)