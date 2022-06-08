# docker常用命令

![](attachments/Pasted%20image%2020220608074345.png)

## 基本概念

- 镜像（Image）：Docker 镜像是一个特殊的文件系统，除了提供容器运行时所需的程序、库、资源、配置等文件外，还包含了一些为运行时准备的一些配置参数（如匿名卷、环境变量、用户等）。镜像不包含任何动态数据，其内容在构建之后也不会被改变。
- 容器（Container）：镜像（Image）和容器（Container）的关系，就像是面向对象程序设计中的 类 和 实例 一样，镜像是静态的定义，容器是镜像运行时的实体。容器可以被创建、启动、停止、删除、暂停等。
- 仓库（Repository）：仓库（Repository）类似Git的远程仓库，集中存放镜像文件。

![](attachments/Pasted%20image%2020220608074710.png)

## 服务

| 作用               | 命令                    |
| ------------------ | ----------------------- |
| 查看docker版本信息 | `docker version`          |
| 查看docker简要信息 | `docker -v`               |
| 启动docker         | `systemctl start docker`  |
| 关闭docker         | `systemctl stop docker`   |
| 设置开机启动       | `systemctl enable docker` |
| 重启docker服务     | `service docker start`    |
| 关闭docker服务     | `service docker stop`     |

## 镜像

### 镜像仓库

可以从[Docker Hub](https://hub.docker.com/search?q=&type=image)等镜像仓库获取大量高质量的镜像。

| 作用     | 命令                     |
| -------- | ------------------------ |
| 检索镜像 | `docker search <关键字>` |
| 拉取镜像 | `docker pull [选项] [Docker Registry 地址[:端口号]/]<仓库名>[:标签]`                         |

### 镜像管理

| 作用                         | 命令                               |
| ---------------------------- | ---------------------------------- |
| 列出镜像                     | `docker image ls`或`docker images` |
| 删除镜像                     | `docker rmi <镜像ID>`              |
| 导出镜像                     | `docker save`                      |
| 导入镜像                     | `docker load`                      |
| 镜像运行：创建并运行一个容器 | `docker run <镜像ID>`              |
| 使用Dockerfile创建镜像       | `docker build`                                   |

## Dockerfile构建镜像

Dockerfile是一个文本格式的配置文件，可以使用 Dockerfile 来快速创建自定义的镜像。Dockerfile 由一行行行命令语句组成，并且支持以#开头的注释行。

### Dockerfile常见命令

- FROM：指定基础镜像
- RUN：执行命令
- COPY：复制文件
- ADD：更高级的复制文件
- CMD：容器启动命令
- ENV：设置环境变量
- EXPOSE：暴露端口

示例：

```

FROM java:8
MAINTAINER "jinshw"<jinshw@qq.com>
ADD mapcharts-0.0.1-SNAPSHOT.jar mapcharts.jar
EXPOSE 8080
CMD java -jar mapcharts.jar
```

## 容器

### 容器生命周期

#### 启动

启动容器有两种方式，一是基于镜像新建容器并启动，二是将处于终止状态（stopped）的容器重新启动。

```shell

# 新建并启动
docker run <镜像名/镜像ID>
# 启动已终止容器
docker start <容器ID>
```

#### 查看容器

```shell

# 列出本机运行的容器
docker ps
# 列出本机所有的容器（包括停止和运行）
docker ps -a
```

#### 停止容器

```shell

# 停止运行的容器
docker stop <容器ID>
# 杀死容器进程
docker kill <容器ID>
```

#### 重启容器

```shell

docker restart <容器ID>
```

#### 删除容器

```shell

docker rm <容器ID>
```

注意到，`docker rm`删除容器，`docker rmi`删除镜像。

### 进入容器

进入容器有两种方式：

```shell

# 如果从这个 stdin 中 exit，会导致容器的停止
docker attach [容器ID]
# 交互式进入容器
docker exec [容器ID]
```

通常采用交互式方式进入容器，`docker exec`常见参数：

- `-d`/`--detach`在容器中后台执行命令。
- `-i`/`--interactive=true|false`：打开标准输入接受用户输入命令。

### 导入和导出

#### 容器导出

```shell

# 导出一个已经创建的容器到一个文件
docker export <容器ID>
```

#### 容器导入

```shell

# 导出的容器快照文件可以再导入为镜像
docker import <路径>
```

注意到，`docker load`将文件导入为镜像，而`docker import`将容器重新导入为镜像。

## 其它

### 查看日志

```shell

# 导出的容器快照文件可以再导入为镜像
docker logs <容器ID>
```

该命令的常用参数：

- `-f`：跟踪日志输出。
- `--since`：显示某个开始时间的所有日志。
- `-t`：显示时间戳。
- `--tail`：仅列出最新N条容器日志。

### 复制文件

```shell

# 从主机复制到容器
sudo docker cp <宿主机路径> <容器ID>:<容器路径>
# 从容器复制到主机
sudo docker cp <容器ID>:<容器路径> <宿主机路径>
```

> [一张脑图整理Docker常用命令 - SegmentFault 思否](https://segmentfault.com/a/1190000038921337)

