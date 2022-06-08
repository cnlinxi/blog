# top_linux下的任务管理器

`top`工具，类似于Windows下的任务管理器。

## 执行结果

执行的结果形如：

```

$top
    top - 09:14:56 up 264 days, 20:56,  1 user,  load average: 0.02, 0.04, 0.00
    Tasks:  87 total,   1 running,  86 sleeping,   0 stopped,   0 zombie
    Cpu(s):  0.0%us,  0.2%sy,  0.0%ni, 99.7%id,  0.0%wa,  0.0%hi,  0.0%si,  0.2%st
    Mem:    377672k total,   322332k used,    55340k free,    32592k buffers
    Swap:   397308k total,    67192k used,   330116k free,    71900k cached
    PID USER      PR  NI  VIRT  RES  SHR S %CPU %MEM    TIME+  COMMAND
    1 root      20   0  2856  656  388 S  0.0  0.2   0:49.40 init
    2 root      20   0     0    0    0 S  0.0  0.0   0:00.00 kthreadd
    3 root      20   0     0    0    0 S  0.0  0.0   7:15.20 ksoftirqd/0
    4 root      RT   0     0    0    0 S  0.0  0.0   0:00.00 migration/0
```

- CPU信息。如上例所示，需要关注：
	- 第一行：`load average: 0.02, 0.04, 0.00`。系统1分钟、5分钟、15分钟的CPU负载。
	- 第三行Cpu (s)：
		- `99.7%id`：空闲CPU时间百分比。如果这个值过低，表明系统CPU存在瓶颈。
		- `0.0%wa`：等待I/O的CPU时间百分比。如果这个值过高，表明IO存在瓶颈。
- 内存信息。仅查看内存时，也可以使用`free -m`等命令。如上例所示，需要关注：
	- 第四行Mem：
		- `116452k free`：空闲的物理内存量。
	- 第五行Swap：
		- `330116k free`：空闲的交换区量。`Swap`就是Linux下的虚拟内存，用硬盘充当内存。
- 进程信息。
	- PID：进程ID。
	- USER：进程拥有者。
	- PR：进程优先级，越小，优先级越高，越早被执行。
	- %CPU：进程占用的CPU使用率。
	- %MEM：进程占用的物理内存使用率。
	- TIME+：累加的进程CPU使用时间。
	- COMMAND：进程启动命令名称。

## top命令交互操作指令

- s：设置刷新间隔。`<Space>`立即刷新。
- i：不显示闲置或僵死进程。
- P：按%CPU排行。
- T：按TIME+排行。
- M：按%MEM排行。
- u：显示指定用户进程。
- c：显示完整命令。
- q：退出。

> [8. top linux下的任务管理器](https://linuxtools-rst.readthedocs.io/zh_CN/latest/tool/top.html)
> [3. 性能优化](https://linuxtools-rst.readthedocs.io/zh_CN/latest/advance/03_optimization.html)
> [What does SWAP mean in top?](https://unix.stackexchange.com/questions/64981/what-does-swap-mean-in-top)