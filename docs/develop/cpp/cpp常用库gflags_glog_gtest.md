# cpp常用库gflags/glog/gtest

## gflags处理命令行参数

### 简介

如果项目通过CMake构建，只需要在CMakeLists.txt中加入类似以下配置即可：

```

find_package (gflags REQUIRED)

include_directories (${gflags_INCLUDE_DIR})
add_executable (main main.cpp)
target_link_libraries (main gflags)
```

### 简要用法

#### `gflags`头文件

```cpp

#include <gflags/gflags.h>
```

#### 定义命令行参数

```cpp

DEFINE_string(<变量名>, <默认值>, <描述>);
```

比如：

```cpp

DEFINE_string(mystr, "hello", "help for demo");
```

其中，`DEFINE_string`只是其中一种数据类型，支持的全部数据类型有：

| gflags定义类型 | 描述           |
| -------------- | -------------- |
| DEFINE_bool    | 布尔类型       |
| DEFINE_int32   | 32位整型       |
| DEFINE_int64   | 64位整型       |
| DEFINE_uint64  | 无符号64位整型 |
| DEFINE_double  | 浮点类型       |
| DEFINE_string  | C++ string类型 |

#### 在`main()`函数加入如下代码，解析命令行参数：

```cpp

gflags::ParseCommandLineFlags(&argc, &argv, true);
```

一般都将上述代码放到`main()`函数的开始位置。

#### 代码内访问传入的参数

使用`FLAGS_<变量名>`就可以在代码中访问解析得到的参数值。例如：

```cpp

std::cout<<"My str is: "<<FLAGS_mystr<<std::endl;
```

#### 使用

```shell

>>> g++ <gflags_demo.cpp> -o gflags_demo -lgflags -lpthread  # -l 链接库进行编译
>>> ./gflags_demo -mystr="this is a value of gflags member"
My str is: this is a value of gflags member
```

其它变量类型类似，但布尔类型的参数有些独特性。比如定义一个布尔类型的参数`debug_bool`，在命令行中可以进行如下指定：

```shell

>>> ./gflags_demo -debug_bool  # debug_bool的值为true
>>> ./gflags_demo -debug_bool=true  # debug_bool的值为true
>>> ./gflags_demo -debug_bool=1  # debug_bool的值为true
>>> ./gflags_demo -debug_bool=0  # 和直接传入false一样，debug_bool的值为false
```

### 进阶

#### 跨文件调用

访问在另一个文件中定义的`gflags变量`：使用`DECLARE_`，作用类似于`extern`声明变量。

为了方便管理变量，推荐在`*.cpp/*.cc`文件中`DEFINE_`变量，在对应的`*.h`文件或单元测试中`DECLARE_`变量。比如：

```cpp

// foo.cpp
# include "foo.h"
DEFINE_string(mystr, "hello", "help for demo");  // 定义一个gflags变量name

// foo.h
DECLARE_string(mystr);  // extern声明变量name
```

#### 参数检查

```cpp

gflags::RegisterFlagValidator
```

#### gflags使用配置文件传入参数

配置文件内容类似于：

```

// my.flags配置文件的内容
--mystr="hello"
--myvalue=10
```

使用时，直接通过`--flagfile`指定该配置文件my.flags即可：

```shell

>>> ./gflags_demo --flagfile my.flags
```

> [C++ gflags库使用说明](https://blog.csdn.net/wcy23580/article/details/89222962)
> [gflags](https://github.com/gflags/gflags)

## glog轻量级日志

### 简介

安装glog之前需要安装gflags，这样glog就可以使用gflags去解析命令行参数。如果项目通过CMake构建，只需要在CMakeLists.txt中加入类似以下配置即可：

```

# 利用CMakeLists.txt配置glog的示例
find_package (glog 0.3.5 REQUIRED)

add_executable (main main.cpp)
target_link_libraries (main glog::glog)
```

### 简要用法

#### `glog`头文件

```cpp

#include <glog/logging.h>
```

#### 初始化

`main()`函数加入以下代码，初始化`glog`：

```cpp

// 解析命令行参数
gflags::ParseCommandLineFlags(&argc, &argv, true);
// 初始化日志库
google::InitGoogleLogging(argv[0]);
```

#### 使用

```cpp

LOG(ERROR) << "Hello, World!";
```

### 进阶

#### 日志级别

- `LOG(<level>)`默认4个级别，`<level>`可选项：`INFO/WARNING/ERROR/FATAL`。
- 条件日志：`LOG_IF/LOG_EVERY_N/LOG_IF_EVERY_N/LOG_FIRST_N`。
- 自定义级别宏`VLOG(n)`。自定义日志级别，通过参数`--v=<val>`指定输出的日志级别。比如：`--v=3`则只输出`n<=3`的日志。同样，自定义级别宏`VLOG`也支持`VLOG_IF/VLOG_EVERY_N/VLOG_IF_EVERY_N`等。
- 在`LOG`宏名前加`D`，指定只在Debug模式（即没有开启`NDEBUG`）下生效，比如`DLOG/DLOG_IF/DLOG_EVERY_N`等。

#### 常用命令行参数

`glog`依赖于`gflags`，以下为`glog`常见的命令行参数。

| 命令行参数        | 描述                                                                                                                                        |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| --logtostderr     | 默认false。将所有日志信息都输出到stderr上                                                                                                   |
| --log_dir         | 日志输出目录，默认为/tmp/。当`--logtostderr`设置为false时，则日志输出到文件                                                                 |
| --minloglevel     | 超过该级别的日志才会记录                                                                                                                    |
| --stderrthreshold | 超过该级别的日志除了输出到日志外，还会输出到stderr                                                                                          |
| --v               | `VLOG(<level>)`的最大输出日志级别                                                                                                           |
| --vmodule         | 分模块/文件名指定`VLOG(<level>)`的最大输出日志级别，以字符串`<module1>=<level1>,<module2>=<level2>...`形式指定，支持通配符，优先级高于`--v` |

#### `CHECK`宏检查错误

当`CHECK()`的条件不满足时，`glog`会记录`FATAL`级别日志，并输出调用堆栈。

```cpp

CHECK(fun() == 0) << "Call fun() failed!";
```

和C++自带的断言`assert()`不同，无论是否开启`NDEBUG`，`CHECK()`都会执行。除了`CHECK()`，`glog`还提供其它的宏，包括：

| CHECK_XXX     | 条件   |
| ------------- | ------ |
| CHECK_EQ(x,y) | `x==y` |
| CHECK_NE(x,y) | x!=y   |
| CHECK_LE(x,y) | x<=y   |
| CHECK_LT(x,y) | x<y    |
| CHECK_GE(x,y) | x>=y   |
| CHECK_GT(x,y) | x>y    |

在判断指针是否为空时，需要将`NULL`转换为相应的类型再进行比较，比如：

```cpp

CHECK_EQ(some_ptr, static_cast<SomeType*>(NULL)) << "some_ptr is a null pointer";
```

在判断`char *`类型的字符串时，可以使用`CHECK_STREQ()/CHECK_STRNE()`，相对应带有CASE的版本`CHECK_STRCASEEQ()/CHECK_STRCASENE()`为大小写不敏感的。传递NULL值给这些宏是安全的。

可以使用`CHECK_DOUBLE_EQ()`检查两个浮点数是否相等，并允许出现比较小的误差。如果需要自己提供的误差范围，可以使用`CHECK_NEAR()`，该宏的第三个参数就是指定的误差范围。

#### 其它

1. 当需要线程安全时，使用`RAW_LOG/RAW_CHECK`等；
2. 系统级日志记录`SYSLOG/SYSLOG_IF/SYSLOG_EVERY_N`宏，将调用syslog()函数来记录系统级别的日志；
3. perror风格日志`PLOG/PLOG_IF`
4. 精简日志信息，删除日志级别、所在代码行数等信息：

	```cpp
	
	#define GOOGLE_STRIP_LOG 1    // this must go before the #include!
	#include <glog/logging.h>
	```

> [glog](https://github.com/google/glog)
> [How To Use Google Logging Library (glog)](https://rpg.ifi.uzh.ch/docs/glog.html)
> [使用 Google 的 glog 日志库](http://senlinzhan.github.io/2017/10/07/glog/)
> [Google C++库](http://notes.tanchuanqi.com/language/cpp/google_library.html)

## gtest断言

- `ASSERT_*`系列的断言，当检查点失败时，退出执行。
- `EXPECT_*`系列的断言，当检查点失败时，继续向下执行。
- 还有`testing::StaticAssertTypeEq<int, T>();`可以用来检查类型T是不是int类型，否则产生一个编译时的静态断言错误。