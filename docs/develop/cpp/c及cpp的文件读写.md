# c及cpp的文件读写

## c风格文件读写

```c

#include <stdio.h>

FILE* fp = fopen(<filename>, "rb");
fread(<buffer>, 8, sizeof(char), fp);
fclose(fp);
```

### 头文件

c风格读写文件标准库包含于头文件`<stdio.h>`或`<cstdio>`（cpp）。

### 打开文件

使用`fopen()`新建或打开文件，`fopen()`会返回`FILE`对象：

```c

FILE *fopen(const char* filename, const char* mode);
```

其中，参数`mode`的可选值如下：

| mode | 描述                                                             |
| ---- | ---------------------------------------------------------------- |
| r    | 打开已有的文本文件，允许读写                                     |
| w    | 打开文本文件，允许写入，如果文件不存在则新建，如果文件存在则覆盖 |
| a    | 打开文本文本，追加写入，如果文件不存在则新建                     |
| r+   | 打开文本文件，允许读写                                           |
| w+   | 打开文本文本，允许读写，如果文件不存在则新建，如果文件存在则覆盖 |
| a+   | 打开文本文本，追加写入，如果文件不存在则新建，从头读取，追加写入 |

对应的二进制模式`mode`为：`rb/wb/ab/rb+/wb+/ab+/r+b/w+b/a+b`。

### 读取文件

#### 二进制读取

```c

size_t fread(void *buffer, size_t size_of_elements, size_t number_of_elements, FILE *fp);
```

- `buffer`：存放文件读取数据的起始地址。
- `size_of_elements`：每个数据块的字节数。
- `number_of_elements`：数据块的个数。
- `fp`：文件指针。

从`fp`指向的文件中读取数据块，读取的总字节数为`size_of_elements*number_of_elements`，读取出来的数据存放到`buffer`为起始地址的内存中，如果文件结束或发生错误，返回值为0。

#### 字符串读取

```c

char *fgets(char *buffer, int n, FILE *fp);
```

- `buffer`：字符数组指针，该数组存储读取的字符串。
- `n`：要读取的最大字符数，包括最后的空字符，通常是`buffer`数组长度。
- `fp`：文件指针。

如果成功，该函数返回`buffer`指针；如果错误，返回空指针。

```c

int fscanf(FILE *fp, const char *format, ...);
```

- `fp`：文件指针。
- `format`：c字符串，比如`%c/%s/%d/%u/%o/%x`（字符/字符串/十进制数/无符号十进制数/八进制数/十六进制数）。

该函数类似于`scanf()`，需要提供读取数据的类型和格式：

```c

char str1[10], str2[10], str3[10];
int year;
FILE * fp;

fp = fopen ("file.txt", "w+");
fputs("We are in 2014", fp);

rewind(fp);  // 将fp指针移动到文件开头
fscanf(fp, "%s %s %s %d", str1, str2, str3, &year);

fclose(fp);
```

#### 字符读取

```c

int fgetc(FILE * fp);
```

`fgetc()`函数从`fp`指向的文件中读取一个字符，返回值为读取的字符，如果发生错误则返回`EOF`。

### 写入文件

```c

size_t fwrite(const void *buffer, size_t size_of_elements, size_t number_of_elements, FILE *fp);
```

- `buffer`：待写入数据的起始地址。
- `size_of_elements`：每个数据块的字节数。
- `number_of_elements`：数据块的个数。
- `fp`：文件指针。

如果执行成功，返回写入的数据块个数。

类似于读取文件，写入文件同样有对应的字符串读取和字符读取版本：

```c

// 写入字符串，将字符串buffer写入fp指向的文件中
int fputs(const char *buffer, FILE *fp);
// 写入字符串，类似于fscanf，需要指定写入格式
int fprintf(FILE *fp,const char *format, ...);
// 写入字符，将字符c写入fp指向的文件中
int fputc(int c, FILE *fp);
```

### 关闭文件

### 移动文件位置指针

- `rewind()`用来将文件指针移动到文件开头，原型为：

	```c
	
	void rewind (FILE *fp);
	```

- `fseek()`用来将文件指针移动到任意位置，原型为：

	```c
	
	int fseek (FILE *fp, long offset, int origin);
	```

	- `fp`：待移动的文件指针。
	- `offset`：偏移量，要移动的字节数。
	- `origin`：起始位置，文件开头、当前位置和文件末尾：

| 常量名   | 含义     | 常量值 |
| -------- | -------- | ------ |
| SEEK_SET | 文件开头 | 0      |
| SEEK_CUR | 当前位置 | 1      |
| SEEK_END | 文件末尾 | 2       |

## cpp文件流和读写文件

### 头文件

cpp风格读写文件标准库包含于头文件`<fstream>`。cpp标准库中有三个类可以用于文件操作，统称为文件流类，这三个类分别是：

- `ifstream`：用于从文件中读取数据。
- `ofstream`：用于向文件中写入数据。
- `fstream`：既可用于从文件中读取数据，又可用于向文件中写入数据。

![](attachments/Pasted%20image%2020220526000009.png)

### cpp读写文件示例

```cpp

#include <iostream>
#include <fstream>
using namespace std;

int main()
{
    string line;

    // ##文件读写示例1##
    // 打开文件，并将test.txt文件与输出文件流对象fout关联
    ofstream fout("D:/test.txt", ios::out | ios::trunc);
    // << 输出
    fout << "hello fstream";
    // 关闭文件，切断和文件流对象的关联
    fout.close();

    // 打开文件，并将test.txt文件与输入文件流对象fin关联
    ifstream fin("D:/test.txt", ios::in);
    // >> 输入
    while (fin >> line) {
        cout << line << endl;
    } 
    fin.close();

    // ##文件读写示例2##
    const char *text = "hello world";
    // 创建一个fstream类对象
    fstream fs;
    // 将test.txt文件和fs文件流关联
    fs.open("E:/test.txt", ios::out);
    // 向test.txt文件中写入字符串
    fs.write(text, 12);
    // 关闭文件，切断和文件流对象的关联
    fs.close();
   
    return 0;
}
```

### fstream常用成员方法

![](attachments/Pasted%20image%2020220526112240.png)

### 打开模式

![](attachments/Pasted%20image%2020220526112329.png)

### 移动文件读写指针

- `ifstream`类和`fstream`类有`seekg()`成员函数，可以设置文件读指针的位置。
- `ofstream`类和`fstream`类有`seekp`成员函数，可以设置文件写指针的位置。

函数原型如下：

```cpp

ostream & seekp (int offset, int mode);
istream & seekg (int offset, int mode);
```

其中，`mode`指定移动指针的起始位置，可选项为文件开头`ios::beg`、文件当前位置`ios::cur`和文件末尾`ios::end`，当`mode`设置为文件末尾时，参数`offset`只能是0或者负数。

相对应地，可以通过`tellg()`和`tellp()`成员函数获取文件读指针和写指针的位置。

> [C/C++ 操作文件](https://blog.oneneko.com/posts/2021/06/11/operate-file-c-cpp.html)
> [C++文件操作](http://c.biancheng.net/cplus/60/)
> [C++ 文件和流](https://www.runoob.com/cplusplus/cpp-files-streams.html)
> [File Handling through C++ Classes](https://www.geeksforgeeks.org/file-handling-c-classes/)
> [Input/output with files](https://www.cplusplus.com/doc/tutorial/files/)
