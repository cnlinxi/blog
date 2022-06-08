# cpp中的int_t

## int_t

int_t是一类基本数据类型的缩写，并非新的数据类型。由于跨平台，不同的平台会有不同的字长，所以利用预编译和`typedef`可以最有效地维护代码。

```c

// stdint.h
/* 7.18.1.1  Exact-width integer types */
typedef signed char int8_t;
typedef unsigned char   uint8_t;
typedef short  int16_t;
typedef unsigned short  uint16_t;
typedef int  int32_t;
typedef unsigned   uint32_t;
__MINGW_EXTENSION typedef long long  int64_t;
__MINGW_EXTENSION typedef unsigned long long   uint64_t;
...
```

> [C中int8_t、int16_t、int32_t、int64_t、uint8_t、size_t、ssize_t区别_yz930618的博客-CSDN博客_int8_t](https://blog.csdn.net/yz930618/article/details/84785970)