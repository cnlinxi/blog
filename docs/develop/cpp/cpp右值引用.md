# cpp右值引用和std::move

## 左值和右值的区别

- 左值可以取地址，位于等号左边。比如变量。
- 右值无法取地址，位于等号右边。比如字面值、临时值。

```cpp

int a = 5; // 变量a是左值，可以通过&取地址，位于等号左边；5是右值，无法取地址，位于等号右边
A a = A(); // A是结构体名。对象a是左值；A()是临时值，是右值。
```

## 左值引用和右值引用

引用本质是别名，使用引用传参时可避免拷贝，从而提高性能。

1. 左值引用`&`能指向左值，不能指向右值。

```cpp

int a = 5;
int &ref_a = a; // 编译通过，左值引用
int &ref_a = 5; // 编译失败，左值引用无法指向右值
```

2. 右值引用`&&`可以指向右值，不能指向左值。

```cpp

int &&ref_a_right = 5; // 编译通过，右值引用

int a = 5;
int &&ref_a_right = a; // 编译失败，右值引用无法指向左值
```

## std::move

`std::move`将左值强制转换为右值，让右值引用可以指向左值，等同于类型转换：

```cpp

// std::move等同于：将左值强转到右值的类型转换
std::move == static_cast<T&&>(lvalue);
```

可移动对象在“需要拷贝且*被拷贝者不再需要*”的场景下，可以使用`std::move`触发移动语义，提升性能。

```cpp

class Array {
public:
    ......
 
    // 右值引用为参数的`移动构造函数`
    Array(Array&& temp_array) {
        data_ = temp_array.data_;
        size_ = temp_array.size_;
        // 为防止temp_array析构时delete data，提前置空其data_      
        temp_array.data_ = nullptr;
    }
     
 
public:
    int *data_;
    int size_;
};

// 使用
int main(){
    Array a;  // 对象a是左值，本身右值引用无法指向对象a
 
    // 做一些操作
    .....
     
    // 左值a，用std::move转化为右值
    Array b(std::move(a));
}
```

可以看到，`std::move`本身只做类型转换，对性能无影响。 但是可以在类中实现**移动**语义，只转移内部对象所有权（浅拷贝），避免深拷贝，从而提高性能。

STL的很多容器中，都实现了以右值引用为参数的`移动构造函数`和`移动赋值重载函数`：

```cpp

// std::vector方法定义
void push_back (const value_type& val);
void push_back (value_type&& val);
void emplace_back (Args&&... args);

// 使用
std::string str1 = "aacasxs";
std::vector<std::string> vec;

vec.push_back(str1); // 传统方法，copy
vec.push_back(std::move(str1)); // 调用移动语义的push_back方法，避免拷贝，str1会失去原有值，变成空字符串
vec.emplace_back(std::move(str1)); // emplace_back效果相同，str1会失去原有值
vec.emplace_back("axcsddcas"); // 当然可以直接接右值
```

> [一文读懂C++右值引用和std::move](https://zhuanlan.zhihu.com/p/335994370)