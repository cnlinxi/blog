# string操作

## string基础操作

- string的头文件：`#include <string>`
- string拼接：`string1+=string2`，注意，由于`+=`的运算符重载是引用，不同于`string1=string1+string2`，`+=`效率更高
- int/float等转换为string：`to_string(val)`
- string比较大小：`string1 < string2`
- c++排序：

```c++
#include <algorithm>

sort(begin,last,compare);

bool compare(int x1,int x2){
    return x1>x2;
}
sort(vec.begin(),vec.end(),compare);
```

## string迭代器

### begin/end/rbegin/rend的用法

- iterator begin：begin()函数返回一个迭代器,指向字符串的第一个元素
- iterator end：end()函数返回一个迭代器，指向字符串的末尾(最后一个字符的下一个位置)
- iterator rbegin：rbegin()返回一个逆向迭代器，指向字符串的最后一个字符
- iterator rend：rend()函数返回一个逆向迭代器，指向字符串的开头（第一个字符的前一个位置）

### find_if和find_if_not的用法

find_if() 函数会根据指定的查找规则，在指定区域内查找第一个符合该函数要求（使函数返回 true）的元素。

包含于头文件：`#include <algorithm>`

find_if() 函数的语法格式如下：

```c++
InputIterator find_if(InputIterator first, InputIterator last, UnaryPredicate pred);
```

其中，first 和 last 都为输入迭代器，其组合 [first, last) 用于指定要查找的区域；pred 用于自定义查找规则。

该函数会返回一个输入迭代器，当查找成功时，该迭代器指向的是第一个符合查找规则的元素；反之，如果 find_if() 函数查找失败，则该迭代器的指向和 last 迭代器相同。

注意：`find`用于查找特定的元素，而`find_if`根据指定的查找规则，在指定区域内查找第一个符合该函数要求（使函数返回 true）的元素，因此后者`find_if`可通过传入lambda自定义查找目标，更为灵活。

> https://www.cnblogs.com/pandamohist/p/13854705.html

### 通过reverse_iterator的base()得到iterator

end()与rbegin()、begin()与rend()不在同一个位置，这是为了保证区间保持左开右闭的原则，删除、插入等函数需要传入正向迭代器，所以需要用reverse_iterator逆向迭代器的成员函数base()将其转换为iterator。

### string的字符串拼接

字符串拼接时，`+=`和`append`基本没什么不同。

> https://www.jianshu.com/p/c86d38db63ce

### 示例：去除空格

string去除前后空格、中间多余空格

```c++
#include <string>
#include <iostream>
#include <algorithm>

using namespace std;

void trimLeft(string &s){
    s.erase(s.begin(),find_if(s.begin(),s.end(),[](char ch){
        return !isspace(ch);
    }));
}

void trimRight(string &s){
    s.erase(find_if(s.rbegin(),s.rend(),[](char ch){
        return !isspace(ch);
    }).base(),s.end());
}

void trim(string &s){
    trimLeft(s);
    trimRight(s);
}

string removeSurpluseStr(string s){
    string ret="";
    bool isPrevSpace=false;
    for(char ch:s){
        if(ch==' '){
            if(isPrevSpace){
                continue;
            }
            else{
                ret+=ch;
                isPrevSpace=true;
            }
        }else{
            ret+=ch;
        }
    }
    return ret;
}
```

## string读写流

### stringstream

用于将string读入流stream，必须引用头文件：

```c++
#include <sstream>
```

### getline

getline用于读取字符流，并将其存储到string，可以传入分隔符

> https://www.cplusplus.com/reference/string/string/getline/

### 示例：分割字符串

```c++
#include <iostream>
#include <sstream>
#include <vector>

using std::string;
using std::vector;
using std::cout;
using std::endl;

string StringTrim(string str,string trimed_char){
    if(str.empty()){
        return str;
    }
    str.erase(0,str.find_first_not_of(trimed_char));
    str.erase(str.find_last_not_of(trimed_char)+1);
    return str;
}

vector<string> StringSplit(string str){
    vector<string> vecString;
    if(str.empty()){
        return vecString;
    }
    str=StringTrim(str,"[");
    str=StringTrim(str,"]");
    std::stringstream ss;
    ss.str(str);
    string item;
    while(getline(ss,item,',')){
        if(!item.empty()){
            vecString.push_back(item);
        }
    }
    return vecString;
}

vector<int> StringToInt(vector<string> strs){
    vector<int> vecInt;
    for(auto item:strs){
        vecInt.push_back(stoi(item));
    }
    return vecInt;
}


int main(){
    string data="[1,2,3,3]";
    vector<string> strs=StringSplit(data);
    for(auto item:strs){
        cout<<item<<"\n";
    }
    return 1;
}
```

