# STL的map

## 键值对容器map和unordered_map

- map基于红黑树实现，红黑树是一种自平衡的二叉树，保证有序且最坏情况下的运行时间。
- unordered_map基于hash_table实现。

在需要有序性或者对单次查询有时间要求的应用场景下，应使用map，其余情况应使用unordered_map。

> https://zhuanlan.zhihu.com/p/48066839

map和unordered_map使用方法基本相同。

> http://www.cplusplus.com/reference/unordered_map/unordered_map/
> https://blog.csdn.net/qq_21997625/article/details/84672775
> https://blog.csdn.net/shuzfan/article/details/53115922#%E4%BA%8C-%E6%8F%92%E5%85%A5%E6%93%8D%E4%BD%9C

## map的使用示例

```c++
#include <iostream>
#include <unordered_map>
#include <string>
int main(int argc, char **argv) {
    std::unordered_map<int, std::string> map;
    map.insert(std::make_pair(1, "Scala"));
    map.insert(std::make_pair(2, "Haskell"));
    std::unordered_map<int, std::string>::iterator it;
    if ((it = map.find(6)) != map.end()) {
        std::cout << it->second << std::endl;
    }
    return 0;
}
```