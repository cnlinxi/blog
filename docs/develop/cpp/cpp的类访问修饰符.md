# cpp的类访问修饰符

## 类成员访问修饰符

- 公有成员。使用`public`修饰，类内部可以访问类中public和protected成员，但是类外只能通过类对象访问类的public成员。
- 受保护成员。使用`protected`修饰，类外不能通过类对象访问类的成员，但可以在类中添加公有成员函数接口间接访问类中的protected成员。
- 私有成员。使用`private`修饰，和`protected`相似，但`protected`成员在派生类（也即子类）中是可访问的。

一个类可以有多个`public`、`protected`或`private`修饰，类的默认访问修饰符为`private`，结构体的默认访问修饰符为`public`。

设置私有成员的机制叫作“隐藏”。“隐藏”的一个目的就是强制对成员变量的访问一定要通过成员函数进行。这样做的好处是，如果以后修改了成员变量的类型等属性，只需要更改成员函数即可；否则，所有直接访问成员变量的语句都需要修改。

### 示例

```cpp

class SuperClass {
private:
  double private_member_;

protected:
  double protected_member_;

public:
  double public_member_;
  void SetPrivateMember(double val);
  double GetPrivateMember();
};

void SuperClass::SetPrivateMember(double val) { private_member_ = val; }

double SuperClass::GetPrivateMember() { return private_member_; }

class SubClass : SuperClass // SubClass是派生类
{
public:
  // 基类的受保护成员可以在派生类中访问，这是protected和private的主要区别
  void SetProtectedMember(double val);
  double GetProtected();
};

void SubClass::SetProtectedMember(double val) { protected_member_ = val; }

double SubClass::GetProtected() { return protected_member_; }

int main() {
  SuperClass super_class; // 基类
  SubClass sub_class;     // 派生类
  // 编译通过，公有成员可以通过类对象访问
  super_class.public_member_ = 10;
  // 编译失败，私有成员或受保护成员不能通过类对象访问
  super_class.private_member_ = 10;
  // 编译通过，私有成员或受保护成员可以通过公有函数接口访问
  super_class.SetPrivateMember(10);
  // 编译通过，受保护成员和私有成员的区别在于，受保护成员可以在派生类中访问
  sub_class.SetProtectedMember(10);

  return 0;
}
```

## cpp类的继承控制

有`public`、`protected`、`private`三种继承方式，主要改变基类成员在子类中的访问属性。

- `public`继承：基类`public`成员、`protected`成员、`private`成员的访问属性在派生类中分别变成：`public`、`protected`、`private`。
- `protected`继承：基类`public`成员、`protected`成员、`private`成员的访问属性在派生类中分别变成：`protected`、`protected`、`private`。
- `private`继承：基类`public`成员、`protected`成员、`private`成员的访问属性在派生类中分别变成：`private`、`private`、`private`。

> [C++ 类访问修饰符](https://www.runoob.com/cplusplus/cpp-class-access-modifiers.html)
> [C++ public、protected 、 private和friend（最通俗易懂）](https://blog.csdn.net/a3192048/article/details/82191795)
> [CPP公有继承、保护继承以及私有继承](https://www.jianshu.com/p/00adb0c0e6d6)
