# python的取反运算符

`~`：按位取反运算符，对数据的二进制位取反，将数据对应的二进制位0变为1，1变为0.

因此，`~x`相当于`-x-1`。

- 示例：

```python
x=np.array([1,0,1])
print(~x)  # 结果为[-2,-1,-2]
```

> https://blog.csdn.net/weixin_43915860/article/details/107656101