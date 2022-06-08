# cpp宽字符wchar

`std::string`基于char，8bit。`std::wstring`基于`wchar_t`，Linux上4字节，Windows上2字节。在Windows上一般要用`std::wstring`而非`std::string`。

```cpp

// wenet/runtime/core/utils/string.cc
#ifdef _MSC_VER
std::wstring ToWString(const std::string& str) {
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t* p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}
#endif
```

> [c++ - std::wstring VS std::string - Stack Overflow](https://stackoverflow.com/questions/402283/stdwstring-vs-stdstring)