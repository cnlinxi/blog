# STL的vector

## 常见成员函数

| 方法         | 含义                                 |
| ------------ | ------------------------------------ |
| push_back    | 在容器最后一个元素之后添加一个新元素 |
| emplace_back | 与push_back同义，容器尾部追加新元素  |
| resize       | 调整容器大小，如果该函数有第二个参数，则新元素初始化为该参数的副本                                     |


### 示例：创建二维数组

1. STL的vector版本

```cpp

vector<vector<int> > get2DVector(int m,int n){
    vector<vector<int> > res(m);
    for(int i=0;i<m;++i){
        res[i].resize(n);
    }
    return res;
}

void print2DVector(vector<vector<int> > &vec){
    int m=vec.size();
    if(m<=0) return;
    int n=vec[0].size();
    if(n<=0) return;
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            cout<<vec[i][j]<<"\t";
        }
        cout<<endl;
    }
}
```

2. 动态内存版本

```cpp

int** get2DArrary(int m,int n){
    int **p=new int*[m];
    for(int i=0;i<m;++i){
        p[i]=new int[n];
    }
    return p;
}

void print2DArray(int** p,int m,int n){
    if(p==nullptr||m<=0||n<=0) return;
    for(int i=0;i<m;++i){
        for(int j=0;j<n;++j){
            cout<<p[i][j]<<"\t";
        }
        cout<<endl;
    }
}
```

