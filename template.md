- [ACM 模板](#acm-%E6%A8%A1%E6%9D%BF)
  - [数据结构](#%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84)
    - [并查集](#%E5%B9%B6%E6%9F%A5%E9%9B%86)
    - [树状数组](#%E6%A0%91%E7%8A%B6%E6%95%B0%E7%BB%84)
    - [线段树](#%E7%BA%BF%E6%AE%B5%E6%A0%91)
    - [可持久化线段树](#%E5%8F%AF%E6%8C%81%E4%B9%85%E5%8C%96%E7%BA%BF%E6%AE%B5%E6%A0%91)
  - [图论](#%E5%9B%BE%E8%AE%BA)
    - [最短路](#%E6%9C%80%E7%9F%AD%E8%B7%AF)
      - [dijkstra](#dijkstra)
    - [树上问题](#%E6%A0%91%E4%B8%8A%E9%97%AE%E9%A2%98)
      - [最近公公祖先](#%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%AC%E7%A5%96%E5%85%88)
        - [倍增法](#%E5%80%8D%E5%A2%9E%E6%B3%95)
        - [树剖](#%E6%A0%91%E5%89%96)
      - [树链剖分](#%E6%A0%91%E9%93%BE%E5%89%96%E5%88%86)
    - [强连通分量](#%E5%BC%BA%E8%BF%9E%E9%80%9A%E5%88%86%E9%87%8F)
    - [拓扑排序](#%E6%8B%93%E6%89%91%E6%8E%92%E5%BA%8F)
  - [字符串](#%E5%AD%97%E7%AC%A6%E4%B8%B2)
    - [kmp](#kmp)
    - [哈希](#%E5%93%88%E5%B8%8C)
    - [manacher](#manacher)
  - [数学](#%E6%95%B0%E5%AD%A6)
    - [线性筛法](#%E7%BA%BF%E6%80%A7%E7%AD%9B%E6%B3%95)
    - [分解质因数](#%E5%88%86%E8%A7%A3%E8%B4%A8%E5%9B%A0%E6%95%B0)
    - [组合数](#%E7%BB%84%E5%90%88%E6%95%B0)
    - [盒子与球](#%E7%9B%92%E5%AD%90%E4%B8%8E%E7%90%83)
    - [线性基](#%E7%BA%BF%E6%80%A7%E5%9F%BA)
    - [矩阵快速幂](#%E7%9F%A9%E9%98%B5%E5%BF%AB%E9%80%9F%E5%B9%82)
  - [计算几何](#%E8%AE%A1%E7%AE%97%E5%87%A0%E4%BD%95)
  - [杂项](#%E6%9D%82%E9%A1%B9)
    - [高精度](#%E9%AB%98%E7%B2%BE%E5%BA%A6)
    - [扫描线](#%E6%89%AB%E6%8F%8F%E7%BA%BF)
    - [模运算](#%E6%A8%A1%E8%BF%90%E7%AE%97)
    - [分数](#%E5%88%86%E6%95%B0)
    - [表达式求值](#%E8%A1%A8%E8%BE%BE%E5%BC%8F%E6%B1%82%E5%80%BC)
    - [对拍](#%E5%AF%B9%E6%8B%8D)
      - [linux/Mac](#linuxmac)
      - [windows](#windows)
    - [开栈](#%E5%BC%80%E6%A0%88)
      - [windowns](#windowns)
      - [linux](#linux)
    - [日期](#%E6%97%A5%E6%9C%9F)

# ACM 模板

## 数据结构

### 并查集

```cpp
struct dsu{
    int n;
    vector<int> fa;
    dsu(int _n) :n(_n){
        fa.resize(n + 1);
        iota(fa.begin(), fa.end(), 0);
    }
    int find(int x){
        return x == fa[x] ? x : fa[x] = find(fa[x]);
    }
    int merge(int x, int y){
        int fax = find(x), fay = find(y);
        if(fax == fay)return 0; // 一个集合
        return fa[find(x)] = find(y); // 合并到哪个集合了
    }
};
```

### 树状数组

```cpp
#define lowbit(x) ((x)&(-(x)))
template<class T>
struct Fenwick_tree{
    Fenwick_tree(int size){
        n = size;
        tree.assign(n + 1, 0);
    }
    T query(int l, int r){
        auto query = [&](int pos){
            T res = 0;
            while(pos){ res += tree[pos]; pos -= lowbit(pos); }
            return res;
        };
        return query(r) - query(l - 1);
    }
    void update(int pos, T num){
        while(pos <= n){ tree[pos] += num; pos += lowbit(pos); }
    }
private:
    int n;
    vector<T> tree;
};
```

### 线段树

```cpp
template <class Data, class Num>
struct Segment_Tree{
    inline void update(int l, int r, Num x){ update(1, l, r, x); }
    inline Data query(int l, int r){ return query(1, l, r); }
    Segment_Tree(vector<Data>& a){
        n = a.size();
        tree.assign(n * 4 + 1, {});
        build(a, 1, 1, n);
    }
private:
    int n;
    struct Tree{ int l, r; Data data; };
    vector<Tree> tree;
    inline void pushup(int pos){
        tree[pos].data = tree[pos << 1].data + tree[pos << 1 | 1].data;
    }
    inline void pushdown(int pos){
        tree[pos << 1].data = tree[pos << 1].data + tree[pos].data.lazytag;
        tree[pos << 1 | 1].data = tree[pos << 1 | 1].data + tree[pos].data.lazytag;
        tree[pos].data.lazytag = Num::zero();
    }
    void build(vector<Data>& a, int pos, int l, int r){
        tree[pos].l = l; tree[pos].r = r;
        if(l == r){ tree[pos].data = a[l - 1]; return; }
        int mid = (tree[pos].l + tree[pos].r) >> 1;
        build(a, pos << 1, l, mid);
        build(a, pos << 1 | 1, mid + 1, r);
        pushup(pos);
    }
    void update(int pos, int& l, int& r, Num& x){
        if(l > tree[pos].r || r < tree[pos].l)return;
        if(l <= tree[pos].l && tree[pos].r <= r){ tree[pos].data = tree[pos].data + x; return; }
        pushdown(pos);
        update(pos << 1, l, r, x); update(pos << 1 | 1, l, r, x);
        pushup(pos);
    }
    Data query(int pos, int& l, int& r){
        if(l > tree[pos].r || r < tree[pos].l)return Data::zero();
        if(l <= tree[pos].l && tree[pos].r <= r)return tree[pos].data;
        pushdown(pos);
        return query(pos << 1, l, r) + query(pos << 1 | 1, l, r);
    }
};
struct Num{
    ll add;
    inline static Num zero(){ return { 0 }; }
    inline Num operator+(Num b){ return { add + b.add }; }
};
struct Data{
    ll sum, len;
    Num lazytag;
    inline static Data zero(){ return { 0,0,Num::zero() }; }
    inline Data operator+(Num b){ return { sum + len * b.add,len,lazytag + b }; }
    inline Data operator+(Data b){ return { sum + b.sum,len + b.len,Num::zero() }; }
};
```

### 可持久化线段树

```cpp
constexpr int MAXN = 200000;
vector<int> root(MAXN << 5);
struct Persistent_seg {
    int n;
    struct Data {
        int ls, rs;
        int val;
    };
    vector<Data> tree;
    Persistent_seg(int n, vector<int>& a):n(n) { root[0] = build(1, n, a); }
    int build(int l, int r, vector<int>& a) {
        if (l == r) {
            tree.push_back({ 0,0,a[l] });
            return tree.size() - 1;
        }
        int mid = l + r >> 1;
        int ls = build(l, mid, a), rs = build(mid + 1, r, a);
        tree.push_back({ ls,rs,tree[ls].val + tree[rs].val });
        return tree.size() - 1;
    }
    int update(int rt, const int& idx, const int& val, int l, int r) {
        if (l == r) {
            tree.push_back({ 0,0,tree[rt].val + val });
            return tree.size() - 1;
        }
        int mid = l + r >> 1, ls = tree[rt].ls, rs = tree[rt].rs;
        if (idx <= mid)ls = update(ls, idx, val, l, mid);
        else rs = update(rs, idx, val, mid + 1, r);
        tree.push_back({ ls,rs,tree[ls].val + tree[rs].val });
        return tree.size() - 1;
    }
    int query(int rt1, int rt2, int k, int l, int r) {
        if (l == r)return l;
        int mid = l + r >> 1;
        int lcnt = tree[tree[rt2].ls].val - tree[tree[rt1].ls].val;
        if (k <= lcnt)return query(tree[rt1].ls, tree[rt2].ls, k, l, mid);
        else return query(tree[rt1].rs, tree[rt2].rs, k - lcnt, mid + 1, r);
    }
};
```

## 图论

存图

```cpp
struct Graph{
    int n;
    struct Edge{ int to, w; };
    vector<vector<Edge>> graph;
    Graph(int _n){ n = _n; graph.assign(n + 1, vector<Edge>()); };
    void add(int u, int v, int w){ graph[u].push_back({ v,w }); }
};
```

### 最短路

#### dijkstra

```cpp
void dij(Graph& graph, vector<int>& dis, int t){
    vector<int> visit(graph.n + 1, 0);
    priority_queue<pair<int, int>> que;
    dis[t] = 0;
    que.emplace(0, t);
    while(!que.empty()){
        int u = que.top().second; que.pop();
        if(visit[u])continue;
        visit[u] = 1;
        for(auto& [to, w] : graph.graph[u]){
            if(dis[to] > dis[u] + w){
                dis[to] = dis[u] + w;
                que.emplace(-dis[to], to);
            }
        }
    }
}
```

### 树上问题

#### 最近公公祖先

##### 倍增法

```cpp
vector<int> dep;
vector<array<int, 21>> fa;
dep.assign(n + 1, 0);
fa.assign(n + 1, array<int, 21>{});
void binary_jump(int root){
    function<void(int)> dfs = [&](int t){
        dep[t] = dep[fa[t][0]] + 1;
        for(auto& [to] : graph[t]){
            if(to == fa[t][0])continue;
            fa[to][0] = t;
            dfs(to);
        }
    };
    dfs(root);
    for(int j = 1; j <= 20; j++)
        for(int i = 1; i <= n; i++)
            fa[i][j] = fa[fa[i][j - 1]][j - 1];
}
int lca(int x, int y){
    if(dep[x] < dep[y])swap(x, y);
    for(int i = 20; i >= 0; i--){
        if(dep[fa[x][i]] >= dep[y])x = fa[x][i];
    }
    if(x == y)return x;
    for(int i = 20; i >= 0; i--){
        if(fa[x][i] != fa[y][i]){
            x = fa[x][i];
            y = fa[y][i];
        }
    }
    return fa[x][0];
}
```

##### 树剖

```cpp
int lca(int x, int y){
    while(top[x] != top[y]){
        if(dep[top[x]] < dep[top[y]])swap(x, y);
        x = fa[top[x]];
    }
    if(dep[x] < dep[y])swap(x, y);
    return y;
}
```

#### 树链剖分

```cpp
vector<int> fa, siz, dep, son, dfn, rnk, top;
fa.assign(n + 1, 0);
siz.assign(n + 1, 0);
dep.assign(n + 1, 0);
son.assign(n + 1, 0);
dfn.assign(n + 1, 0);
rnk.assign(n + 1, 0);
top.assign(n + 1, 0);
void hld(int root){
    function<void(int)> dfs1 = [&](int t){
        dep[t] = dep[fa[t]] + 1;
        siz[t] = 1;
        for(auto& [to, w] : graph[t]){
            if(to == fa[t])continue;
            fa[to] = t;
            dfs1(to);
            if(siz[son[t]] < siz[to])son[t] = to;
            siz[t] += siz[to];
        }
    }; dfs1(root);
    int dfn_tail = 0;
    for(int i = 1; i <= n; i++)top[i] = i;
    function<void(int)> dfs2 = [&](int t){
        dfn[t] = ++dfn_tail;
        rnk[dfn_tail] = t;
        if(!son[t])return;
        top[son[t]] = top[t];
        dfs2(son[t]);
        for(auto& [to, w] : graph[t]){
            if(to == fa[t] || to == son[t])continue;
            dfs2(to);
        }
    }; dfs2(root);
}
```

### 强连通分量

```cpp
void tarjan(Graph& g1, Graph& g2){
    int dfn_tail = 0, cnt = 0;
    vector<int> dfn(g1.n + 1, 0), low(g1.n + 1, 0), exist(g1.n + 1, 0), belong(g1.n + 1, 0);
    stack<int> sta;
    function<void(int)> dfs = [&](int t){
        dfn[t] = low[t] = ++dfn_tail;
        sta.push(t); exist[t] = 1;
        for(auto& [to] : g1.graph[t]){
            if(!dfn[to]){
                dfs(to);
                low[t] = min(low[t], low[to]);
            }
            else if(exist[to])low[t] = min(low[t], dfn[to]);
        }
        if(dfn[t] == low[t]){
            cnt++;
            while(int temp = sta.top()){
                belong[temp] = cnt;
                exist[temp] = 0;
                sta.pop();
                if(temp == t)break;
            }
        }
    };
    for(int i = 1; i <= g1.n; i++)if(!dfn[i])dfs(i);
    g2 = Graph(cnt);
    for(int i = 1; i <= g1.n; i++)g2.w[belong[i]] += g1.w[i];
    for(int i = 1; i <= g1.n; i++)
        for(auto& [to] : g1.graph[i])
            if(belong[i] != belong[to])g2.add(belong[i], belong[to]);
}
```

### 拓扑排序

```cpp
void toposort(Graph& g, vector<int>& dis){
    vector<int> in(g.n + 1, 0);
    for(int i = 1; i <= g.n; i++)
        for(auto& [to] : g.graph[i])in[to]++;
    queue<int> que;
    for(int i = 1; i <= g.n; i++)
        if(!in[i]){
            que.push(i);
            dis[i] = g.w[i]; // dp
        }
    while(!que.empty()){
        int u = que.front(); que.pop();
        for(auto& [to] : g.graph[u]){
            in[to]--;
            dis[to] = max(dis[to], dis[u] + g.w[to]); // dp
            if(!in[to])que.push(to);
        }
    }
}
```

## 字符串

### kmp

```cpp
vector<int> kmp(string&& s) {
    vector<int> next(s.size(), -1);
    for (int i = 1, j = -1; i < s.size(); i++) {
        while (j >= 0 && s[i] != s[j + 1])j = next[j];
        if (s[i] == s[j + 1])j++;
        next[i] = j;
    }
    return next;
}
```

### 哈希

```cpp
constexpr int N = 2e6;
constexpr ll mod[2] = { 2000000011, 2000000033 }, base[2] = { 20011,20033 };
vector<array<ll, 2>> pow_base(N);

pow_base[0][0] = pow_base[0][1] = 1;
for(int i = 1; i < N; i++){
    pow_base[i][0] = pow_base[i - 1][0] * base[0] % mod[0];
    pow_base[i][1] = pow_base[i - 1][1] * base[1] % mod[1];
}

struct Hash{
    int size;
    vector<array<ll, 2>> hash;
    Hash(){}
    Hash(const string& s){
        size = s.size();
        hash.resize(size);
        hash[0][0] = hash[0][1] = s[0];
        for(int i = 1; i < size; i++){
            hash[i][0] = (hash[i - 1][0] * base[0] + s[i]) % mod[0];
            hash[i][1] = (hash[i - 1][1] * base[1] + s[i]) % mod[1];
        }
    }
    array<ll, 2> operator[](const array<int, 2>& range)const{
        int l = range[0], r = range[1];
        if(l == 0)return hash[r];
        auto single_hash = [&](bool flag){
            return (hash[r][flag] - hash[l - 1][flag] * pow_base[r - l + 1][flag] % mod[flag] + mod[flag]) % mod[flag];
        };
        return { single_hash(0),single_hash(1) };
    }
};
```

### manacher

```cpp
void manacher(const string& _s, vector<int>& r){
    string s(_s.size() * 2 + 1, '$');
    for(int i = 0; i < _s.size(); i++)s[2 * i + 1] = _s[i];
    r.resize(_s.size() * 2 + 1);
    for(int i = 0, maxr = 0, mid = 0; i < s.size(); i++){
        if(i < maxr)r[i] = min(r[mid * 2 - i], maxr - i);
        while(i - r[i] - 1 >= 0 && i + r[i] + 1 <s.size() && s[i - r[i] - 1] == s[i + r[i] + 1]) ++r[i];
        if(i + r[i] > maxr) maxr = i + r[i], mid = i;
    }
}
```

## 数学

### 线性筛法

```cpp
auto [min_prime,primes] = [](){
    constexpr int N = 10000000;
    vector<int> min_prime(N + 1, 0), primes;
    for (int i = 2; i <= N; i++) {
        if (min_prime[i] == 0) {
            min_prime[i] = i;
            primes.push_back(i);
        }
        for (auto& prime : primes) {
            if (prime > min_prime[i] || prime > N / i)break;
            min_prime[prime * i] = prime;
        }
    }
    return tuple{ min_prime, primes };
}();
```

### 分解质因数

```cpp
void num_primes(int num, vector<int>& ans) {
    for (auto& prime : primes) {
        if (prime > num / prime)break;
        if (num % prime == 0) {
            while (num % prime == 0)num /= prime;
            ans.push_back(prime);
        }
    }
    if (num > 1)ans.push_back(num);
}
```

### 组合数

```cpp
modint C(int n, int m){
    if(m == 0)return 1;
    if(n <= mod)return factorial[n] * factorial[m].inv() * factorial[n - m].inv();
    else return C(n % mod, m % mod) * C(n / mod, m / mod); // n >= mod 时需要这个
}
```

### 盒子与球

$n个球,m个盒$

|球同|盒同|可空|公式|
|---|---|---|---|
|✓|✓|✓|$f_{n,m}=f_{n-1,m-1}+f_{n-m,m}$|
|✓|✓|✕|$f_{n-m,m}$|
|✕|✓|✓|$\Sigma_{i=1}^{m}g_{n,i}$|
|✕|✓|✕|$g_{n,m}=g_{n-1,m-1}+m*g_{n-1,m}$|
|✓|✕|✓|$C_{n+m-1}^{m-1}$|
|✓|✕|✕|$C_{n-1}^{m-1}$|
|✕|✕|✓|$m^n$|
|✕|✕|✕|$m!*g_{n,m}$|

### 线性基

```cpp
// 线性基
struct basis{
    array<unsigned ll, 64> p{};

    // 将x插入此线性基中
    void insert(unsigned ll x){
        for(int i = 63; i >= 0; i--){
            if((x >> i) & 1){
                if(p[i])x ^= p[i];
                else{
                    p[i] = x; break;
                }
            }
        }
    }

    // 将另一个线性基插入此线性基中
    void insert(basis other){
        for(int i = 0; i <= 63; i++){
            if(!other.p[i])continue;
            insert(other.p[i]);
        }
    }

    // 最大异或值
    unsigned ll max_basis(){
        unsigned ll res = 0;
        for(int i = 63; i >= 0; i--){
            if((res ^ p[i]) > res)res ^= p[i];
        }
        return res;
    }
};
```

### 矩阵快速幂

```cpp
constexpr ll mod = 2147493647;
struct Mat{
    int n, m;
    vector<vector<ll>> mat;
    Mat(int n, int m) :n(n), m(n), mat(n, vector<ll>(m, 0)){}
    Mat(vector<vector<ll>> mat) :n(mat.size()), m(mat[0].size()), mat(mat){}
    Mat operator*(const Mat& other){
        assert(m == other.n);
        Mat res(n, other.m);
        for(int i = 0; i < res.n; i++)
            for(int j = 0; j < res.m; j++)
                for(int k = 0; k < m; k++)
                    res.mat[i][j] = (res.mat[i][j] + mat[i][k] * other.mat[k][j] % mod) % mod;
        return res;
    }
};
Mat ksm(Mat a, ll b){
    assert(a.n == a.m);
    Mat res(a.n, a.m);
    for(int i = 0; i < res.n; i++)res.mat[i][i] = 1;
    while(b){
        if(b & 1)res = res * a;
        b >>= 1;
        a = a * a;
    }
    return res;
}
```

## 计算几何

```cpp
constexpr double PI = 3.141592653589793116;
constexpr double eps = 1e-8;
using T = double;

// 两浮点数是否相等
bool equal(const T& a, const T& b){
    return abs(a - b) < eps;
}

// 向量
struct vec{
    T x, y;
    vec() :x(0), y(0){}
    vec(const T& _x, const T& _y) :x(_x), y(_y){}

    // 模
    double length()const{
        return sqrt(x * x + y * y);
    }

    // 与x轴正方向的夹角
    double angle()const{
        double angle = atan2(y, x);
        if(angle < 0)angle += 2 * PI;
        return angle;
    }

    // 逆时针旋转
    void rotate(const double& theta){
        double temp = x;
        x = x * cos(theta) - y * sin(theta);
        y = y * cos(theta) + temp * sin(theta);
    }

    bool operator==(const vec& other)const{ return equal(x, other.x) && equal(y, other.y); }
    bool operator<(const vec& other)const{ return angle() == other.angle() ? x < other.x : angle() < other.angle(); }

    vec operator+(const vec& other)const{ return { x + other.x,y + other.y }; }
    vec operator-()const{ return { -x,-y }; }
    vec operator-(const vec& other)const{ return -other + (*this); }
    vec operator*(const T& other)const{ return { other * x,other * y }; }
    T operator*(const vec& other)const{ return x * other.x + y * other.y; }

    // 叉积 结果大于0，a在b的顺时针，小于0，a在b的逆时针，等于0共线，可能同向或反向，结果绝对值表示 a b形成的平行四边行的面积
    T operator^(const vec& other)const{ return x * other.y - y * other.x; }

    friend istream& operator>>(istream& input, vec& data){
        input >> data.x >> data.y;
        return input;
    }
    friend ostream& operator<<(ostream& output, const vec& data){
        output << fixed << setprecision(6);
        output << data.x << " " << data.y;
        return output;
    }
};

// 求两点间的距离
double distance(const vec& a, const vec& b){
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

// 求两向量夹角
double angle(const vec& a, const vec& b){
    double theta = abs(a.angle() - b.angle());
    if(theta > PI)theta = 2 * PI - theta;
    return theta;
}

// 计算多边形的面积，polygon里必须是存的相邻的点
T polygon_area(const vector<vec>& polygon){
    T ans = 0;
    for(int i = 1; i < polygon.size(); i++)ans += polygon[i - 1] ^ polygon[i];
    ans += polygon[polygon.size() - 1] ^ polygon[0];
    return abs(ans / 2);
}

// 直线
struct Line{
    vec point, direction;

    Line(){}
    Line(const vec& _point, const vec& _direction) :point(_point), direction(_direction){}
};

// 两直线是否垂直
bool perpendicular(const Line& a, const Line& b){
    return a.direction * b.direction == 0;
}

// 两直线是否平行
bool parallel(const Line& a, const Line& b){
    return (a.direction ^ b.direction) == 0;
}

// 两直线交点
vec intersection(const T& A, const  T& B, const T& C, const T& D, const T& E, const T& F){
    return { (B * F - C * E) / (A * E - B * D),(C * D - A * F) / (A * E - B * D) };
}

// 两直线交点
vec intersection(const Line& a, const Line& b){
    return intersection(a.direction.y, -a.direction.x, a.direction.x * a.point.y - a.direction.y * a.point.x,
        b.direction.y, -b.direction.x, b.direction.x * b.point.y - b.direction.y * b.point.x);
}
```

## 杂项

### 高精度

```cpp
struct bignum{
    string num;

    bignum() :num("0"){}
    bignum(const string& num) :num(num){ reverse(this->num.begin(), this->num.end()); }
    bignum(ll num) :num(to_string(num)){ reverse(this->num.begin(), this->num.end()); }

    bignum operator+(const bignum& other){
        bignum res;
        res.num.pop_back();
        res.num.reserve(max(num.size(), other.num.size()) + 1);
        for(int i = 0, j = 0, x; i < num.size() || i < other.num.size() || j; i++){
            x = j; j = 0;
            if(i < num.size())x += num[i] - '0';
            if(i < other.num.size())x += other.num[i] - '0';
            if(x >= 10)j = 1, x -= 10;
            res.num.push_back(x + '0');
        }
        res.num.capacity();
        return res;
    }

    bignum operator*(const bignum& other){
        vector<int> res(num.size() + other.num.size() - 1, 0);
        for(int i = 0; i < num.size(); i++){
            for(int j = 0; j < other.num.size(); j++){
                res[i + j] += (num[i] - '0') * (other.num[j] - '0');
            }
        }
        int g = 0;
        for(int i = 0; i < res.size(); i++){
            res[i] += g;
            g = res[i] / 10;
            res[i] %= 10;
        }
        while(g){
            res.push_back(g % 10);
            g /= 10;
        }
        int lim = res.size();
        while(lim > 1 && res[lim - 1] == 0)lim--;
        bignum res2;
        res2.num.resize(lim);
        for(int i = 0; i < lim; i++)res2.num[i] = res[i] + '0';
        return res2;
    }

    bool operator<(const bignum& other){
        if(num.size() == other.num.size())
            for(int i = num.size() - 1; i >= 0; i--)
                if(num[i] == other.num[i])continue;
                else return num[i] < other.num[i];
        return num.size() < other.num.size();
    }

    friend istream& operator>>(istream& in, bignum& a){
        in >> a.num;
        reverse(a.num.begin(), a.num.end());
        return in;
    }
    friend ostream& operator<<(ostream& out, bignum a){
        reverse(a.num.begin(), a.num.end());
        return out << a.num;
    }
};
```

### 扫描线

```cpp
#define ls (pos << 1)
#define rs (ls | 1)
#define mid ((tree[pos].l + tree[pos].r) >> 1)
struct Rectangle{
    ll x_l, y_l, x_r, y_r;
};
ll area(vector<Rectangle>& rec){
    struct Line{
        ll x, y_up, y_down;
        int pd;
    };
    vector<Line> line(rec.size() * 2);
    vector<ll> y_set(rec.size() * 2);
    for(int i = 0; i < rec.size(); i++){
        y_set[i * 2] = rec[i].y_l;
        y_set[i * 2 + 1] = rec[i].y_r;
        line[i * 2] = { rec[i].x_l,rec[i].y_r,rec[i].y_l,1 };
        line[i * 2 + 1] = { rec[i].x_r,rec[i].y_r,rec[i].y_l,-1 };
    }
    sort(y_set.begin(), y_set.end());
    y_set.erase(unique(y_set.begin(), y_set.end()), y_set.end());
    sort(line.begin(), line.end(), [](Line a, Line b){return a.x < b.x; });
    struct Data{
        int l, r;
        ll len, cnt, raw_len;
    };
    vector<Data> tree(4 * y_set.size());
    function<void(int, int, int)> build = [&](int pos, int l, int r){
        tree[pos].l = l;
        tree[pos].r = r;
        if(l == r){
            tree[pos].raw_len = y_set[r + 1] - y_set[l];
            tree[pos].cnt = tree[pos].len = 0;
            return;
        }
        build(ls, l, mid);
        build(rs, mid + 1, r);
        tree[pos].raw_len = tree[ls].raw_len + tree[rs].raw_len;
    };
    function<void(int, int, int, int)> update = [&](int pos, int l, int r, int num){
        if(l <= tree[pos].l && tree[pos].r <= r){
            tree[pos].cnt += num;
            tree[pos].len = tree[pos].cnt ? tree[pos].raw_len : tree[pos].l == tree[pos].r ? 0 : tree[ls].len + tree[rs].len;
            return;
        }
        if(l <= mid)update(ls, l, r, num);
        if(r > mid)update(rs, l, r, num);
        tree[pos].len = tree[pos].cnt ? tree[pos].raw_len : tree[ls].len + tree[rs].len;
    };
    build(1, 0, y_set.size() - 2);
    auto find_pos = [&](ll num){
        return lower_bound(y_set.begin(), y_set.end(), num) - y_set.begin();
    };
    ll res = 0;
    for(int i = 0; i < line.size() - 1; i++){
        update(1, find_pos(line[i].y_down), find_pos(line[i].y_up) - 1, line[i].pd);
        res += (line[i + 1].x - line[i].x) * tree[1].len;
    }
    return res;
}
```

### 模运算

```cpp
class modint{
    ll num;
public:
    modint(ll num = 0) :num(num % mod){}
    modint pow(modint other){
        modint res(1), temp = *this;
        while(other.num){
            if(other.num & 1)res = res * temp;
            temp = temp * temp;
            other.num >>= 1;
        }
        return res;
    }
    modint inv(){ return this->pow(mod - 2); }
    modint operator+(modint other){ return modint(this->num + other.num); }
    modint operator-(){ return { -this->num }; }
    modint operator-(modint other){ return modint(-other + *this); }
    modint operator*(modint other){ return modint(this->num * other.num); }
    modint operator/(modint other){ return *this * other.inv(); }
    friend istream& operator>>(istream& is, modint& other){ is >> other.num; other.num %= mod; return is; }
    friend ostream& operator<<(ostream& os, modint other){ other.num = (other.num + mod) % mod; return os << other.num; }
};
```

### 分数

```cpp
struct frac{
    ll a, b;
    frac() :a(0), b(1){}
    frac(ll a, ll b = 1){
        if(a){
            ll temp = gcd(a, b);
            this->a = a / temp; this->b = b / temp;
        }
        else{
            this->a = 0; this->b = 1;
        }
    }
    frac operator+(const frac& other){
        return frac(a * other.b + other.a * b, b * other.b);
    }
    frac operator-()const{
        frac res = *this;
        res.a = -res.a;
        return res;
    }
    frac operator-(const frac& other){
        return -other + *this;
    }
    frac operator*(const frac& other){
        return frac(a * other.a, b * other.b);
    }
    frac operator/(const frac& other){
        assert(other.a);
        return *this * frac(other.b, other.a);
    }
    bool operator<(const frac& other){
        return (*this - other).a < 0;
    }
    bool operator<=(const frac& other){
        return (*this - other).a <= 0;
    }
    bool operator>(const frac& other){
        return (*this - other).a > 0;
    }
    bool operator>=(const frac& other){
        return (*this - other).a >= 0;
    }
    bool operator==(const frac& other){
        return a == other.a && b == other.b;
    }
    bool operator!=(const frac& other){
        return !(*this == other);
    }
};
```

### 表达式求值

```cpp
// 格式化表达式
string format(const string& s1){
    stringstream ss(s1);
    string s2;
    char ch;
    while((ch = ss.get()) != EOF){
        if(ch == ' ')continue;
        if(isdigit(ch))s2 += ch;
        else{
            if(s2.back() != ' ')s2 += ' ';
            s2 += ch; s2 += ' ';
        }
    }
    return s2;
}

// 中缀表达式转后缀表达式
string convert(const string& s1){
    
    unordered_map<char, int> rank{ {'+',2},{'-',2},{'*',1},{'/',1},{'^',0} };
    stringstream ss(s1);
    string s2, temp;
    stack<char> op;
    while(ss >> temp){
        if(isdigit(temp[0]))s2 += temp + ' ';
        else if(temp[0] == '(')op.push('(');
        else if(temp[0] == ')'){
            while(op.top() != '('){
                s2 += op.top(); s2 += ' '; op.pop();
            }
            op.pop();
        }
        else{
            while(!op.empty() && op.top() != '(' && (temp[0] != '^' && rank[op.top()] <= rank[temp[0]] || rank[op.top()] < rank[temp[0]])){
                s2 += op.top(); s2 += ' '; op.pop();
            }
            op.push(temp[0]);
        }
    }
    while(!op.empty()){
        s2 += op.top(); s2 += ' '; op.pop();
    }
    return s2;
}

// 计算后缀表达式
int calc(const string& s){
    stack<int> num;
    stringstream ss(s);
    string temp;
    while(ss >> temp){
        if(isdigit(temp[0]))num.push(stoi(temp));
        else{
            int b = num.top(); num.pop();
            int a = num.top(); num.pop();
            if(temp[0] == '+')a += b;
            else if(temp[0] == '-')a -= b;
            else if(temp[0] == '*')a *= b;
            else if(temp[0] == '/')a /= b;
            else if(temp[0] == '^')a = ksm(a, b);
            num.push(a);
        }
    }
    return num.top();
}
```

### 对拍

#### linux/Mac

```bash
g++ a.cpp -o program/a -O2 -std=c++17
g++ b.cpp -o program/b -O2 -std=c++17
g++ suiji.cpp -o program/suiji -O2 -std=c++17

cnt=0

while true; do
    let cnt++
    echo TEST:$cnt

    ./program/suiji > in
    ./program/a < in > out.a
    ./program/b < in > out.b

    diff out.a out.b
    if [ $? -ne 0 ];then break;fi
done
```

#### windows

```bash
@echo off

g++ a.cpp -o program/a -O2 -std=c++17
g++ b.cpp -o program/b -O2 -std=c++17
g++ suiji.cpp -o program/suiji -O2 -std=c++17

set cnt=0

:again
    set /a cnt=cnt+1
    echo TEST:%cnt%
    .\program\suiji > in
    .\program\a < in > out.a
    .\program\b < in > out.b

    fc output.a output.b
if not errorlevel 1 goto again
```

### 开栈

#### windowns

```bash
-Wl,--stack=268435456
```

#### linux

```bash
-Wl,-stack_size -Wl,0x10000000
```

### 日期

```cpp
int month[] = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
int pre[13];
vector<int> leap;
struct Date {
    int y, m, d;
    bool operator < (const Date& other) const {
        return array<int, 3>{y, m, d} < array<int, 3>{other.y, other.m, other.d};
    }
    Date(const string& s) {
        stringstream ss(s);
        char ch;
        ss >> y >> ch >> m >> ch >> d;
    }
    int dis()const {
        int yd = (y - 1) * 365 + (upper_bound(leap.begin(), leap.end(), y - 1) - leap.begin());
        int md = pre[m - 1] + (m > 2 && (y % 4 == 0 && y % 100 || y % 400 == 0));
        return yd + md + d;
    }
    int dis(const Date& other)const {
        return other.dis() - dis();
    }
};
for (int i = 1; i <= 12; i++)pre[i] = pre[i - 1] + month[2];
for (int i = 1; i <= 1000000; i++)
    if (i % 4 == 0 && i % 100 || i % 400 == 0)leap.push_back(i);
```
