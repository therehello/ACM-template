# ACM常用算法模板

<!-- TOC -->

- [数据结构](#%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84)
    - [并查集](#%E5%B9%B6%E6%9F%A5%E9%9B%86)
    - [树状数组](#%E6%A0%91%E7%8A%B6%E6%95%B0%E7%BB%84)
        - [一维](#%E4%B8%80%E7%BB%B4)
        - [二维](#%E4%BA%8C%E7%BB%B4)
        - [三维](#%E4%B8%89%E7%BB%B4)
    - [线段树](#%E7%BA%BF%E6%AE%B5%E6%A0%91)
    - [普通平衡树](#%E6%99%AE%E9%80%9A%E5%B9%B3%E8%A1%A1%E6%A0%91)
        - [树状数组实现](#%E6%A0%91%E7%8A%B6%E6%95%B0%E7%BB%84%E5%AE%9E%E7%8E%B0)
    - [可持久化线段树](#%E5%8F%AF%E6%8C%81%E4%B9%85%E5%8C%96%E7%BA%BF%E6%AE%B5%E6%A0%91)
    - [st 表](#st-%E8%A1%A8)
- [图论](#%E5%9B%BE%E8%AE%BA)
    - [最短路](#%E6%9C%80%E7%9F%AD%E8%B7%AF)
        - [dijkstra](#dijkstra)
    - [树上问题](#%E6%A0%91%E4%B8%8A%E9%97%AE%E9%A2%98)
        - [最近公公祖先](#%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%AC%E7%A5%96%E5%85%88)
        - [树链剖分](#%E6%A0%91%E9%93%BE%E5%89%96%E5%88%86)
    - [强连通分量](#%E5%BC%BA%E8%BF%9E%E9%80%9A%E5%88%86%E9%87%8F)
    - [拓扑排序](#%E6%8B%93%E6%89%91%E6%8E%92%E5%BA%8F)
- [字符串](#%E5%AD%97%E7%AC%A6%E4%B8%B2)
    - [kmp](#kmp)
    - [哈希](#%E5%93%88%E5%B8%8C)
    - [manacher](#manacher)
- [数学](#%E6%95%B0%E5%AD%A6)
    - [扩展欧几里得](#%E6%89%A9%E5%B1%95%E6%AC%A7%E5%87%A0%E9%87%8C%E5%BE%97)
    - [线性筛法](#%E7%BA%BF%E6%80%A7%E7%AD%9B%E6%B3%95)
    - [分解质因数](#%E5%88%86%E8%A7%A3%E8%B4%A8%E5%9B%A0%E6%95%B0)
    - [pollard rho](#pollard-rho)
    - [组合数](#%E7%BB%84%E5%90%88%E6%95%B0)
    - [数论分块](#%E6%95%B0%E8%AE%BA%E5%88%86%E5%9D%97)
    - [积性函数](#%E7%A7%AF%E6%80%A7%E5%87%BD%E6%95%B0)
        - [定义](#%E5%AE%9A%E4%B9%89)
        - [例子](#%E4%BE%8B%E5%AD%90)
    - [狄利克雷卷积](#%E7%8B%84%E5%88%A9%E5%85%8B%E9%9B%B7%E5%8D%B7%E7%A7%AF)
        - [性质](#%E6%80%A7%E8%B4%A8)
        - [例子](#%E4%BE%8B%E5%AD%90)
    - [欧拉函数](#%E6%AC%A7%E6%8B%89%E5%87%BD%E6%95%B0)
    - [莫比乌斯反演](#%E8%8E%AB%E6%AF%94%E4%B9%8C%E6%96%AF%E5%8F%8D%E6%BC%94)
        - [莫比乌斯函数性质](#%E8%8E%AB%E6%AF%94%E4%B9%8C%E6%96%AF%E5%87%BD%E6%95%B0%E6%80%A7%E8%B4%A8)
        - [莫比乌斯变换/反演](#%E8%8E%AB%E6%AF%94%E4%B9%8C%E6%96%AF%E5%8F%98%E6%8D%A2%E5%8F%8D%E6%BC%94)
    - [杜教筛](#%E6%9D%9C%E6%95%99%E7%AD%9B)
        - [示例](#%E7%A4%BA%E4%BE%8B)
    - [多项式](#%E5%A4%9A%E9%A1%B9%E5%BC%8F)
    - [盒子与球](#%E7%9B%92%E5%AD%90%E4%B8%8E%E7%90%83)
        - [球同，盒同，可空](#%E7%90%83%E5%90%8C%E7%9B%92%E5%90%8C%E5%8F%AF%E7%A9%BA)
        - [球不同，盒同，可空](#%E7%90%83%E4%B8%8D%E5%90%8C%E7%9B%92%E5%90%8C%E5%8F%AF%E7%A9%BA)
        - [球同，盒不同，可空](#%E7%90%83%E5%90%8C%E7%9B%92%E4%B8%8D%E5%90%8C%E5%8F%AF%E7%A9%BA)
        - [球同，盒不同，不可空](#%E7%90%83%E5%90%8C%E7%9B%92%E4%B8%8D%E5%90%8C%E4%B8%8D%E5%8F%AF%E7%A9%BA)
        - [球不同，盒不同，可空](#%E7%90%83%E4%B8%8D%E5%90%8C%E7%9B%92%E4%B8%8D%E5%90%8C%E5%8F%AF%E7%A9%BA)
        - [球不同，盒不同，不可空](#%E7%90%83%E4%B8%8D%E5%90%8C%E7%9B%92%E4%B8%8D%E5%90%8C%E4%B8%8D%E5%8F%AF%E7%A9%BA)
    - [线性基](#%E7%BA%BF%E6%80%A7%E5%9F%BA)
    - [矩阵快速幂](#%E7%9F%A9%E9%98%B5%E5%BF%AB%E9%80%9F%E5%B9%82)
- [计算几何](#%E8%AE%A1%E7%AE%97%E5%87%A0%E4%BD%95)
    - [整数](#%E6%95%B4%E6%95%B0)
    - [浮点数](#%E6%B5%AE%E7%82%B9%E6%95%B0)
    - [扫描线](#%E6%89%AB%E6%8F%8F%E7%BA%BF)
- [杂项](#%E6%9D%82%E9%A1%B9)
    - [高精度](#%E9%AB%98%E7%B2%BE%E5%BA%A6)
    - [模运算](#%E6%A8%A1%E8%BF%90%E7%AE%97)
    - [分数](#%E5%88%86%E6%95%B0)
    - [表达式求值](#%E8%A1%A8%E8%BE%BE%E5%BC%8F%E6%B1%82%E5%80%BC)
    - [日期](#%E6%97%A5%E6%9C%9F)
    - [对拍](#%E5%AF%B9%E6%8B%8D)
    - [编译常用选项](#%E7%BC%96%E8%AF%91%E5%B8%B8%E7%94%A8%E9%80%89%E9%A1%B9)
    - [开栈](#%E5%BC%80%E6%A0%88)
    - [clang-format](#clang-format)

<!-- /TOC -->

## 数据结构

### 并查集

```cpp
struct dsu {
    int n;
    vector<int> fa, sz;
    dsu(int _n) : n(_n), fa(n + 1), sz(n + 1, 1) {
        iota(fa.begin(), fa.end(), 0);
    }
    int find(int x) { return x == fa[x] ? x : fa[x] = find(fa[x]); }
    int merge(int x, int y) {
        int fax = find(x), fay = find(y);
        if (fax == fay) return 0;  // 一个集合
        sz[fay] += sz[fax];
        return fa[fax] = fay;  // 合并到哪个集合了
    }
    int size(int x) { return sz[find(x)]; }
};
```

### 树状数组

#### 一维

```cpp
template <class T>
struct fenwick {
    int n;
    vector<T> t;
    fenwick(int _n) : n(_n), t(n + 1) {}
    T query(int l, int r) {
        auto query = [&](int pos) {
            T res = 0;
            while (pos) {
                res += t[pos];
                pos -= lowbit(pos);
            }
            return res;
        };
        return query(r) - query(l - 1);
    }
    void add(int pos, T num) {
        while (pos <= n) {
            t[pos] += num;
            pos += lowbit(pos);
        }
    }
};
```

#### 二维

```cpp
template <class T>
struct Fenwick_tree_2 {
    Fenwick_tree_2(int n, int m) : n(n), m(m), tree(n + 1, vector<T>(m + 1)) {}
    T query(int l1, int r1, int l2, int r2) {
        auto query = [&](int l, int r) {
            T res = 0;
            for (int i = l; i; i -= lowbit(i))
                for (int j = r; j; j -= lowbit(j)) res += tree[i][j];
            return res;
        };
        return query(l2, r2) - query(l2, r1 - 1) - query(l1 - 1, r2) +
               query(l1 - 1, r1 - 1);
    }
    void update(int x, int y, T num) {
        for (int i = x; i <= n; i += lowbit(i))
            for (int j = y; j <= m; j += lowbit(j)) tree[i][j] += num;
    }
private:
    int n, m;
    vector<vector<T>> tree;
};
```

#### 三维

```cpp
template <class T>
struct Fenwick_tree_3 {
    Fenwick_tree_3(int n, int m, int k)
        : n(n),
          m(m),
          k(k),
          tree(n + 1, vector<vector<T>>(m + 1, vector<T>(k + 1))) {}
    T query(int a, int b, int c, int d, int e, int f) {
        auto query = [&](int x, int y, int z) {
            T res = 0;
            for (int i = x; i; i -= lowbit(i))
                for (int j = y; j; j -= lowbit(j))
                    for (int p = z; p; p -= lowbit(p)) res += tree[i][j][p];
            return res;
        };
        T res = query(d, e, f);
        res -= query(a - 1, e, f) + query(d, b - 1, f) + query(d, e, c - 1);
        res += query(a - 1, b - 1, f) + query(a - 1, e, c - 1) +
               query(d, b - 1, c - 1);
        res -= query(a - 1, b - 1, c - 1);
        return res;
    }
    void update(int x, int y, int z, T num) {
        for (int i = x; i <= n; i += lowbit(i))
            for (int j = y; j <= m; j += lowbit(j))
                for (int p = z; p <= k; p += lowbit(p)) tree[i][j][p] += num;
    }
private:
    int n, m, k;
    vector<vector<vector<T>>> tree;
};
```

### 线段树

```cpp
template <class Data, class Num>
struct Segment_Tree {
    inline void update(int l, int r, Num x) { update(1, l, r, x); }
    inline Data query(int l, int r) { return query(1, l, r); }
    Segment_Tree(vector<Data>& a) {
        n = a.size();
        tree.assign(n * 4 + 1, {});
        build(a, 1, 1, n);
    }
private:
    int n;
    struct Tree {
        int l, r;
        Data data;
    };
    vector<Tree> tree;
    inline void pushup(int pos) {
        tree[pos].data = tree[pos << 1].data + tree[pos << 1 | 1].data;
    }
    inline void pushdown(int pos) {
        tree[pos << 1].data = tree[pos << 1].data + tree[pos].data.lazytag;
        tree[pos << 1 | 1].data =
            tree[pos << 1 | 1].data + tree[pos].data.lazytag;
        tree[pos].data.lazytag = Num::zero();
    }
    void build(vector<Data>& a, int pos, int l, int r) {
        tree[pos].l = l;
        tree[pos].r = r;
        if (l == r) {
            tree[pos].data = a[l - 1];
            return;
        }
        int mid = (tree[pos].l + tree[pos].r) >> 1;
        build(a, pos << 1, l, mid);
        build(a, pos << 1 | 1, mid + 1, r);
        pushup(pos);
    }
    void update(int pos, int& l, int& r, Num& x) {
        if (l > tree[pos].r || r < tree[pos].l) return;
        if (l <= tree[pos].l && tree[pos].r <= r) {
            tree[pos].data = tree[pos].data + x;
            return;
        }
        pushdown(pos);
        update(pos << 1, l, r, x);
        update(pos << 1 | 1, l, r, x);
        pushup(pos);
    }
    Data query(int pos, int& l, int& r) {
        if (l > tree[pos].r || r < tree[pos].l) return Data::zero();
        if (l <= tree[pos].l && tree[pos].r <= r) return tree[pos].data;
        pushdown(pos);
        return query(pos << 1, l, r) + query(pos << 1 | 1, l, r);
    }
};
struct Num {
    ll add;
    inline static Num zero() { return {0}; }
    inline Num operator+(Num b) { return {add + b.add}; }
};
struct Data {
    ll sum, len;
    Num lazytag;
    inline static Data zero() { return {0, 0, Num::zero()}; }
    inline Data operator+(Num b) {
        return {sum + len * b.add, len, lazytag + b};
    }
    inline Data operator+(Data b) {
        return {sum + b.sum, len + b.len, Num::zero()};
    }
};
```
### 普通平衡树

#### 树状数组实现
    
需要预先处理出来所有可能的数。

```cpp
int lowbit(int x) { return x & -x; }

template <typename T>
struct treap {
    int n, size;
    vector<int> t;
    vector<T> t2, S;
    treap(const vector<T>& a) : S(a) {
        sort(S.begin(), S.end());
        S.erase(unique(S.begin(), S.end()), S.end());
        n = S.size();
        size = 0;
        t = vector<int>(n + 1);
        t2 = vector<T>(n + 1);
    }
    int pos(T x) { return lower_bound(S.begin(), S.end(), x) - S.begin() + 1; }
    int sum(int pos) {
        int res = 0;
        while (pos) {
            res += t[pos];
            pos -= lowbit(pos);
        }
        return res;
    }

    // 插入cnt个x
    void insert(T x, int cnt) {
        size += cnt;
        int i = pos(x);
        assert(i <= n && S[i - 1] == x);
        for (; i <= n; i += lowbit(i)) {
            t[i] += cnt;
            t2[i] += cnt * x;
        }
    }

    // 删除cnt个x
    void erase(T x, int cnt) {
        assert(cnt <= count(x));
        insert(x, -cnt);
    }

    // x的排名
    int rank(T x) {
        assert(count(x));
        return sum(pos(x) - 1) + 1;
    }

    // 统计出现次数
    int count(T x) { return sum(pos(x)) - sum(pos(x) - 1); }

    // 第k小
    T kth(int k) {
        assert(0 < k && k <= size);
        int cnt = 0, x = 0;
        for (int i = __lg(n); i >= 0; i--) {
            x += 1 << i;
            if (x >= n || cnt + t[x] >= k) x -= 1 << i;
            else cnt += t[x];
        }
        return S[x];
    }

    // 前k小的数之和
    T pre_sum(int k) {
        assert(0 < k && k <= size);
        int cnt = 0, x = 0;
        T res = 0;
        for (int i = __lg(n); i >= 0; i--) {
            x += 1 << i;
            if (x >= n || cnt + t[x] >= k) x -= 1 << i;
            else {
                cnt += t[x];
                res += t2[x];
            }
        }
        return res + (k - cnt) * S[x];
    }

    // 小于x，最大的数
    T prev(T x) { return kth(sum(pos(x) - 1)); }

    // 大于x，最小的数
    T next(T x) { return kth(sum(pos(x)) + 1); }
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
    Persistent_seg(int n, vector<int>& a) : n(n) { root[0] = build(1, n, a); }
    int build(int l, int r, vector<int>& a) {
        if (l == r) {
            tree.push_back({0, 0, a[l]});
            return tree.size() - 1;
        }
        int mid = l + r >> 1;
        int ls = build(l, mid, a), rs = build(mid + 1, r, a);
        tree.push_back({ls, rs, tree[ls].val + tree[rs].val});
        return tree.size() - 1;
    }
    int update(int rt, const int& idx, const int& val, int l, int r) {
        if (l == r) {
            tree.push_back({0, 0, tree[rt].val + val});
            return tree.size() - 1;
        }
        int mid = l + r >> 1, ls = tree[rt].ls, rs = tree[rt].rs;
        if (idx <= mid) ls = update(ls, idx, val, l, mid);
        else rs = update(rs, idx, val, mid + 1, r);
        tree.push_back({ls, rs, tree[ls].val + tree[rs].val});
        return tree.size() - 1;
    }
    int query(int rt1, int rt2, int k, int l, int r) {
        if (l == r) return l;
        int mid = l + r >> 1;
        int lcnt = tree[tree[rt2].ls].val - tree[tree[rt1].ls].val;
        if (k <= lcnt) return query(tree[rt1].ls, tree[rt2].ls, k, l, mid);
        else return query(tree[rt1].rs, tree[rt2].rs, k - lcnt, mid + 1, r);
    }
};
```

### st 表

```cpp
auto lg = []() {
    array<int, 10000001> lg;
    lg[1] = 0;
    for (int i = 2; i <= 10000000; i++) lg[i] = lg[i >> 1] + 1;
    return lg;
}();
template <typename T>
struct st {
    int n;
    vector<vector<T>> a;
    st(vector<T>& _a) : n(_a.size()) {
        a.assign(lg[n] + 1, vector<int>(n));
        for (int i = 0; i < n; i++) a[0][i] = _a[i];
        for (int j = 1; j <= lg[n]; j++)
            for (int i = 0; i + (1 << j) - 1 < n; i++)
                a[j][i] = max(a[j - 1][i], a[j - 1][i + (1 << (j - 1))]);
    }
    T query(int l, int r) {
        int k = lg[r - l + 1];
        return max(a[k][l], a[k][r - (1 << k) + 1]);
    }
};
```

## 图论

存图

```cpp
struct Graph {
    int n;
    struct Edge {
        int to, w;
    };
    vector<vector<Edge>> graph;
    Graph(int _n) {
        n = _n;
        graph.assign(n + 1, vector<Edge>());
    };
    void add(int u, int v, int w) { graph[u].push_back({v, w}); }
};
```

### 最短路

#### dijkstra

```cpp
void dij(Graph& graph, vector<int>& dis, int t) {
    vector<int> visit(graph.n + 1, 0);
    priority_queue<pair<int, int>> que;
    dis[t] = 0;
    que.emplace(0, t);
    while (!que.empty()) {
        int u = que.top().second;
        que.pop();
        if (visit[u]) continue;
        visit[u] = 1;
        for (auto& [to, w] : graph.graph[u]) {
            if (dis[to] > dis[u] + w) {
                dis[to] = dis[u] + w;
                que.emplace(-dis[to], to);
            }
        }
    }
}
```

### 树上问题

#### 最近公公祖先

倍增法

```cpp
vector<int> dep;
vector<array<int, 21>> fa;
dep.assign(n + 1, 0);
fa.assign(n + 1, array<int, 21>{});
void binary_jump(int root) {
    function<void(int)> dfs = [&](int t) {
        dep[t] = dep[fa[t][0]] + 1;
        for (auto& [to] : graph[t]) {
            if (to == fa[t][0]) continue;
            fa[to][0] = t;
            dfs(to);
        }
    };
    dfs(root);
    for (int j = 1; j <= 20; j++)
        for (int i = 1; i <= n; i++) fa[i][j] = fa[fa[i][j - 1]][j - 1];
}
int lca(int x, int y) {
    if (dep[x] < dep[y]) swap(x, y);
    for (int i = 20; i >= 0; i--)
        if (dep[fa[x][i]] >= dep[y]) x = fa[x][i];
    if (x == y) return x;
    for (int i = 20; i >= 0; i--) {
        if (fa[x][i] != fa[y][i]) {
            x = fa[x][i];
            y = fa[y][i];
        }
    }
    return fa[x][0];
}
```

树剖

```cpp
int lca(int x, int y) {
    while (top[x] != top[y]) {
        if (dep[top[x]] < dep[top[y]]) swap(x, y);
        x = fa[top[x]];
    }
    if (dep[x] < dep[y]) swap(x, y);
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
void hld(int root) {
    function<void(int)> dfs1 = [&](int t) {
        dep[t] = dep[fa[t]] + 1;
        siz[t] = 1;
        for (auto& [to, w] : graph[t]) {
            if (to == fa[t]) continue;
            fa[to] = t;
            dfs1(to);
            if (siz[son[t]] < siz[to]) son[t] = to;
            siz[t] += siz[to];
        }
    };
    dfs1(root);
    int dfn_tail = 0;
    for (int i = 1; i <= n; i++) top[i] = i;
    function<void(int)> dfs2 = [&](int t) {
        dfn[t] = ++dfn_tail;
        rnk[dfn_tail] = t;
        if (!son[t]) return;
        top[son[t]] = top[t];
        dfs2(son[t]);
        for (auto& [to, w] : graph[t]) {
            if (to == fa[t] || to == son[t]) continue;
            dfs2(to);
        }
    };
    dfs2(root);
}
```

### 强连通分量

```cpp
void tarjan(Graph& g1, Graph& g2) {
    int dfn_tail = 0, cnt = 0;
    vector<int> dfn(g1.n + 1, 0), low(g1.n + 1, 0), exist(g1.n + 1, 0),
        belong(g1.n + 1, 0);
    stack<int> sta;
    function<void(int)> dfs = [&](int t) {
        dfn[t] = low[t] = ++dfn_tail;
        sta.push(t);
        exist[t] = 1;
        for (auto& [to] : g1.graph[t])
            if (!dfn[to]) {
                dfs(to);
                low[t] = min(low[t], low[to]);
            } else if (exist[to]) low[t] = min(low[t], dfn[to]);
        if (dfn[t] == low[t]) {
            cnt++;
            while (int temp = sta.top()) {
                belong[temp] = cnt;
                exist[temp] = 0;
                sta.pop();
                if (temp == t) break;
            }
        }
    };
    for (int i = 1; i <= g1.n; i++)
        if (!dfn[i]) dfs(i);
    g2 = Graph(cnt);
    for (int i = 1; i <= g1.n; i++) g2.w[belong[i]] += g1.w[i];
    for (int i = 1; i <= g1.n; i++)
        for (auto& [to] : g1.graph[i])
            if (belong[i] != belong[to]) g2.add(belong[i], belong[to]);
}
```

### 拓扑排序

```cpp
void toposort(Graph& g, vector<int>& dis) {
    vector<int> in(g.n + 1, 0);
    for (int i = 1; i <= g.n; i++)
        for (auto& [to] : g.graph[i]) in[to]++;
    queue<int> que;
    for (int i = 1; i <= g.n; i++)
        if (!in[i]) {
            que.push(i);
            dis[i] = g.w[i];  // dp
        }
    while (!que.empty()) {
        int u = que.front();
        que.pop();
        for (auto& [to] : g.graph[u]) {
            in[to]--;
            dis[to] = max(dis[to], dis[u] + g.w[to]);  // dp
            if (!in[to]) que.push(to);
        }
    }
}
```

## 字符串

### kmp

```cpp
auto kmp(string& s) {
    vector next(s.size(), -1);
    for (int i = 1, j = -1; i < s.size(); i++) {
        while (j >= 0 && s[i] != s[j + 1]) j = next[j];
        if (s[i] == s[j + 1]) j++;
        next[i] = j;
    }
    // next 意为长度
    for (auto& i : next) i++;
    return next;
}
```

### 哈希

```cpp
constexpr int N = 1e6;
int pow_base[N + 1][2];
constexpr ll mod[2] = {(int)2e9 + 11, (int)2e9 + 33},
             base[2] = {(int)2e5 + 11, (int)2e5 + 33};

struct Hash {
    int size;
    vector<array<int, 2>> a;
    Hash() {}
    Hash(const string& s) {
        size = s.size();
        a.resize(size);
        a[0][0] = a[0][1] = s[0];
        for (int i = 1; i < size; i++) {
            a[i][0] = (a[i - 1][0] * base[0] + s[i]) % mod[0];
            a[i][1] = (a[i - 1][1] * base[1] + s[i]) % mod[1];
        }
    }
    array<int, 2> get(int l, int r) const {
        if (l == 0) return a[r];
        auto getone = [&](bool f) {
            int x =
                (a[r][f] - 1ll * a[l - 1][f] * pow_base[r - l + 1][f]) % mod[f];
            if (x < 0) x += mod[f];
            return x;
        };
        return {getone(0), getone(1)};
    }
};

auto _ = []() {
    pow_base[0][0] = pow_base[0][1] = 1;
    for (int i = 1; i <= N; i++) {
        pow_base[i][0] = pow_base[i - 1][0] * base[0] % mod[0];
        pow_base[i][1] = pow_base[i - 1][1] * base[1] % mod[1];
    }
    return true;
}();
```

### manacher

```cpp
auto manacher(const string& _s) {
    string s(_s.size() * 2 + 1, '$');
    for (int i = 0; i < _s.size(); i++) s[2 * i + 1] = _s[i];
    vector r(s.size(), 0);
    for (int i = 0, maxr = 0, mid = 0; i < s.size(); i++) {
        if (i < maxr) r[i] = min(r[mid * 2 - i], maxr - i);
        while (i - r[i] - 1 >= 0 && i + r[i] + 1 < s.size() &&
               s[i - r[i] - 1] == s[i + r[i] + 1])
            ++r[i];
        if (i + r[i] > maxr) maxr = i + r[i], mid = i;
    }
    return r;
}
```

## 数学

### 扩展欧几里得

需保证 $a,b>=0$

$x=x+k*dx,y=y-k*dy$

若要求 $x\ge p$，$k\ge\lceil \frac{p-x}{dx}\rceil$

若要求 $x\le q$，$k\le\lfloor \frac{q-x}{dx}\rfloor$

若要求 $y\ge p$，$k\le\lfloor \frac{y-p}{dy}\rfloor$

若要求 $y\le q$，$k\ge\lceil \frac{y-q}{dy}\rceil$

```cpp
int __exgcd(int a, int b, int& x, int& y) {
    if (!b) {
        x = 1;
        y = 0;
        return a;
    }
    int g = __exgcd(b, a % b, y, x);
    y -= a / b * x;
    return g;
}

array<int, 2> exgcd(int a, int b, int c) {
    int x, y;
    int g = __exgcd(a, b, x, y);
    if (c % g) return {INT_MAX, INT_MAX};
    int dx = b / g;
    int dy = a / g;
    x = c / g % dx * x % dx;
    if (x < 0) x += dx;
    y = (c - a * x) / b;
    return {x, y};
}
```

### 线性筛法

```cpp
constexpr int N = 10000000;
array<int, N + 1> min_prime;
vector<int> primes;
bool ok = []() {
    for (int i = 2; i <= N; i++) {
        if (min_prime[i] == 0) {
            min_prime[i] = i;
            primes.push_back(i);
        }
        for (auto& j : primes) {
            if (j > min_prime[i] || j > N / i) break;
            min_prime[j * i] = j;
        }
    }
    return 1;
}();
```

### 分解质因数

```cpp
auto getprimes(int n) {
    vector<array<int, 2>> res;
    for (auto& i : primes) {
        if (i > n / i) break;
        if (n % i == 0) {
            res.push_back({i, 0});
            while (n % i == 0) {
                n /= i;
                res.back()[1]++;
            }
        }
    }
    if (n > 1) res.push_back({n, 1});
    return res;
}
```

### pollard rho

```cpp
using LL = __int128_t;

random_device rd;
mt19937 seed(rd());

ll power(ll a, ll b, ll mod) {
    ll res = 1;
    while (b) {
        if (b & 1) res = (LL)res * a % mod;
        a = (LL)a * a % mod;
        b >>= 1;
    }
    return res;
}

bool isprime(ll n) {
    static array primes{2, 3, 5, 7, 11, 13, 17, 19, 23};
    static unordered_map<ll, bool> S;
    if (n < 2) return 0;
    if (S.count(n)) return S[n];
    ll d = n - 1, r = 0;
    while (!(d & 1)) {
        r++;
        d >>= 1;
    }
    for (auto& a : primes) {
        if (a == n) return S[n] = 1;
        ll x = power(a, d, n);
        if (x == 1 || x == n - 1) continue;
        for (int i = 0; i < r - 1; i++) {
            x = (LL)x * x % n;
            if (x == n - 1) break;
        }
        if (x != n - 1) return S[n] = 0;
    }
    return S[n] = 1;
}

ll pollard_rho(ll n) {
    ll s = 0, t = 0;
    ll c = seed() % (n - 1) + 1;
    ll val = 1;
    for (int goal = 1;; goal *= 2, s = t, val = 1) {
        for (int step = 1; step <= goal; step++) {
            t = ((LL)t * t + c) % n;
            val = (LL)val * abs(t - s) % n;
            if (step % 127 == 0) {
                ll g = gcd(val, n);
                if (g > 1) return g;
            }
        }
        ll g = gcd(val, n);
        if (g > 1) return g;
    }
}
auto getprimes(ll n) {
    unordered_set<ll> S;
    auto get = [&](auto self, ll n) {
        if (n < 2) return;
        if (isprime(n)) {
            S.insert(n);
            return;
        }
        ll mx = pollard_rho(n);
        self(self, n / mx);
        self(self, mx);
    };
    get(get, n);
    return S;
}
```

### 组合数

```cpp
constexpr int N = 1e6;
array<modint, N + 1> fac, ifac;

modint C(int n, int m) {
    if (n < m) return 0;
    if (n <= mod) return fac[n] * ifac[m] * ifac[n - m];
    // n >= mod 时需要这个
    return C(n % mod, m % mod) * C(n / mod, m / mod);
}

auto _ = []() {
    fac[0] = 1;
    for (int i = 1; i <= N; i++) fac[i] = fac[i - 1] * i;
    ifac[N] = fac[N].inv();
    for (int i = N - 1; i >= 0; i--) ifac[i] = ifac[i + 1] * (i + 1);
    return true;
}();
```

### 数论分块

求解形如 $\sum_{i=1}^{n}f(i)g(\lfloor\frac{n}{i}\rfloor)$ 的合式

$s(n) = \sum_{i=1}^{n}f(i)$

```cpp
modint sqrt_decomposition(int n) {
    auto s = [&](int x) { return x; };
    auto g = [&](int x) { return x; };
    modint res = 0;
    while (l <= R) {
        int r = n / (n / l);
        res = res + (s(r) - s(l - 1)) * g(n / l);
        l = r + 1;
    }
    return res;
}
```

### 积性函数

#### 定义

函数 $f(n)$ 满足 $f(1)=1$ 且 $\forall x,y\in\mathbf{N}^*,\gcd(x,y)=1$ 都有 $f(xy)=f(x)f(y)$，则 $f(n)$ 为积性函数。

函数 $f(n)$ 满足 $f(1)=1$ 且 $\forall x,y\in\mathbf{N}^*$ 都有 $f(xy)=f(x)f(y)$，则 $f(n)$ 为完全积性函数。

#### 例子

- 单位函数：$\varepsilon(n)=[n=1]$。（完全积性）
- 恒等函数：$\operatorname{id}_k(n)=n^k$。（完全积性）
- 常数函数：$1(n)=1$。（完全积性）
- 除数函数：$\sigma_{k}(n)=\sum_{d\mid n}d^{k}$。$\sigma_{0}(n)$ 通常简记作 $d(n)$ 或 $\tau(n)$，$\sigma_{1}(n)$ 通常简记作 $\sigma(n)$。
- 欧拉函数：$\varphi(n)=\sum_{i=1}^n[\gcd(i,n)=1]$。
- 莫比乌斯函数：$\mu(n)=\begin{cases}1&n=1\\0&\exists d>1,d^{2}\mid n\\(-1)^{\omega(n)}&\text{otherwise}\end{cases}$，其中 $\omega(n)$ 表示 $n$ 的本质不同质因子个数，它是一个加性函数。

### 狄利克雷卷积

对于两个数论函数 $f(x)$ 和 $g(x)$，则它们的狄利克雷卷积得到的结果 $h(x)$ 定义为：

$h(x)=\sum_{d\mid x}{f(d)g\left(\dfrac xd \right)}=\sum_{ab=x}{f(a)g(b)}$

可以简记为：$h=f*g$。

#### 性质

**交换律：**$f*g=g*f$。

**结合律：**$(f*g)*h=f*(g*h)$。

**分配律：**$(f+g)*h=f*h+g*h$。

**等式的性质：**$f=g$ 的充要条件是 $f*h=g*h$，其中数论函数 $h(x)$ 要满足 $h(1)\ne 0$。

#### 例子

- $\varepsilon=\mu \ast 1\iff\varepsilon(n)=\sum_{d\mid n}\mu(d)$
- $id = \varphi * 1 \iff id(n)=\sum_{d\mid n} \varphi(d)$
- $d=1 \ast 1\iff d(n)=\sum_{d\mid n}1$
- $\sigma=\operatorname{id} \ast 1\iff\sigma(n)=\sum_{d\mid n}d$
- $\varphi=\mu \ast \operatorname{id}\iff\varphi(n)=\sum_{d\mid n}d\cdot\mu(\frac{n}{d})$

### 欧拉函数

```cpp
constexpr int N = 1e6;
array<int, N + 1> phi;
auto _ = []() {
    iota(phi.begin() + 1, phi.end(), 1);
    for (int i = 2; i <= N; i++) {
        if (phi[i] == i)
            for (int j = i; j <= N; j += i) phi[j] = phi[j] / i * (i - 1);
    }
    return true;
}();
```

### 莫比乌斯反演

#### 莫比乌斯函数性质

- $\sum_{d\mid n}\mu(d)=\begin{cases}1&n=1\\0&n\neq 1\\\end{cases}$，即 $\sum_{d\mid n}\mu(d)=\varepsilon(n)$，$\mu * 1 =\varepsilon$
- $\displaystyle [\gcd(i,j)=1]=\sum_{d\mid\gcd(i,j)}\mu(d)$

```cpp
constexpr int N = 1e6;
array<int, N + 1> miu;
array<bool, N + 1> ispr;

auto _ = []() {
    miu.fill(1);
    ispr.fill(1);
    for (int i = 2; i <= N; i++) {
        if (!ispr[i]) continue;
        miu[i] = -1;
        for (int j = 2 * i; j <= N; j += i) {
            ispr[j] = 0;
            if ((j / i) % i == 0) miu[j] = 0;
            else miu[j] *= -1;
        }
    }
    return true;
}();
```

#### 莫比乌斯变换/反演

$f(n)=\sum_{d\mid n}g(d)$，那么有 $g(n)=\sum_{d\mid n}\mu(d)f(\frac{n}{d})=\sum_{n|d}\mu(\frac{d}{n})f(d)$
。

用狄利克雷卷积表示则为 $f=g\ast1$，有 $g=f\ast\mu$。

$f \rightarrow g$ 称为莫比乌斯反演，$g \rightarrow f$ 称为莫比乌斯反演。

### 杜教筛

杜教筛被用于处理一类数论函数的前缀和问题。对于数论函数 $f$，杜教筛可以在低于线性时间的复杂度内计算 $S(n)=\sum_{i=1}^{n}f(i)$。

$$
\begin{aligned}
    S(n) & = \frac{\sum_{i=1}^n (f * g)(i) - \sum_{i=2}^n g(i)S\left(\left\lfloor\frac{n}{i}\right\rfloor\right)}{g(1)}
\end{aligned}
$$

可以构造恰当的数论函数 $g$ 使得：

- 可以快速计算 $\sum_{i=1}^n(f * g)(i)$。
- 可以快速计算 $g$ 的单点值，用数论分块求解 $\sum_{i=2}^ng(i)S\left(\left\lfloor\frac{n}{i}\right\rfloor\right)$。

#### 示例

```cpp
ll sum_phi(ll n) {
    if (n <= N) return sp[n];
    if (sp2.count(n)) return sp2[n];
    ll res = 0, l = 2;
    while (l <= n) {
        ll r = n / (n / l);
        res = res + (r - l + 1) * sum_phi(n / l);
        l = r + 1;
    }
    return sp2[n] = (ll)n * (n + 1) / 2 - res;
}

ll sum_miu(ll n) {
    if (n <= N) return sm[n];
    if (sm2.count(n)) return sm2[n];
    ll res = 0, l = 2;
    while (l <= n) {
        ll r = n / (n / l);
        res = res + (r - l + 1) * sum_miu(n / l);
        l = r + 1;
    }
    return sm2[n] = 1 - res;
}
```

### 多项式

```cpp
#define countr_zero(n) __builtin_ctz(n)
constexpr int N = 1e6;
array<int, N + 1> inv;

int power(int a, int b) {
    int res = 1;
    while (b) {
        if (b & 1) res = 1ll * res * a % mod;
        a = 1ll * a * a % mod;
        b >>= 1;
    }
    return res;
}

namespace NFTS {
int g = 3;
vector<int> rev, roots{0, 1};
void dft(vector<int> &a) {
    int n = a.size();
    if (rev.size() != n) {
        int k = countr_zero(n) - 1;
        rev.resize(n);
        for (int i = 0; i < n; ++i) rev[i] = rev[i >> 1] >> 1 | (i & 1) << k;
    }
    if (roots.size() < n) {
        int k = countr_zero(roots.size());
        roots.resize(n);
        while ((1 << k) < n) {
            int e = power(g, (mod - 1) >> (k + 1));
            for (int i = 1 << (k - 1); i < (1 << k); ++i) {
                roots[2 * i] = roots[i];
                roots[2 * i + 1] = 1ll * roots[i] * e % mod;
            }
            ++k;
        }
    }
    for (int i = 0; i < n; ++i)
        if (rev[i] < i) swap(a[i], a[rev[i]]);
    for (int k = 1; k < n; k *= 2) {
        for (int i = 0; i < n; i += 2 * k) {
            for (int j = 0; j < k; ++j) {
                int u = a[i + j];
                int v = 1ll * a[i + j + k] * roots[k + j] % mod;
                int x = u + v, y = u - v;
                if (x >= mod) x -= mod;
                if (y < 0) y += mod;
                a[i + j] = x;
                a[i + j + k] = y;
            }
        }
    }
}
void idft(vector<int> &a) {
    int n = a.size();
    reverse(a.begin() + 1, a.end());
    dft(a);
    int inv_n = power(n, mod - 2);
    for (int i = 0; i < n; ++i) a[i] = 1ll * a[i] * inv_n % mod;
}
}  // namespace NFTS

struct poly {
    poly &format() {
        while (!a.empty() && a.back() == 0) a.pop_back();
        return *this;
    }
    poly &reverse() {
        ::reverse(a.begin(), a.end());
        return *this;
    }
    vector<int> a;
    poly() {}
    poly(int x) {
        if (x) a = {x};
    }
    poly(const vector<int> &_a) : a(_a) {}
    int size() const { return a.size(); }
    int &operator[](int id) { return a[id]; }
    int at(int id) const {
        if (id < 0 || id >= (int)a.size()) return 0;
        return a[id];
    }
    poly operator-() const {
        auto A = *this;
        for (auto &x : A.a) x = (x == 0 ? 0 : mod - x);
        return A;
    }
    poly mulXn(int n) const {
        auto b = a;
        b.insert(b.begin(), n, 0);
        return poly(b);
    }
    poly modXn(int n) const {
        if (n > size()) return *this;
        return poly({a.begin(), a.begin() + n});
    }
    poly divXn(int n) const {
        if (size() <= n) return poly();
        return poly({a.begin() + n, a.end()});
    }
    poly &operator+=(const poly &rhs) {
        if (size() < rhs.size()) a.resize(rhs.size());
        for (int i = 0; i < rhs.size(); ++i)
            if ((a[i] += rhs.a[i]) >= mod) a[i] -= mod;
        return *this;
    }
    poly &operator-=(const poly &rhs) {
        if (size() < rhs.size()) a.resize(rhs.size());
        for (int i = 0; i < rhs.size(); ++i)
            if ((a[i] -= rhs.a[i]) < 0) a[i] += mod;
        return *this;
    }
    poly &operator*=(poly rhs) {
        int n = size(), m = rhs.size(), tot = max(1, n + m - 1);
        int sz = 1 << __lg(tot * 2 - 1);
        a.resize(sz);
        rhs.a.resize(sz);
        NFTS::dft(a);
        NFTS::dft(rhs.a);
        for (int i = 0; i < sz; ++i) a[i] = 1ll * a[i] * rhs.a[i] % mod;
        NFTS::idft(a);
        return *this;
    }
    poly &operator/=(poly rhs) {
        int n = size(), m = rhs.size();
        if (n < m) return (*this) = poly();
        reverse();
        rhs.reverse();
        (*this) *= rhs.inv(n - m + 1);
        a.resize(n - m + 1);
        reverse();
        return *this;
    }
    poly &operator%=(poly rhs) { return (*this) -= (*this) / rhs * rhs; }
    poly operator+(const poly &rhs) const { return poly(*this) += rhs; }
    poly operator-(const poly &rhs) const { return poly(*this) -= rhs; }
    poly operator*(poly rhs) const { return poly(*this) *= rhs; }
    poly operator/(poly rhs) const { return poly(*this) /= rhs; }
    poly operator%(poly rhs) const { return poly(*this) %= rhs; }
    poly powModPoly(int n, poly p) {
        poly r(1), x(*this);
        while (n) {
            if (n & 1) (r *= x) %= p;
            (x *= x) %= p;
            n >>= 1;
        }
        return r;
    }
    int inner(const poly &rhs) {
        int r = 0, n = min(size(), rhs.size());
        for (int i = 0; i < n; ++i) r = (r + 1ll * a[i] * rhs.a[i]) % mod;
        return r;
    }
    poly derivation() const {
        if (a.empty()) return poly();
        int n = size();
        vector<int> r(n - 1);
        for (int i = 1; i < n; ++i) r[i - 1] = 1ll * a[i] * i % mod;
        return poly(r);
    }
    poly integral() const {
        if (a.empty()) return poly();
        int n = size();
        vector<int> r(n + 1);
        for (int i = 0; i < n; ++i) r[i + 1] = 1ll * a[i] * ::inv[i + 1] % mod;
        return poly(r);
    }
    poly inv(int n) const {
        assert(a[0] != 0);
        poly x(power(a[0], mod - 2));
        int k = 1;
        while (k < n) {
            k *= 2;
            x *= (poly(2) - modXn(k) * x).modXn(k);
        }
        return x.modXn(n);
    }
    // 需要保证首项为 1
    poly log(int n) const {
        return (derivation() * inv(n)).integral().modXn(n);
    }
    // 需要保证首项为 0
    poly exp(int n) const {
        poly x(1);
        int k = 1;
        while (k < n) {
            k *= 2;
            x = (x * (poly(1) - x.log(k) + modXn(k))).modXn(k);
        }
        return x.modXn(n);
    }
    // 需要保证首项为 1，开任意次方可以先 ln 再 exp 实现。
    poly sqrt(int n) const {
        poly x(1);
        int k = 1;
        while (k < n) {
            k *= 2;
            x += modXn(k) * x.inv(k);
            x = x.modXn(k) * inv2;
        }
        return x.modXn(n);
    }
    // 减法卷积，也称转置卷积 {\rm MULT}(F(x),G(x))=\sum_{i\ge0}(\sum_{j\ge
    // 0}f_{i+j}g_j)x^i
    poly mulT(poly rhs) const {
        if (rhs.size() == 0) return poly();
        int n = rhs.size();
        ::reverse(rhs.a.begin(), rhs.a.end());
        return ((*this) * rhs).divXn(n - 1);
    }
    int eval(int x) {
        int r = 0, t = 1;
        for (int i = 0, n = size(); i < n; ++i) {
            r = (r + 1ll * a[i] * t) % mod;
            t = 1ll * t * x % mod;
        }
        return r;
    }
    // 多点求值新科技：https://jkloverdcoi.github.io/2020/08/04/转置原理及其应用/
    // 模板例题：https://www.luogu.com.cn/problem/P5050
    auto evals(vector<int> &x) const {
        if (size() == 0) return vector(x.size(), 0);
        int n = x.size();
        vector ans(n, 0);
        vector<poly> g(4 * n);
        auto build = [&](auto self, int l, int r, int p) -> void {
            if (r - l == 1) {
                g[p] = poly({1, x[l] ? mod - x[l] : 0});
            } else {
                int m = (l + r) / 2;
                self(self, l, m, 2 * p);
                self(self, m, r, 2 * p + 1);
                g[p] = g[2 * p] * g[2 * p + 1];
            }
        };
        build(build, 0, n, 1);
        auto solve = [&](auto self, int l, int r, int p, poly f) -> void {
            if (r - l == 1) {
                ans[l] = f[0];
            } else {
                int m = (l + r) / 2;
                self(self, l, m, 2 * p, f.mulT(g[2 * p + 1]).modXn(m - l));
                self(self, m, r, 2 * p + 1, f.mulT(g[2 * p]).modXn(r - m));
            }
        };
        solve(solve, 0, n, 1, mulT(g[1].inv(size())).modXn(n));
        return ans;
    }
};  // 全家桶测试：https://www.luogu.com.cn/training/3015#information

auto _ = []() {
    inv[0] = inv[1] = 1;
    for (int i = 2; i < inv.size(); i++)
        inv[i] = 1ll * (mod - mod / i) * inv[mod % i] % mod;
    return true;
}();
```
### 盒子与球

$n$ 个球，$m$ 个盒

![box and ball](box-and-ball.png)

#### 球同，盒同，可空

```cpp
int solve(int n, int m) {
    vector a(n + 1, 0);
    for (int i = 1; i <= m; i++)
        for (int j = i, k = 1; j <= n; j += i, k++)
            a[j] = (a[j] + inv[k]) % mod;
    auto p = poly(a).exp(n + 1);
    return (p.a[n] + mod) % mod;
}
```

若要求不超过  $k$ 个，答案为 $[x^ny^m]\prod\limits_{i=0}^k \left(\sum\limits_{j=0}^m x^{ij}y^j\right)$。

#### 球不同，盒同，可空

```cpp
int solve(int n, int m) {
    vector a(n + 1, 0);
    vector b(n + 1, 0);
    for (int i = 0; i <= n; i++) {
        a[i] = ifac[i];
        if (i & 1) a[i] = -a[i];
        b[i] = 1ll * power(i, n) * ifac[i] % mod;
    }
    auto p = poly(a) * poly(b);
    int ans = 0;
    for (int i = 1; i <= min(n, m); i++) ans = (ans + p.a[i]) % mod;
    return (ans + mod) % mod;
}
```

若要求不超过  $k$ 个，答案为 $m! \cdot [x^ny^m]\prod\limits_{i=0}^k \left(\sum\limits_{j=0}^n\frac{1}{i!^j} x^{ij}y^j\right)$。

#### 球同，盒不同，可空

若要求不超过  $k$ 个，答案为 $[x^n]\left(\sum\limits_{i=0}^kx^i\right)^m = [x^n]\frac{(x^{k+1}-1)^m}{(x-1)^m}$。

也可以考虑容斥，令 $f(i)$ 表示至少有 $i$ 个盒子装了 $>k$ 个球方案数，$f(i) = \tbinom{m}{i}\tbinom{n-(k+1)i+m-1}{m-1}$。

总方案数则为 $\sum\limits_{i=0}^{m}(-1)^if(i)$。

#### 球同，盒不同，不可空

若要求不超过  $k$ 个，答案为 $[x^n]\left(\sum\limits_{i=1}^kx^i\right)^m = [x^n]\frac{(x^k-1)^mx^m}{(x-1)^m}$。

也可以考虑容斥，令 $f(i)$ 表示至少有 $i$ 个盒子装了 $>k$ 个球方案数，$f(i) = \tbinom{m}{i}\tbinom{n-ki-1}{m-1}$。

总方案数则为 $\sum\limits_{i=0}^{m}(-1)^if(i)$。

#### 球不同，盒不同，可空

若要求不超过  $k$ 个，答案为 $m!\cdot[x^n]\left(\sum\limits_{i=0}^k\frac{1}{i!}x^i\right)^m$。

#### 球不同，盒不同，不可空

若要求不超过  $k$ 个，答案为 $m!\cdot[x^n]\left(\sum\limits_{i=1}^k\frac{1}{i!}x^i\right)^m$。


### 线性基

```cpp
// 线性基
struct basis {
    int rnk = 0;
    array<ull, 64> p{};

    // 将x插入此线性基中
    void insert(ull x) {
        for (int i = 63; i >= 0; i--) {
            if ((x >> i) & 1) {
                if (p[i]) x ^= p[i];
                else {
                    for (int j = 0; j < i; j++)
                        if (x >> j & 1) x ^= p[j];
                    for (int j = i + 1; j <= 63; j++)
                        if (p[j] >> i & 1) p[j] ^= x;
                    p[i] = x;
                    rnk++;
                    break;
                }
            }
        }
    }

    // 将另一个线性基插入此线性基中
    void insert(basis other) {
        for (int i = 0; i <= 63; i++) {
            if (!other.p[i]) continue;
            insert(other.p[i]);
        }
    }

    // 最大异或值
    ull max_basis() {
        ull res = 0;
        for (int i = 63; i >= 0; i--)
            if ((res ^ p[i]) > res) res ^= p[i];
        return res;
    }
};
```

### 矩阵快速幂

```cpp
constexpr ll mod = 2147493647;
struct Mat {
    int n, m;
    vector<vector<ll>> mat;
    Mat(int n, int m) : n(n), m(n), mat(n, vector<ll>(m, 0)) {}
    Mat(vector<vector<ll>> mat) : n(mat.size()), m(mat[0].size()), mat(mat) {}
    Mat operator*(const Mat& other) {
        assert(m == other.n);
        Mat res(n, other.m);
        for (int i = 0; i < res.n; i++)
            for (int j = 0; j < res.m; j++)
                for (int k = 0; k < m; k++)
                    res.mat[i][j] =
                        (res.mat[i][j] + mat[i][k] * other.mat[k][j] % mod) %
                        mod;
        return res;
    }
};
Mat ksm(Mat a, ll b) {
    assert(a.n == a.m);
    Mat res(a.n, a.m);
    for (int i = 0; i < res.n; i++) res.mat[i][i] = 1;
    while (b) {
        if (b & 1) res = res * a;
        b >>= 1;
        a = a * a;
    }
    return res;
}
```

## 计算几何

### 整数

```cpp
constexpr double inf = 1e100;

// 向量
struct vec {
    static bool cmp(const vec &a, const vec &b) {
        return tie(a.x, a.y) < tie(b.x, b.y);
    }

    ll x, y;
    vec() : x(0), y(0) {}
    vec(ll _x, ll _y) : x(_x), y(_y) {}

    // 模
    ll len2() const { return x * x + y * y; }
    double len() const { return sqrt(x * x + y * y); }

    // 是否在上半轴
    bool up() const { return y > 0 || y == 0 && x >= 0; }

    bool operator==(const vec &b) const { return tie(x, y) == tie(b.x, b.y); }
    // 极角排序
    bool operator<(const vec &b) const {
        if (up() != b.up()) return up() > b.up();
        ll tmp = (*this) ^ b;
        return tmp ? tmp > 0 : cmp(*this, b);
    }

    vec operator+(const vec &b) const { return {x + b.x, y + b.y}; }
    vec operator-() const { return {-x, -y}; }
    vec operator-(const vec &b) const { return -b + (*this); }
    vec operator*(ll b) const { return {x * b, y * b}; }
    ll operator*(const vec &b) const { return x * b.x + y * b.y; }

    // 叉积 结果大于0，a到b为逆时针，小于0，a到b顺时针,
    // 等于0共线，可能同向或反向，结果绝对值表示 a b 形成的平行四边行的面积
    ll operator^(const vec &b) const { return x * b.y - y * b.x; }

    friend istream &operator>>(istream &in, vec &data) {
        in >> data.x >> data.y;
        return in;
    }
    friend ostream &operator<<(ostream &out, const vec &data) {
        out << fixed << setprecision(6);
        out << data.x << " " << data.y;
        return out;
    }
};

ll cross(const vec &a, const vec &b, const vec &c) { return (a - c) ^ (b - c); }

// 多边形的面积a
double polygon_area(vector<vec> &p) {
    ll area = 0;
    for (int i = 1; i < p.size(); i++) area += p[i - 1] ^ p[i];
    area += p.back() ^ p[0];
    return abs(area / 2.0);
}

// 多边形的周长
double polygon_len(vector<vec> &p) {
    double len = 0;
    for (int i = 1; i < p.size(); i++) len += (p[i - 1] - p[i]).len();
    len += (p.back() - p[0]).len();
    return len;
}

// 以整点为顶点的线段上的整点个数
ll count(const vec &a, const vec &b) {
    vec c = a - b;
    return gcd(abs(c.x), abs(c.y)) + 1;
}

// 以整点为顶点的多边形边上整点个数
ll count(vector<vec> &p) {
    ll cnt = 0;
    for (int i = 1; i < p.size(); i++) cnt += count(p[i - 1], p[i]);
    cnt += count(p.back(), p[0]);
    return cnt - p.size();
}

// 判断点是否在凸包内，凸包必须为逆时针顺序
bool in_polygon(const vec &a, vector<vec> &p) {
    int n = p.size();
    if (n == 0) return 0;
    if (n == 1) return a == p[0];
    if (n == 2)
        return cross(a, p[1], p[0]) == 0 && (p[0] - a) * (p[1] - a) <= 0;
    if (cross(a, p[1], p[0]) > 0 || cross(p.back(), a, p[0]) > 0) return 0;
    auto cmp = [&](vec &x, const vec &y) { return ((x - p[0]) ^ y) >= 0; };
    int i =
        lower_bound(p.begin() + 2, p.end() - 1, a - p[0], cmp) - p.begin() - 1;
    return cross(p[(i + 1) % n], a, p[i]) >= 0;
}

// 凸包直径的两个端点
auto polygon_dia(vector<vec> &p) {
    int n = p.size();
    array<vec, 2> res{};
    if (n == 1) return res;
    if (n == 2) return res = {p[0], p[1]};
    ll mx = 0;
    for (int i = 0, j = 2; i < n; i++) {
        while (abs(cross(p[i], p[(i + 1) % n], p[j])) <=
               abs(cross(p[i], p[(i + 1) % n], p[(j + 1) % n])))
            j = (j + 1) % n;
        ll tmp = (p[i] - p[j]).len2();
        if (tmp > mx) {
            mx = tmp;
            res = {p[i], p[j]};
        }
        tmp = (p[(i + 1) % n] - p[j]).len2();
        if (tmp > mx) {
            mx = tmp;
            res = {p[(i + 1) % n], p[j]};
        }
    }
    return res;
}

// 凸包
auto convex_hull(vector<vec> &p) {
    sort(p.begin(), p.end(), vec::cmp);
    int n = p.size();
    vector sta(n + 1, 0);
    vector v(n, false);
    int tp = -1;
    sta[++tp] = 0;
    auto update = [&](int lim, int i) {
        while (tp > lim && cross(p[i], p[sta[tp]], p[sta[tp - 1]]) >= 0)
            v[sta[tp--]] = 0;
        sta[++tp] = i;
        v[i] = 1;
    };
    for (int i = 1; i < n; i++) update(0, i);
    int cnt = tp;
    for (int i = n - 1; i >= 0; i--) {
        if (v[i]) continue;
        update(cnt, i);
    }
    vector<vec> res(tp);
    for (int i = 0; i < tp; i++) res[i] = p[sta[i]];
    return res;
}

// 闵可夫斯基和，两个点集的和构成一个凸包
auto minkowski(vector<vec> &a, vector<vec> &b) {
    rotate(a.begin(), min_element(a.begin(), a.end(), vec::cmp), a.end());
    rotate(b.begin(), min_element(b.begin(), b.end(), vec::cmp), b.end());
    int n = a.size(), m = b.size();
    vector<vec> c{a[0] + b[0]};
    c.reserve(n + m);
    int i = 0, j = 0;
    while (i < n && j < m) {
        vec x = a[(i + 1) % n] - a[i];
        vec y = b[(j + 1) % m] - b[j];
        c.push_back(c.back() + ((x ^ y) >= 0 ? (i++, x) : (j++, y)));
    }
    while (i + 1 < n) {
        c.push_back(c.back() + a[(i + 1) % n] - a[i]);
        i++;
    }
    while (j + 1 < m) {
        c.push_back(c.back() + b[(j + 1) % m] - b[j]);
        j++;
    }
    return c;
}

// 过凸多边形外一点求凸多边形的切线，返回切点下标
auto tangent(const vec &a, vector<vec> &p) {
    int n = p.size();
    int l = -1, r = -1;
    for (int i = 0; i < n; i++) {
        ll tmp1 = cross(p[i], p[(i - 1 + n) % n], a);
        ll tmp2 = cross(p[i], p[(i + 1) % n], a);
        if (l == -1 && tmp1 <= 0 && tmp2 <= 0) l = i;
        else if (r == -1 && tmp1 >= 0 && tmp2 >= 0) r = i;
    }
    return array{l, r};
}

// 直线
struct line {
    vec p, d;
    line() : p(vec()), d(vec()) {}
    line(const vec &_p, const vec &_d) : p(_p), d(_d) {}
};

// 点到直线距离
double dis(const vec &a, const line &b) {
    return abs((b.p - a) ^ (b.p + b.d - a)) / b.d.len();
}

// 点在直线哪边，大于0在左边，等于0在线上，小于0在右边
ll side_line(const vec &a, const line &b) { return b.d ^ (a - b.p); }

// 两直线是否垂直
bool perpen(const line &a, const line &b) { return a.d * b.d == 0; }

// 两直线是否平行
bool parallel(const line &a, const line &b) { return (a.d ^ b.d) == 0; }

// 点的垂线是否与线段有交点
bool perpen(const vec &a, const line &b) {
    vec p(-b.d.y, b.d.x);
    bool cross1 = (p ^ (b.p - a)) > 0;
    bool cross2 = (p ^ (b.p + b.d - a)) > 0;
    return cross1 != cross2;
}

// 点到线段距离
double dis_seg(const vec &a, const line &b) {
    if (perpen(a, b)) return dis(a, b);
    return min((b.p - a).len(), (b.p + b.d - a).len());
}

// 点到凸包距离
double dis(const vec &a, vector<vec> &p) {
    double res = inf;
    for (int i = 1; i < p.size(); i++)
        res = min(dis_seg(a, line(p[i - 1], p[i] - p[i - 1])), res);
    res = min(dis_seg(a, line(p.back(), p[0] - p.back())), res);
    return res;
}

// 两直线交点
vec intersection(ll A, ll B, ll C, ll D, ll E, ll F) {
    return {(B * F - C * E) / (A * E - B * D),
            (C * D - A * F) / (A * E - B * D)};
}

// 两直线交点
vec intersection(const line &a, const line &b) {
    return intersection(a.d.y, -a.d.x, a.d.x * a.p.y - a.d.y * a.p.x, b.d.y,
                        -b.d.x, b.d.x * b.p.y - b.d.y * b.p.x);
}
```

### 浮点数

```cpp
constexpr lf eps = 1e-6;
constexpr lf inf = 1e100;
const lf PI = acos(-1);

int sgn(lf a, lf b) {
    lf c = a - b;
    return c < -eps ? -1 : c < eps ? 0 : 1;
}

// 向量
struct vec {
    static bool cmp(const vec &a, const vec &b) {
        return sgn(a.x, b.x) ? a.x < b.x : sgn(a.y, b.y) < 0;
    }

    lf x, y;
    vec() : x(0), y(0) {}
    vec(lf _x, lf _y) : x(_x), y(_y) {}

    // 模
    lf len2() const { return x * x + y * y; }
    lf len() const { return sqrt(x * x + y * y); }

    // 与x轴正方向的夹角
    lf angle() const {
        lf angle = atan2(y, x);
        if (angle < 0) angle += 2 * PI;
        return angle;
    }

    // 逆时针旋转
    vec rotate(const lf &theta) const {
        return {x * cos(theta) - y * sin(theta),
                y * cos(theta) + x * sin(theta)};
    }

    vec e() const {
        lf tmp = len();
        return {x / tmp, y / tmp};
    }

    // 是否在上半轴
    bool up() const {
        return sgn(y, 0) > 0 || sgn(y, 0) == 0 && sgn(x, 0) >= 0;
    }

    bool operator==(const vec &other) const {
        return sgn(x, other.x) == 0 && sgn(y, other.y) == 0;
    }
    // 极角排序
    bool operator<(const vec &b) const {
        if (up() != b.up()) return up() > b.up();
        lf tmp = (*this) ^ b;
        return sgn(tmp, 0) ? tmp > 0 : cmp(*this, b);
    }

    vec operator+(const vec &b) const { return {x + b.x, y + b.y}; }
    vec operator-() const { return {-x, -y}; }
    vec operator-(const vec &b) const { return -b + (*this); }
    vec operator*(lf b) const { return {x * b, y * b}; }
    vec operator/(lf b) const { return {x / b, y / b}; }
    lf operator*(const vec &b) const { return x * b.x + y * b.y; }

    // 叉积 结果大于0，a到b为逆时针，小于0，a到b顺时针,
    // 等于0共线，可能同向或反向，结果绝对值表示 a b 形成的平行四边行的面积
    lf operator^(const vec &b) const { return x * b.y - y * b.x; }

    friend istream &operator>>(istream &in, vec &data) {
        in >> data.x >> data.y;
        return in;
    }
    friend ostream &operator<<(ostream &out, const vec &data) {
        out << fixed << setprecision(6);
        out << data.x << " " << data.y;
        return out;
    }
};

lf cross(const vec &a, const vec &b, const vec &c) { return (a - c) ^ (b - c); }

lf angle(const vec &a, const vec &b) { return atan2(abs(a ^ b), a * b); }

// 多边形的面积
lf polygon_area(vector<vec> &p) {
    lf area = 0;
    for (int i = 1; i < p.size(); i++) area += p[i - 1] ^ p[i];
    area += p.back() ^ p[0];
    return abs(area / 2.0);
}

// 多边形的周长
lf polygon_len(vector<vec> &p) {
    lf len = 0;
    for (int i = 1; i < p.size(); i++) len += (p[i - 1] - p[i]).len();
    len += (p.back() - p[0]).len();
    return len;
}

// 判断点是否在凸包内，凸包必须为逆时针顺序
bool in_polygon(const vec &a, vector<vec> &p) {
    int n = p.size();
    if (n == 0) return 0;
    if (n == 1) return a == p[0];
    if (n == 2)
        return sgn(cross(a, p[1], p[0]), 0) == 0 &&
               sgn((p[0] - a) * (p[1] - a), 0) <= 0;
    if (sgn(cross(a, p[1], p[0]), 0) > 0 ||
        sgn(cross(p.back(), a, p[0]), 0) > 0)
        return 0;
    auto cmp = [&](vec &x, const vec &y) {
        return sgn((x - p[0]) ^ y, 0) >= 0;
    };
    int i =
        lower_bound(p.begin() + 2, p.end() - 1, a - p[0], cmp) - p.begin() - 1;
    return sgn(cross(p[(i + 1) % n], a, p[i]), 0) >= 0;
}

// 凸包直径的两个端点
auto polygon_dia(vector<vec> &p) {
    int n = p.size();
    array<vec, 2> res{};
    if (n == 1) return res;
    if (n == 2) return res = {p[0], p[1]};
    lf mx = 0;
    for (int i = 0, j = 2; i < n; i++) {
        while (sgn(abs(cross(p[i], p[(i + 1) % n], p[j])),
                   abs(cross(p[i], p[(i + 1) % n], p[(j + 1) % n]))) <= 0)
            j = (j + 1) % n;
        lf tmp = (p[i] - p[j]).len();
        if (tmp > mx) {
            mx = tmp;
            res = {p[i], p[j]};
        }
        tmp = (p[(i + 1) % n] - p[j]).len();
        if (tmp > mx) {
            mx = tmp;
            res = {p[(i + 1) % n], p[j]};
        }
    }
    return res;
}

// 凸包
auto convex_hull(vector<vec> &p) {
    sort(p.begin(), p.end(), vec::cmp);
    int n = p.size();
    vector sta(n + 1, 0);
    vector v(n, false);
    int tp = -1;
    sta[++tp] = 0;
    auto update = [&](int lim, int i) {
        while (tp > lim && sgn(cross(p[i], p[sta[tp]], p[sta[tp - 1]]), 0) >= 0)
            v[sta[tp--]] = 0;
        sta[++tp] = i;
        v[i] = 1;
    };
    for (int i = 1; i < n; i++) update(0, i);
    int cnt = tp;
    for (int i = n - 1; i >= 0; i--) {
        if (v[i]) continue;
        update(cnt, i);
    }
    vector<vec> res(tp);
    for (int i = 0; i < tp; i++) res[i] = p[sta[i]];
    return res;
}

// 闵可夫斯基和，两个点集的和构成一个凸包
auto minkowski(vector<vec> &a, vector<vec> &b) {
    rotate(a.begin(), min_element(a.begin(), a.end(), vec::cmp), a.end());
    rotate(b.begin(), min_element(b.begin(), b.end(), vec::cmp), b.end());
    int n = a.size(), m = b.size();
    vector<vec> c{a[0] + b[0]};
    c.reserve(n + m);
    int i = 0, j = 0;
    while (i < n && j < m) {
        vec x = a[(i + 1) % n] - a[i];
        vec y = b[(j + 1) % m] - b[j];
        c.push_back(c.back() + (sgn(x ^ y, 0) >= 0 ? (i++, x) : (j++, y)));
    }
    while (i + 1 < n) {
        c.push_back(c.back() + a[(i + 1) % n] - a[i]);
        i++;
    }
    while (j + 1 < m) {
        c.push_back(c.back() + b[(j + 1) % m] - b[j]);
        j++;
    }
    return c;
}

// 过凸多边形外一点求凸多边形的切线，返回切点下标
auto tangent(const vec &a, vector<vec> &p) {
    int n = p.size();
    int l = -1, r = -1;
    for (int i = 0; i < n; i++) {
        lf tmp1 = cross(p[i], p[(i - 1 + n) % n], a);
        lf tmp2 = cross(p[i], p[(i + 1) % n], a);
        if (l == -1 && sgn(tmp1, 0) <= 0 && sgn(tmp2, 0) <= 0) l = i;
        else if (r == -1 && sgn(tmp1, 0) >= 0 && sgn(tmp2, 0) >= 0) r = i;
    }
    return array{l, r};
}

// 直线
struct line {
    vec p, d;
    line() : p(vec()), d(vec()) {}
    line(const vec &_p, const vec &_d) : p(_p), d(_d) {}
};

// 点到直线距离
lf dis(const vec &a, const line &b) {
    return abs((b.p - a) ^ (b.p + b.d - a)) / b.d.len();
}

// 点在直线哪边，大于0在左边，等于0在线上，小于0在右边
int side_line(const vec &a, const line &b) { return sgn(b.d ^ (a - b.p), 0); }

// 两直线是否垂直
bool perpen(const line &a, const line &b) { return sgn(a.d * b.d, 0) == 0; }

// 两直线是否平行
bool parallel(const line &a, const line &b) { return sgn(a.d ^ b.d, 0) == 0; }

// 点的垂线是否与线段有交点
bool perpen(const vec &a, const line &b) {
    vec p(-b.d.y, b.d.x);
    bool cross1 = sgn(p ^ (b.p - a), 0) > 0;
    bool cross2 = sgn(p ^ (b.p + b.d - a), 0) > 0;
    return cross1 != cross2;
}

// 点到线段距离
lf dis_seg(const vec &a, const line &b) {
    if (perpen(a, b)) return dis(a, b);
    return min((b.p - a).len(), (b.p + b.d - a).len());
}

// 点到凸包距离
lf dis(const vec &a, vector<vec> &p) {
    lf res = inf;
    for (int i = 1; i < p.size(); i++)
        res = min(dis_seg(a, line(p[i - 1], p[i] - p[i - 1])), res);
    res = min(dis_seg(a, line(p.back(), p[0] - p.back())), res);
    return res;
}

// 两直线交点
vec intersection(lf A, lf B, lf C, lf D, lf E, lf F) {
    return {(B * F - C * E) / (A * E - B * D),
            (C * D - A * F) / (A * E - B * D)};
}

// 两直线交点
vec intersection(const line &a, const line &b) {
    return intersection(a.d.y, -a.d.x, a.d.x * a.p.y - a.d.y * a.p.x, b.d.y,
                        -b.d.x, b.d.x * b.p.y - b.d.y * b.p.x);
}

struct circle {
    vec o;
    lf r;
    circle(const vec &_o, lf _r) : o(_o), r(_r){};
    // 点与圆的关系 -1在圆内，0在圆上，1在圆外
    int relation(const vec &a) const {
        lf len = (a - o).len();
        return sgn(len, r);
    }
    lf area() { return PI * r * r; }
};

// 圆与直线交点
auto intersection(const circle &c, const line &l) {
    lf d = dis(c.o, l);
    vector<vec> res;
    vec mid = l.p + l.d.e() * ((c.o - l.p) * l.d / l.d.len());
    if (sgn(d, c.r) == 0) res.push_back(mid);
    else if (sgn(d, c.r) < 0) {
        d = sqrt(c.r * c.r - d * d);
        res.push_back(mid + l.d.e() * d);
        res.push_back(mid - l.d.e() * d);
    }
    return res;
}

// oab三角形与圆相交的面积
lf area(const circle &c, const vec &a, const vec &b) {
    if (sgn(cross(a, b, c.o), 0) == 0) return 0;
    vector<vec> p;
    p.push_back(a);
    line l(a, b - a);
    auto tmp = intersection(c, l);
    if (tmp.size() == 2) {
        for (auto &i : tmp)
            if (sgn((a - i) * (b - i), 0) < 0) p.push_back(i);
    }
    p.push_back(b);
    if (p.size() == 4 && sgn((p[0] - p[1]) * (p[2] - p[1]), 0) > 0)
        swap(p[1], p[2]);
    lf res = 0;
    for (int i = 1; i < p.size(); i++)
        if (c.relation(p[i - 1]) == 1 || c.relation(p[i]) == 1) {
            lf ang = angle(p[i - 1] - c.o, p[i] - c.o);
            res += c.r * c.r * ang / 2;
        } else res += abs(cross(p[i - 1], p[i], c.o)) / 2.0;
    return res;
}

// 多边形与圆相交的面积
lf area(vector<vec> &p, circle c) {
    lf res = 0;
    for (int i = 0; i < p.size(); i++) {
        int j = i + 1 == p.size() ? 0 : i + 1;
        if (sgn(cross(p[i], p[j], c.o), 0) <= 0) res += area(c, p[i], p[j]);
        else res -= area(c, p[i], p[j]);
    }
    return abs(res);
}
```

### 扫描线

```cpp
#define ls (pos << 1)
#define rs (ls | 1)
#define mid ((tree[pos].l + tree[pos].r) >> 1)
struct Rectangle {
    ll x_l, y_l, x_r, y_r;
};
ll area(vector<Rectangle>& rec) {
    struct Line {
        ll x, y_up, y_down;
        int pd;
    };
    vector<Line> line(rec.size() * 2);
    vector<ll> y_set(rec.size() * 2);
    for (int i = 0; i < rec.size(); i++) {
        y_set[i * 2] = rec[i].y_l;
        y_set[i * 2 + 1] = rec[i].y_r;
        line[i * 2] = {rec[i].x_l, rec[i].y_r, rec[i].y_l, 1};
        line[i * 2 + 1] = {rec[i].x_r, rec[i].y_r, rec[i].y_l, -1};
    }
    sort(y_set.begin(), y_set.end());
    y_set.erase(unique(y_set.begin(), y_set.end()), y_set.end());
    sort(line.begin(), line.end(), [](Line a, Line b) { return a.x < b.x; });
    struct Data {
        int l, r;
        ll len, cnt, raw_len;
    };
    vector<Data> tree(4 * y_set.size());
    function<void(int, int, int)> build = [&](int pos, int l, int r) {
        tree[pos].l = l;
        tree[pos].r = r;
        if (l == r) {
            tree[pos].raw_len = y_set[r + 1] - y_set[l];
            tree[pos].cnt = tree[pos].len = 0;
            return;
        }
        build(ls, l, mid);
        build(rs, mid + 1, r);
        tree[pos].raw_len = tree[ls].raw_len + tree[rs].raw_len;
    };
    function<void(int, int, int, int)> update = [&](int pos, int l, int r,
                                                    int num) {
        if (l <= tree[pos].l && tree[pos].r <= r) {
            tree[pos].cnt += num;
            tree[pos].len = tree[pos].cnt ? tree[pos].raw_len
                            : tree[pos].l == tree[pos].r
                                ? 0
                                : tree[ls].len + tree[rs].len;
            return;
        }
        if (l <= mid) update(ls, l, r, num);
        if (r > mid) update(rs, l, r, num);
        tree[pos].len =
            tree[pos].cnt ? tree[pos].raw_len : tree[ls].len + tree[rs].len;
    };
    build(1, 0, y_set.size() - 2);
    auto find_pos = [&](ll num) {
        return lower_bound(y_set.begin(), y_set.end(), num) - y_set.begin();
    };
    ll res = 0;
    for (int i = 0; i < line.size() - 1; i++) {
        update(1, find_pos(line[i].y_down), find_pos(line[i].y_up) - 1,
               line[i].pd);
        res += (line[i + 1].x - line[i].x) * tree[1].len;
    }
    return res;
}
```

## 杂项

### 高精度

```cpp
struct bignum {
    string num;

    bignum() : num("0") {}
    bignum(const string& num) : num(num) {
        reverse(this->num.begin(), this->num.end());
    }
    bignum(ll num) : num(to_string(num)) {
        reverse(this->num.begin(), this->num.end());
    }

    bignum operator+(const bignum& other) {
        bignum res;
        res.num.pop_back();
        res.num.reserve(max(num.size(), other.num.size()) + 1);
        for (int i = 0, j = 0, x; i < num.size() || i < other.num.size() || j;
             i++) {
            x = j;
            j = 0;
            if (i < num.size()) x += num[i] - '0';
            if (i < other.num.size()) x += other.num[i] - '0';
            if (x >= 10) j = 1, x -= 10;
            res.num.push_back(x + '0');
        }
        res.num.capacity();
        return res;
    }

    bignum operator*(const bignum& other) {
        vector<int> res(num.size() + other.num.size() - 1, 0);
        for (int i = 0; i < num.size(); i++)
            for (int j = 0; j < other.num.size(); j++)
                res[i + j] += (num[i] - '0') * (other.num[j] - '0');
        int g = 0;
        for (int i = 0; i < res.size(); i++) {
            res[i] += g;
            g = res[i] / 10;
            res[i] %= 10;
        }
        while (g) {
            res.push_back(g % 10);
            g /= 10;
        }
        int lim = res.size();
        while (lim > 1 && res[lim - 1] == 0) lim--;
        bignum res2;
        res2.num.resize(lim);
        for (int i = 0; i < lim; i++) res2.num[i] = res[i] + '0';
        return res2;
    }

    bool operator<(const bignum& other) {
        if (num.size() == other.num.size())
            for (int i = num.size() - 1; i >= 0; i--)
                if (num[i] == other.num[i]) continue;
                else return num[i] < other.num[i];
        return num.size() < other.num.size();
    }

    friend istream& operator>>(istream& in, bignum& a) {
        in >> a.num;
        reverse(a.num.begin(), a.num.end());
        return in;
    }
    friend ostream& operator<<(ostream& out, bignum a) {
        reverse(a.num.begin(), a.num.end());
        return out << a.num;
    }
};
```

### 模运算

```cpp
struct modint {
    int x;
    modint(ll _x = 0) : x(_x % mod) {}
    modint inv() const { return power(*this, mod - 2); }
    modint operator+(const modint& b) { return {x + b.x}; }
    modint operator-() const { return {-x}; }
    modint operator-(const modint& b) { return {-b + *this}; }
    modint operator*(const modint& b) { return {(ll)x * b.x}; }
    modint operator/(const modint& b) { return *this * b.inv(); }
    friend istream& operator>>(istream& is, modint& other) {
        ll _x;
        is >> _x;
        other = modint(_x);
        return is;
    }
    friend ostream& operator<<(ostream& os, modint other) {
        other.x = (other.x + mod) % mod;
        return os << other.x;
    }
};
```

### 分数

```cpp
struct frac {
    ll a, b;
    frac() : a(0), b(1) {}
    frac(ll _a, ll _b) : a(_a), b(_b) {
        assert(b);
        if (a) {
            int tmp = gcd(a, b);
            a /= tmp;
            b /= tmp;
        } else *this = frac();
    }
    frac operator+(const frac& other) {
        return frac(a * other.b + other.a * b, b * other.b);
    }
    frac operator-() const {
        frac res = *this;
        res.a = -res.a;
        return res;
    }
    frac operator-(const frac& other) const { return -other + *this; }
    frac operator*(const frac& other) const {
        return frac(a * other.a, b * other.b);
    }
    frac operator/(const frac& other) const {
        assert(other.a);
        return *this * frac(other.b, other.a);
    }
    bool operator<(const frac& other) const { return (*this - other).a < 0; }
    bool operator<=(const frac& other) const { return (*this - other).a <= 0; }
    bool operator>=(const frac& other) const { return (*this - other).a >= 0; }
    bool operator>(const frac& other) const { return (*this - other).a > 0; }
    bool operator==(const frac& other) const {
        return a == other.a && b == other.b;
    }
    bool operator!=(const frac& other) const { return !(*this == other); }
};
```

### 表达式求值

```cpp
// 格式化表达式
string format(const string& s1) {
    stringstream ss(s1);
    string s2;
    char ch;
    while ((ch = ss.get()) != EOF) {
        if (ch == ' ') continue;
        if (isdigit(ch)) s2 += ch;
        else {
            if (s2.back() != ' ') s2 += ' ';
            s2 += ch;
            s2 += ' ';
        }
    }
    return s2;
}

// 中缀表达式转后缀表达式
string convert(const string& s1) {
    unordered_map<char, int> rank{
        {'+', 2}, {'-', 2}, {'*', 1}, {'/', 1}, {'^', 0}};
    stringstream ss(s1);
    string s2, temp;
    stack<char> op;
    while (ss >> temp) {
        if (isdigit(temp[0])) s2 += temp + ' ';
        else if (temp[0] == '(') op.push('(');
        else if (temp[0] == ')') {
            while (op.top() != '(') {
                s2 += op.top();
                s2 += ' ';
                op.pop();
            }
            op.pop();
        } else {
            while (!op.empty() && op.top() != '(' &&
                   (temp[0] != '^' && rank[op.top()] <= rank[temp[0]] ||
                    rank[op.top()] < rank[temp[0]])) {
                s2 += op.top();
                s2 += ' ';
                op.pop();
            }
            op.push(temp[0]);
        }
    }
    while (!op.empty()) {
        s2 += op.top();
        s2 += ' ';
        op.pop();
    }
    return s2;
}

// 计算后缀表达式
int calc(const string& s) {
    stack<int> num;
    stringstream ss(s);
    string temp;
    while (ss >> temp) {
        if (isdigit(temp[0])) num.push(stoi(temp));
        else {
            int b = num.top();
            num.pop();
            int a = num.top();
            num.pop();
            if (temp[0] == '+') a += b;
            else if (temp[0] == '-') a -= b;
            else if (temp[0] == '*') a *= b;
            else if (temp[0] == '/') a /= b;
            else if (temp[0] == '^') a = ksm(a, b);
            num.push(a);
        }
    }
    return num.top();
}
```

### 日期

```cpp
int month[] = {0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
int pre[13];
vector<int> leap;
struct Date {
    int y, m, d;
    bool operator<(const Date& other) const {
        return array<int, 3>{y, m, d} <
               array<int, 3>{other.y, other.m, other.d};
    }
    Date(const string& s) {
        stringstream ss(s);
        char ch;
        ss >> y >> ch >> m >> ch >> d;
    }
    int dis() const {
        int yd = (y - 1) * 365 +
                 (upper_bound(leap.begin(), leap.end(), y - 1) - leap.begin());
        int md =
            pre[m - 1] + (m > 2 && (y % 4 == 0 && y % 100 || y % 400 == 0));
        return yd + md + d;
    }
    int dis(const Date& other) const { return other.dis() - dis(); }
};
for (int i = 1; i <= 12; i++) pre[i] = pre[i - 1] + month[2];
for (int i = 1; i <= 1000000; i++)
    if (i % 4 == 0 && i % 100 || i % 400 == 0) leap.push_back(i);
```

### 对拍

linux/Mac

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

windows

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

### 编译常用选项

```bash
-Wall -Woverflow -Wextra -Wpedantic -Wfloat-equal -Wshadow -fsanitize=address,undefined
```

### 开栈

不同的编译器可能命令不一样

```bash
-Wl,--stack=0x10000000
-Wl,-stack_size -Wl,0x10000000
-Wl,-z,stack-size=0x10000000
```

### clang-format

```bash
BasedOnStyle: Google
IndentWidth: 4
ColumnLimit: 80
AllowShortIfStatementsOnASingleLine: AllIfsAndElse
AccessModifierOffset: -4
EmptyLineBeforeAccessModifier: Leave
RemoveBracesLLVM: true
```

