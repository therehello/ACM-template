# therehllo 的 算法板子

你可以选择以下方式阅读：

[markdown](/src/template.md)

[pdf](/out/template.pdf)

## 构建

```shell
md-format.py template.md
md2tex.py template.md
```

将`md`中的代码使用`clang-format`统一格式化。然后将`md`转为`tex`，方便导出`pdf`。
`pdf`是使用`vscode`生成的。


曾尝试使用`pandoc`将`md`转`tex`，但转换的很奇怪，所以写了一个最简单的转换工具，表格比较麻烦，建议把表格转为图片。
