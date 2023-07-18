#!/usr/bin/env python
import os.path
import re
import sys

begin = r"""\documentclass[UTF8]{ctexart}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{afterpage}
\usepackage{xcolor}
\usepackage{fontspec}
\usepackage{listings}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage{float}
\usepackage{keyval}

\ctexset{secnumdepth=4,tocdepth=4}
\definecolor{commentgreen}{RGB}{2,112,10}
\definecolor{eminence}{RGB}{108,48,130}
\definecolor{weborange}{RGB}{255,165,0}
\definecolor{frenchplum}{RGB}{129,20,83}

% 定义代码样式
\lstdefinestyle{cpp}{
  language=C++,
  basicstyle=\ttfamily\small,
  backgroundcolor=\color{gray!10},
  keywordstyle=\color{blue}\bfseries,
  commentstyle=\color{green!70!black},
  stringstyle=\color{red},
  numbers=left,
  numbersep=5pt,
  frame=single,
  breaklines=true,
  captionpos=b,
  showstringspaces=false
}

\newcommand\blankpage{
    \null
    \thispagestyle{empty}
    \addtocounter{page}{-1}
    \newpage
}


\setmonofont{Consolas} % 设置代码字体

\CTEXsetup[format={\Large\bfseries}]{section}

\title{ACM常用算法模板}
\author{therehello}
\date{\today}

\begin{document}
\begin{sloppypar}

\begin{titlepage}
    \centering
    \vspace*{\stretch{0.382}} % 将标题位置移至页面垂直位置的38.2%
    {\fontsize{40pt}{0pt}\selectfont \textbf{ACM常用算法模板}\par} % 标题
    \vfill % 将剩余的垂直空间填充至页面底部
    {\fontsize{12pt}{0pt}\selectfont therehello\par} % 作者
    \today % 日期
\end{titlepage}

\blankpage

\tableofcontents

"""

end = "\n\n\\end{sloppypar}\n\\end{document}"


def section(match: re.Match):
    level = len(match.group(1)) - 2
    name = match.group(2)
    clearpage = ""
    if level == 0:
        clearpage = "\\clearpage\n\n"
    if level == 3:
        return "\\" + "paragraph{" + name + "}\n"
    return clearpage + "\\" + "sub" * level + "section{" + name + "}\n"


def cpp(match: re.Match):
    code = match.group(1)
    return f"\\begin{{lstlisting}}[style=cpp]\n{code}\\end{{lstlisting}}"


def picture(match: re.Match):
    return f"\\begin{{figure}}[H]\n    \\flushleft\n    \\includegraphics[]{{{match.group(1)}}}\n    \\label{{fig:left}}\n\\end{{figure}}"


args = sys.argv[1:]

if len(args) < 1:
    print("请提供路径作为命令行参数")
    sys.exit(1)

file_path = args[0]

with open(file_path) as f:
    md = f.read()

file_name = os.path.splitext(os.path.basename(file_path))[0]

dir_name = os.path.dirname(file_path)

tex_file_name = file_name + ".tex"

tex_file_path = os.path.join(os.path.dirname(file_path), tex_file_name)

md = re.sub("(##+) (.*?)\n", section, md)
md = re.sub("```.*?\n(.*?)```", cpp, md, flags=re.DOTALL)
md = re.sub("# .*?\n", "", md)
md = re.sub("<!-- TOC -->.*?<!-- /TOC -->", "", md, flags=re.DOTALL)
md = re.sub(r"!\[.*?]\((.*?)\)", picture, md)
md = md.strip()

tex = begin + md + end

with open(tex_file_path, "w") as f:
    f.write(tex)
