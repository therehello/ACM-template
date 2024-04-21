#!/usr/bin/env python
import os.path
import re
import sys

def cpp(match: re.Match):
    code = match.group(2)
    if match.group(1) == "cpp":
        code_path = os.path.join(dir_name, "tmp_creat_by_md-format.cpp")
        with open(code_path, "w") as f:
            f.write(code)
        cmd = f'clang-format -i {code_path}'
        os.system(cmd)
        with open(code_path, "r") as f:
            code = f.read()
        os.remove(code_path)
    return f"```{match.group(1)}\n{code}```"


args = sys.argv[1:]

if len(args) < 1:
    print("请提供路径作为命令行参数")
    sys.exit(1)

file_path = args[0]

with open(file_path) as f:
    md = f.read()

file_name = os.path.splitext(os.path.basename(file_path))[0]

dir_name = os.path.dirname(file_path)

md = re.sub("```(.*?)\n(.*?)```", cpp, md, flags=re.DOTALL)

with open(file_path, "w") as f:
    f.write(md)
