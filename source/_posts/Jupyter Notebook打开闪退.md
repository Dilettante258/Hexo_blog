---
title: Jupyter Notebook打开闪退
categories: Python
date: 2023-09-12 18:45:00
---

错误提示：

```python
Traceback (most recent call last):
File "C:\Anaconda3\Scripts\jupyter-notebook-script.py", line 6, in 
from notebook.notebookapp import main
ModuleNotFoundError: No module named 'notebook.notebookapp'
```

解决办法：

Powershell Prompt中输入指令

```bash
pip install --upgrade --force-reinstall notebook
```