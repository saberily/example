#!/usr/bin/env python
import os, subprocess

processCmd = ["python"]
task = subprocess.Popen(processCmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
task.stdin.write("print(1) \n")
task.stdin.write("print(2) \n")
task.stdin.write("print(3) \n")
out, err = task.communicate()
print(out)
print(err)