#!/usr/bin/env python
"""reducer_pi.py"""

# input comes from standard input 
import sys

count=0
for line in sys.stdin:
    count=count+int(line) 

print(count)
