#!/usr/bin/env python
"""mapper_pi.py"""
import random
import sys


for line in sys.stdin:

        key=line.strip()
       
        x, y = random.random(), random.random()
        if x*x + y*y < 1:
            print(1)
        else:
            print(0) 


