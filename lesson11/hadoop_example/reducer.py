#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

current_word = None
current_count = 0
word = None

# input comes from standard input 
for line in sys.stdin:
    # remove leading and trailing whitespace
    line = line.strip()

    # parse the input obtained from mapper.py
    word, count = line.split('\t', 1)

    # convert count (currently a string) to int
    try:
        count = int(count)
    except ValueError:
        # count was not a number, so silently
        # ignore/discard this line
        continue

    # this statement assumes that  Hadoop sorts map output
    # by key (here: word) before passing it to the reducer
    if current_word == word:
        current_count += count
    else:
        if current_word:
            # write result to standard output 
            print( '%s\t%s' % (current_word, current_count) )
        current_count = count
        current_word = word

# ensure last word output if needed 
if current_word == word:
    print( '%s\t%s' % (current_word, current_count))
