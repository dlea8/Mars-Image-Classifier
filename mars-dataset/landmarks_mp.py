#!/usr/bin/env python
# Helpful problem-specific info for landmark classification
# Kiri Wagstaff, 7/20/17

class_map = {0: 'other',
             1: 'crater',
             2: 'dune',
             3: 'edge'}

# reverse_class_map needs to be consistent with class_map
reverse_class_map = {v: k for k,v in class_map.items()}
