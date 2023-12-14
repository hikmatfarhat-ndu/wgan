# from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# parser.add_argument("--weights-path", type=str, help="weights path")
# parser.add_argument("--device",choices=['cpu','cuda'])
# args=parser.parse_args()
# print(args)
from dcgan import Generator

import torch
g=Generator(64,3,128,"GroupNorm","tanh")

# def recurse(m):
#     c=m.children()
#     for x in c:
#         if list(x.children())==[]:
#             print(x) 
#         else:
#             recurse(x)
def recurse(m):
    c=list(m.children())
    if c==[]:
        print(m)
    for x in c:
        recurse(x)
recurse(g)
# c=g.children()
# for x in c:
#     if x.children()==[]:
#         print(x) 
