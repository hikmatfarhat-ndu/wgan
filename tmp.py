from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--weights-path", type=str, help="weights path")
parser.add_argument("--device",choices=['cpu','cuda'])
args=parser.parse_args()
print(args)

