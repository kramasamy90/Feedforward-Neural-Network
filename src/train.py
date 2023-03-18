import sys
import ann
import maps

args = {}

# Read and store argument in a map/dict.

for i in range(1, int((len(sys.argv) - 1)/2) + 1):
    print(i)
    arg = sys.argv[2 * i - 1]
    print(arg)
    val = sys.argv[2 * i]
    if(arg[1] == '-'):
        arg = arg[2:]
        args[arg] = val
    else:
        arg = maps.arg_short_to_long[arg[1:]]
    args[arg] = val


nn = ann.ann()