import time
import sys

for i in range(100):
    # we do the work here!
    # time.sleep(0.1)
    sys.stdout.write("\r%d%%" % i)
    sys.stdout.flush()