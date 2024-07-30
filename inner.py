get_ipython()
from tqdm.notebook import tqdm
import time
def doit():
    for i in tqdm(range(10)):
        time.sleep(1)