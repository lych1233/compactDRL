from tqdm import tqdm
import time

for _ in tqdm(range(1000), ncols=100):
    time.sleep(0.01)