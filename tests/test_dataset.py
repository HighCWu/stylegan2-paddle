import math
import argparse 
from paddle import io
from stylegan2_paddle.stylegan2_paddle import Dataset, cycle


parser = argparse.ArgumentParser() 

parser.add_argument('--folder', type=str, default='../../data/ffhq')
args, _ = parser.parse_known_args()

dataset = Dataset(args.folder, 128, transparent = False, aug_prob = 0.)
num_workers = 8
dataloader = io.DataLoader(dataset, num_workers = num_workers, batch_size = 5, shuffle = True, drop_last = True)
loader = cycle(dataloader)
for i, img in enumerate(loader):
    print(i, img[0].shape)