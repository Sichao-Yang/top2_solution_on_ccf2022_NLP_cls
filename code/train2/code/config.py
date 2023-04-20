from os import path as osp
import toml
from easydict import EasyDict as edict
join = lambda x,y: osp.abspath(osp.join(x, y))

root = join(osp.dirname(__file__), '..')

def parse_args():
    args = toml.load(join(root, 'code/cfg.toml'))
    for k,v in args.items():
        if 'dir' in k or 'filepath' in k:
            args[k] = join(root, v)
    args['root'] = root
    return edict(args)

if __name__=='__main__':
    args = parse_args()
    print(args)
    print(args.seed)
    print(args.root)
