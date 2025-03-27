import torch.utils.data as data
import torch
import random
import numpy as np

from .middlebury_dataset import MiddleburyDataset
from .middlebury2021_dataset import Middlebury2021Dataset
from .kittistereo_dataset import KITTIStereoDataset
from .booster_dataset import BoosterDataset
from .layeredflow_dataset import LayeredFlowDataset

def worker_init_fn(worker_id):                                                          
    torch_seed = torch.randint(0, 2**30, (1,)).item()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2**30
    np.random.seed(torch_seed + worker_id)

def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    if args.dataset in ['kitti_stereo', 'kitti2015', 'kitti2012']:
        if args.test:
            dataset = KITTIStereoDataset(args.datapath,test=True,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=args.numworkers, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise NotImplementedError

    elif args.dataset == 'middlebury2021':
        if args.test:
            dataset = Middlebury2021Dataset(args.datapath,test=True,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=args.numworkers, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise NotImplementedError
    
    elif args.dataset in ['middlebury', 'eth3d']:
        # _mono_model = args.monomodel #if args.monomodel == 'DAv2' else None
        datapaths = args.datapath.split(";")
        if args.test:
            dataset = MiddleburyDataset(datapaths[0],test=True,overfit=args.overfit,mono=None)
            for i in range(1,len(datapaths)):
                dataset += MiddleburyDataset(datapaths[i],test=True,overfit=args.overfit,mono=None)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=args.numworkers, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise NotImplementedError

    elif 'booster' in args.dataset:
        # _mono_model = args.monomodel #if args.monomodel == 'DAv2' else None        
        if args.test:
            dataset = BoosterDataset(args.datapath,test=True,overfit=args.overfit,mono=None)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=args.numworkers, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise NotImplementedError

    elif args.dataset == 'layeredflow':
        # _mono_model = args.monomodel
        if args.test:
            dataset = LayeredFlowDataset(args.datapath,test=True,overfit=args.overfit)
            loader = data.DataLoader(dataset, batch_size=args.batch_size, 
                pin_memory=False, shuffle=False, num_workers=args.numworkers, drop_last=True)
            print('Testing with %d image pairs' % len(dataset))
        else:
            raise NotImplementedError

    return loader
            