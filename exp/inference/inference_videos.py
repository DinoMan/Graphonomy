# Standard libraries
import timeit
from pathlib import Path
from glob import glob
from datetime import datetime
import os
import sys
from collections import OrderedDict
import argparse
from typing import List
sys.path.append('./')

# PyTorch
import torch
torch.set_num_threads(1)
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
torch.set_grad_enabled(False)

# Other third-party libraries
import numpy as np
from PIL import Image
import cv2
cv2.setNumThreads(0)
from tqdm import tqdm
import decord
from math import ceil, floor
import dtk.transforms as dtf
import dtk.nn as dnn
import progressbar


# Custom imports
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


def flip_cihp(tail_list):
    """
        Swap channels in a probability map so that "left foot" becomes "right foot" etc.

        tail_list: (B x n_class x h x w)
    """
    return torch.cat((
        tail_list[:, :14],
        tail_list[:, 14:15],
        tail_list[:, 15:16],
        tail_list[:, 17:18],
        tail_list[:, 16:17],
        tail_list[:, 19:20],
        tail_list[:, 18:19]), dim=1)

def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def img_transform(img, transform):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample['image']

if __name__ == '__main__':
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=Path,
        help="Where the model weights are.")
    parser.add_argument('--video_path', required=True, type=Path,
        help="Where to look for images. Can be a file with a list of paths, or a " \
             "directory (will be searched recursively for png/jpg/jpeg files).")
    parser.add_argument('--output_dir', required=True, type=Path,
        help="A directory where to save the results. Will be created if doesn't exist.")
    parser.add_argument('--common_prefix', type=Path,
        help="Common prefix relative to which save the output files.")
    parser.add_argument('--tta', default='1,0.75,0.5,1.25,1.5,1.75', type=str,
        help="A list of scales for test-time augmentation.")
    parser.add_argument('--save_extra_data', action='store_true',
        help="Save parts' segmentation masks, colored segmentation masks and images with removed background.")
    parser.add_argument('--max_frames', type=int, help="The max frames to process")
    parser.add_argument('--split_len', type=int, default=20, help="The max length of a video that fits onto the GPU")
    parser.add_argument('--multiprocess', nargs='+', type=int, default=(1,1))
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )
    net.load_source_model(torch.load(opts.model_path))
    net.cuda()
    dnn.freeze(net)

    
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    net.eval()

    video_paths_list: List[Path]
    
    common_prefix = None
    if opts.video_path.is_file():
        print(f"`--video_path` ({opts.video_path}) is a file, reading it for a list of files...")
        with open(opts.video_path, 'r') as f:
            video_paths_list = sorted(Path(line.strip()) for line in f)

        common_prefix= opts.common_prefix
    elif opts.video_path.is_dir():
        print(f"`--video_path` ({opts.video_path}) is a directory, recursively looking for images in it...")
        
        def list_files_recursively(path, allowed_extensions):
            retval = []
            for child in path.iterdir():
                if child.is_dir():
                    retval += list_files_recursively(child, allowed_extensions)
                elif child.suffix.lower() in allowed_extensions:
                    retval.append(child)

            return retval

        video_paths_list = sorted(list_files_recursively(opts.video_path, ('.mp4')))

    else:
        raise FileNotFoundError(f"`--video_path` ('{opts.video_path}')")

    print(f"total files {len(video_paths_list)}")
    part_start = int(floor(((opts.multiprocess[0]-1)/ opts.multiprocess[1])*len(video_paths_list)))
    part_end = int(ceil((opts.multiprocess[0]/ opts.multiprocess[1])*len(video_paths_list)))
    print(f"processing part {opts.multiprocess[0]} of {opts.multiprocess[1]}")
    print(f"part start{part_start} part end {part_end}")


    video_paths_list = video_paths_list[part_start:part_end]
    print(f"Found {len(video_paths_list)} images")
    print(f"Will output files in {opts.output_dir}")
    print(f"Example:")
    print(f"The segmentation for: {video_paths_list[0]}")
    
    tta = opts.tta
    try:
        tta = tta.split(',')
        tta = list(map(float, tta))
    except:
        raise ValueError(f'tta must be a sequence of comma-separated float values such as "1.0,0.5,1.5". Got "{opts.tta}".')

    scale_list = tta
    # 1.0 should always go first
    try:
        scale_list.remove(1.0)
    except ValueError:
        pass
    scale_list.insert(0, 1.0)

    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, vid_paths, scale_list, prefix=None, max_frames=None):
            self.vid_paths = vid_paths
            self.scale_list = scale_list
            decord.bridge.set_bridge('torch')
            vr = decord.VideoReader(str(os.path.join(prefix, self.vid_paths[0])))
            self.fps = ceil(vr.get_avg_fps())
            self.dataset_size = vr.next().shape
            self.tf = transforms.Compose([dtf.ToTensorVideo(), dtf.NormalizeVideo((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            self.horizontal_flip_tf = dtf.RandomHorizontalFlipVideo(p=1)
            self.prefix = prefix
            self.max_frames = max_frames

        def __len__(self):
            return len(self.vid_paths)

        def __getitem__(self, idx):
            video_path = self.vid_paths[idx]
            if self.prefix is not None:
                in_video_path = os.path.join(self.prefix, video_path)
            else:
                in_video_path = video_path                
            vr = decord.VideoReader(str(in_video_path), height=256, width=256)
            permute = [2, 1, 0]
            max_frames=len(vr) if self.max_frames is None else min(self.max_frames,len(vr))

            video = self.tf(vr.get_batch(range(0, max_frames)))[:, permute]
            
            
            original_size = torch.tensor(self.dataset_size[:2]) # to make `default_collate` happy
            video_flipped = self.horizontal_flip_tf(video)

            # `str()` because `default_collate` doesn't like `Path`
            return video, video_flipped, str(video_path), original_size

    dataset = InferenceDataset(video_paths_list, scale_list, prefix=common_prefix, max_frames=opts.max_frames)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)

    bar = progressbar.ProgressBar(max_value=len(dataset))
    for sample_idx, (videos, videos_flipped, video_paths, original_sizes) in enumerate(dataloader):
        
        video_length = videos.size(1)
        current_frame = 0
        mask = np.zeros((video_length, 256, 256))
        original_sizes = [tuple(original_size.tolist()) for original_size in original_sizes]

        while current_frame < video_length:
            inp_video = videos[0, current_frame:(current_frame + opts.split_len)]
            inp_video_flipped = videos_flipped[0, current_frame:(current_frame + opts.split_len)]
            snippet_len = inp_video.size(0)

            inputs = torch.cat((inp_video, inp_video_flipped)).cuda()
            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda()).squeeze()
            outputs_final = (outputs[:snippet_len] + torch.flip(flip_cihp(outputs[snippet_len:]), dims=[-1,])) / 2

            background_probability = 1.0 - outputs_final.softmax(1)[:, 0] # `B x H x W`
            mask[current_frame:(current_frame + snippet_len)] = (background_probability * 255).round().byte().cpu().numpy()
            current_frame += snippet_len
        
        output_video_path = opts.output_dir / video_paths[0]

        directory = os.path.dirname(output_video_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        writer = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), float(dataset.fps), original_sizes[0], isColor=0)
        for i, frame in enumerate(mask):
            write_frame = cv2.resize(frame.astype('uint8'), original_sizes[0])
            writer.write(write_frame)
    
        writer.release()   
        bar.update(sample_idx)
