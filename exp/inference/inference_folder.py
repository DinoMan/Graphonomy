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
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data
from torchvision import transforms
torch.set_grad_enabled(False)

# Other third-party libraries
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

# Custom imports
from networks import deeplab_xception_transfer, graph
from dataloaders import custom_transforms as tr

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


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

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample['image']

def inference(net, img_paths, output_path, scale_list=[1.0, 0.5, 0.75, 1.25, 1.5, 1.75], use_gpu=True, save_extra_data=False):
    # 1.0 should always go first
    try:
        scale_list.remove(1.0)
    except ValueError:
        pass
    scale_list.insert(0, 1.0)

    # Compute the longest common prefix to determine output paths
    common_prefix = os.path.commonpath(img_paths)

    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # One testing epoch
    net.eval()

    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, img_paths, scale_list):
            self.img_paths = img_paths
            self.scale_list = scale_list

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            image_path = self.img_paths[idx]
            img = read_img(image_path)
            img_flipped = img_transform(img, tr.HorizontalFlip_only_img())

            retval, retval_flipped = [], []
            for scale in self.scale_list:
                transform = transforms.Compose([
                    tr.Scale_only_img(scale),
                    tr.Normalize_xception_tf_only_img(),
                    tr.ToTensor_only_img()])

                retval.append(img_transform(img, transform))
                retval_flipped.append(img_transform(img_flipped, transform))

            return retval, retval_flipped, str(image_path) # because `default_collate` doesn't like `Path`

    dataset = InferenceDataset(img_paths, scale_list)
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=1)

    exec_times = []
    for images, images_flipped, image_path in tqdm(dataloader):
        image_path = Path(image_path[0])

        start_time = timeit.default_timer()

        for iii, (image, image_flipped) in enumerate(zip(images, images_flipped)):
            inputs = torch.cat((image, image_flipped))
            if iii == 0:
                _, _, h, w = inputs.shape

            if use_gpu:
                inputs = inputs.cuda()

            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if iii > 0:
                outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs

        # outputs_final: `B x 20 x H x W`
        end_time = timeit.default_timer()
        exec_times.append(end_time - start_time)

        # Actually write the outputs to disk
        image_path = image_path.relative_to(common_prefix)

        if save_extra_data:
            predictions = torch.max(outputs_final, 1)[1]
            results = predictions.cpu().numpy()

            for output_folder in 'mask_gray', 'mask_color', 'segmented':
                (output_path / output_folder / image_path.parent).mkdir(parents=True, exist_ok=True)

            # saving grayscale mask image
            cv2.imwrite(str(output_path / 'mask_gray' / image_path.with_suffix('.png')), results[0, :, :])

            # saving colored mask image
            vis_res = decode_labels(results)
            parsing_im = Image.fromarray(vis_res[0])
            parsing_im.save(str(output_path / 'mask_color' / image_path.with_suffix('.png')))

            # saving segmented image with masked pixels drawn black
            segmented_img = np.asarray(images[0][0] * 0.5 + 0.5) * (results[0, :, :] > 0).astype(np.float)[np.newaxis]
            cv2.imwrite(str(output_path / 'segmented' / image_path.with_suffix('.png')), segmented_img.transpose(1,2,0) * 255)
        else:
            background_probability = 1.0 - outputs_final.softmax(1)[:, 0] # `B x H x W`
            background_probability = (background_probability * 255).round().byte().cpu().numpy()

            (output_path / image_path.parent).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path / image_path.with_suffix('.png')), background_probability[0, :, :])

    print('Average inference time:', np.mean(exec_times))


if __name__ == '__main__':
    '''argparse begin'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, type=Path,
        help="Where the model weights are.")
    parser.add_argument('--images_path', required=True, type=Path,
        help="Where to look for images. Can be a file with a list of paths, or a " \
             "directory (will be searched recursively for png/jpg/jpeg files).")
    parser.add_argument('--output_dir', required=True, type=Path,
        help="A directory where to save the results. Will be created if doesn't exist.")
    parser.add_argument('--tta', default='1,0.75,0.5,1.25,1.5,1.75', type=str,
        help="A list of scales for test-time augmentation.")
    parser.add_argument('--save_extra_data', action='store_true',
        help="Save parts' segmentation masks, colored segmentation masks and images with removed background.")
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )

    net.load_source_model(torch.load(opts.model_path))
    net.cuda()

    image_paths_list: List[Path]
    
    if opts.images_path.is_file():
        print(f"`--images_path` is a file, reading it for a list of files...")
        with open(opts.images_path, 'r') as f:
            image_paths_list = sorted(Path(line.strip()) for line in f)
    elif opts.images_path.is_dir():
        print(f"`--images_path` is a directory, recursively looking for images in it...")
        image_paths_list = sorted(
            x for x in opts.images_path.rglob("*") \
            if x.is_file() and x.suffix.lower() in ('.png', '.jpg', '.jpeg')
        )
    else:
        raise FileNotFoundError(f"`--images_path` ('{opts.images_path}')")

    print(f"Found {len(image_paths_list)} images")
    
    tta = opts.tta
    try:
        tta = tta.split(',')
        tta = list(map(float, tta))
    except:
        raise ValueError(f'tta must be a sequence of comma-separated float values such as "1.0,0.5,1.5". Got "{opts.tta}".')

    inference(net, image_paths_list,  opts.output_dir, use_gpu=True, scale_list=tta, save_extra_data=opts.save_extra_data)
