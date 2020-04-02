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

def read_img(img_path):
    _img = Image.open(img_path).convert('RGB')  # return is RGB pic
    return _img

def img_transform(img, transform):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample['image']

def save_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

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
    parser.add_argument('--common_prefix', type=Path,
        help="Common prefix relative to which save the output files.")
    parser.add_argument('--tta', default='1,0.75,0.5,1.25,1.5,1.75', type=str,
        help="A list of scales for test-time augmentation.")
    parser.add_argument('--save_extra_data', action='store_true',
        help="Save parts' segmentation masks, colored segmentation masks and images with removed background.")
    opts = parser.parse_args()

    net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7, )

    # Initialize saver processes
    import multiprocessing

    class ConstrainedTaskPool:
        def __init__(self, num_processes=1, max_tasks=100):
            self.num_processes = num_processes
            self.task_queue = multiprocessing.Queue(maxsize=max_tasks)

        def __enter__(self):
            def worker_function(task_queue):
                for function, args in iter(task_queue.get, 'STOP'):
                    function(*args)

            for _ in range(self.num_processes):
                multiprocessing.Process(target=worker_function, args=(self.task_queue,)).start()

            return self

        def __exit__(self, *args):
            for _ in range(self.num_processes):
                self.task_queue.put('STOP')

        def put_async(self, function, *args):
            self.task_queue.put((function, args))

    with ConstrainedTaskPool(num_processes=4, max_tasks=6000) as background_saver:
        def save_image_async(image, path):
            background_saver.put_async(save_image, image, path)

        net.load_source_model(torch.load(opts.model_path))
        net.cuda()

        # adj
        adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
        adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

        adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
        adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

        cihp_adj = graph.preprocess_adj(graph.cihp_graph)
        adj3_ = Variable(torch.from_numpy(cihp_adj).float())
        adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

        net.eval()

        image_paths_list: List[Path]
        
        if opts.images_path.is_file():
            print(f"`--images_path` ({opts.images_path}) is a file, reading it for a list of files...")
            with open(opts.images_path, 'r') as f:
                image_paths_list = sorted(Path(line.strip()) for line in f)

            if opts.common_prefix is None:
                common_prefix = os.path.commonpath(image_paths_list)
            else:
                common_prefix= opts.common_prefix
                for path in image_paths_list:
                    # TODO optimize by using commonpath
                    assert common_prefix in path.parents
        elif opts.images_path.is_dir():
            print(f"`--images_path` ({opts.images_path}) is a directory, recursively looking for images in it...")
            
            def list_files_recursively(path, allowed_extensions):
                retval = []
                for child in path.iterdir():
                    if child.is_dir():
                        retval += list_files_recursively(child, allowed_extensions)
                    elif child.suffix.lower() in allowed_extensions:
                        retval.append(child)

                return retval

            image_paths_list = sorted(list_files_recursively(opts.images_path, ('.png', '.jpg', '.jpeg')))

            common_prefix = opts.images_path
        else:
            raise FileNotFoundError(f"`--images_path` ('{opts.images_path}')")

        print(f"Found {len(image_paths_list)} images")
        print(f"Will output files in {opts.output_dir} with names relative to {common_prefix}.")
        print(f"Example:")
        print(f"The segmentation for: {image_paths_list[0]}")
        print(f"Will be put in: {opts.output_dir / image_paths_list[0].relative_to(common_prefix).parent}")
        
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
            def __init__(self, img_paths, scale_list):
                self.img_paths = img_paths
                self.scale_list = scale_list

            def __len__(self):
                return len(self.img_paths)

            def __getitem__(self, idx):
                image_path = self.img_paths[idx]
                img = read_img(image_path)
                
                original_size = torch.tensor(img.size) # to make `default_collate` happy
                img = img.resize((256, 256))

                img_flipped = img_transform(img, tr.HorizontalFlip_only_img())

                retval, retval_flipped = [], []
                for scale in self.scale_list:
                    transform = transforms.Compose([
                        tr.Scale_only_img(scale),
                        tr.Normalize_xception_tf_only_img(),
                        tr.ToTensor_only_img()])

                    retval.append(img_transform(img, transform))
                    retval_flipped.append(img_transform(img_flipped, transform))

                # `str()` because `default_collate` doesn't like `Path`
                return retval, retval_flipped, str(image_path), original_size

        dataset = InferenceDataset(image_paths_list, scale_list)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=2)

        exec_times = []
        for sample_idx, (images, images_flipped, image_paths, original_sizes) in enumerate(dataloader):
            # `images`, `images_flipped`: list of length <number-of-scales>,
            #   each element is a tensor of shape (<batch-size> x 3 x H_k x W_k);
            # `image_paths`: tuple of length <batch-size> of str
            # `original_size`: int tensor of shape (<batch-size> x 2)
            if sample_idx % 10 == 0:
                import datetime
                print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {sample_idx} / {len(dataloader)}")

            original_sizes = [tuple(original_size.tolist()) for original_size in original_sizes]
            image_paths = [Path(x).relative_to(common_prefix) for x in image_paths]

            batch_size = len(images[0])

            start_time = timeit.default_timer()

            for iii, (image, image_flipped) in enumerate(zip(images, images_flipped)):
                inputs = torch.cat((image, image_flipped))
                if iii == 0:
                    _, _, h, w = inputs.shape

                inputs = inputs.cuda()

                outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
                outputs = (outputs[:batch_size] + torch.flip(flip_cihp(outputs[batch_size:]), dims=[-1,])) / 2

                if iii > 0:
                    outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                    outputs_final = outputs_final + outputs
                else:
                    outputs_final = outputs

            # outputs_final: `B x 20 x H x W`
            end_time = timeit.default_timer()
            exec_times.append(end_time - start_time)

            # Actually write the outputs to disk
            if opts.save_extra_data:
                predictions = torch.max(outputs_final, 1)[1]
                results = predictions.cpu().numpy()
                vis_results = decode_labels(results)

                for input_image_1xScale, vis_result, result, image_path \
                    in zip(images[0], vis_results, results, image_paths):

                    # saving grayscale mask image
                    save_image_async(result, opts.output_dir / 'mask_gray' / image_path.with_suffix('.png'))

                    # saving colored mask image
                    save_image_async(vis_result, opts.output_dir / 'mask_color' / image_path.with_suffix('.png'))

                    # saving segmented image with masked pixels drawn black
                    segmented_img = np.asarray(input_image_1xScale * 0.5 + 0.5) * (result > 0).astype(np.float)[np.newaxis]
                    save_image_async(
                        segmented_img.transpose(1,2,0) * 255,
                        opts.output_dir / 'segmented' / image_path.with_suffix('.png'))
            else:
                background_probability = 1.0 - outputs_final.softmax(1)[:, 0] # `B x H x W`
                background_probability = (background_probability * 255).round().byte().cpu().numpy()

                for background_probability_single_sample, image_path, original_size \
                    in zip(background_probability, image_paths, original_sizes):

                    output_image_path = opts.output_dir / image_path.with_suffix('.png')
                    save_image_async(
                        cv2.resize(background_probability_single_sample, original_size), output_image_path)

    print('Average inference time:', np.mean(exec_times))
