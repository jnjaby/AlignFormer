import argparse
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import numpy as np
import pyiqa
import torch
import cv2
import mmcv
import os

def get_imglist(root):
    img_list = [os.path.join(root, img) for img in sorted(os.listdir(root))]
    return img_list

def imread(path):
    img = mmcv.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
    return img.astype(np.float32)

def img2tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1))[None, ...]

def tensor2img(tensor):
    if tensor.dim() == 4:
        return tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    elif tensor.dim() == 3:
        return tensor.squeeze().cpu().numpy()
    else:
        raise ValueError('Dimension of tensor should be 3 or 4.')



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--id', type=str, help='File id')
    # parser.add_argument('--output', type=str, help='Save path')
    # args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_dict = {
        'L1': '../results/FGTransformer/063_test30_real_200k/visualization/test/',
        'CX': '../results/FGTransformer/124_test30_real_150k/visualization/test/',
        'syn': '../results/FGTransformer/036_test30_real_500k/visualization/syn_pair_test/',
        'Ours': '../results/FGTransformer/061_test30_real_500k//visualization/test/',
    }

    metrics = ['niqe', 'nrqm', 'musiq-koniq']
    metric_dict = OrderedDict()

    for metric in metrics:
        iqa_metric = pyiqa.create_metric(metric).to(device)
        metric_dict[metric] = defaultdict(list)
        
        for model_name, root_path in model_dict.items():
            imgs = get_imglist(root_path)
            suffix = root_path.split('/')[2]
            assert len(imgs) == 450

            for img_path in tqdm(imgs):
                img_name = os.path.basename(img_path)
                rst_img = imread(img_path)

                try:
                    metric_value = iqa_metric(img2tensor(rst_img).to(device)).item()
                    metric_dict[metric][model_name].append(metric_value)
                except:
                    print(f'Cannot calc {img_name} on {metric}. Drop it.')
                    continue

            metric_dict[metric][model_name] = np.array(metric_dict[metric][model_name])

        print(metric + ':')
        for k, v in metric_dict[metric].items():
            print(k, np.mean(v))
