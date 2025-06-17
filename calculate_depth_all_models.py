
# %%
import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
import time
import cv2
import matplotlib.pyplot as plt
import glob
from icecream import ic
import pdb
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation, AutoModelForDepthEstimation, DPTImageProcessor, ZoeDepthImageProcessor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers.models.zoedepth.modeling_zoedepth import ZoeDepthDepthEstimatorOutput
from typing import Union, List, Tuple, Dict
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path_input', type=str, default="/mnt/storage/jorge/SISTEMAS/sample_files")
parser.add_argument('--path_output', type=str, default="/mnt/storage/jorge/depth_results")
parser.add_argument('--model_id', type=str, default="DAv2", choices=["DAv2", "Metric3D",
    "ZoeDepth", "Patchfusion", "ZoeDepthHuggingface"])
parser.add_argument('--get_depth_prediction_thresholded', type=bool, default=False)
parser.add_argument('--save_npy_flag', type=bool, default=True)
parser.add_argument('--pretrain_mode', type=str, default="indoor", choices=["indoor", "outdoor", "combined"])
parser.add_argument('--patchfusion_mode', type=str, default="DA", choices=["DA", "ZoeDepth"])

args = parser.parse_args()

print(args)
# %%
# !pip install -U transformers

# %%
def post_process_depth_estimation_zoedepth(
    outputs: ZoeDepthDepthEstimatorOutput,
    source_sizes: Union[torch.Tensor, List[Tuple[int, int]]],
    target_sizes: Union[torch.Tensor, List[Tuple[int, int]]] = None,
    remove_padding: bool = True,
) -> List[Dict] :
    """
    Converts the raw output of [`ZoeDepthDepthEstimatorOutput`] into final depth predictions and depth PIL image.
    Only supports PyTorch.

    Args:
        outputs ([`ZoeDepthDepthEstimatorOutput`]):
            Raw outputs of the model.
        source_sizes (`torch.Tensor` or `List[Tuple[int, int]]`):
            Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the source size
            (height, width) of each image in the batch before preprocessing.
        target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
            Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
            (height, width) of each image in the batch. If left to None, predictions will not be resized.
        remove_padding (`bool`):
            By default ZoeDepth addes padding to fix the boundary artifacts in the output depth map, so we need
            remove this padding during post_processing. The parameter exists here in case the user changed the
            image preprocessing to not include padding.

    Returns:
        `List[Dict]`: A list of dictionaries, each dictionary containing the depth predictions and a depth PIL
        image as predicted by the model.
    """
    predicted_depth = outputs.predicted_depth

    if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
        raise ValueError(
            "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
        )

    if (source_sizes is None) or (len(predicted_depth) != len(source_sizes)):
        raise ValueError(
            "Make sure that you pass in as many source image sizes as the batch dimension of the logits"
        )

    # Zoe Depth model adds padding around the images to fix the boundary artifacts in the output depth map
    # The padding length is `int(np.sqrt(img_h/2) * fh)` for the height and similar for the width
    # fh (and fw respectively) are equal to '3' by default
    # Check [here](https://github.com/isl-org/ZoeDepth/blob/edb6daf45458569e24f50250ef1ed08c015f17a7/zoedepth/models/depth_model.py#L57)
    # for the original implementation.
    # In this section, we remove this padding to get the final depth image and depth prediction
    if isinstance(source_sizes, List):
        img_h = torch.Tensor([i[0] for i in source_sizes])
        img_w = torch.Tensor([i[1] for i in source_sizes])
    else:
        img_h, img_w = source_sizes.unbind(1)

    fh = fw = 3

    results = []
    for i, (d, s) in enumerate(zip(predicted_depth, source_sizes)):
        if remove_padding:
            pad_h = int(np.sqrt(s[0]/2) * fh)
            pad_w = int(np.sqrt(s[1]/2) * fw)
            d = F.interpolate(
                d.unsqueeze(0).unsqueeze(1), size=[s[0] + 2*pad_h, s[1] + 2*pad_w],
                mode="bicubic", align_corners=False
            )
    
            if pad_h > 0:
                d = d[:, :, pad_h:-pad_h, :]
            if pad_w > 0:
                d = d[:, :, :, pad_w:-pad_w]

        if target_sizes is not None:
            target_size = target_sizes[i]
            d = F.interpolate(d, size=target_size, mode="bicubic", align_corners=False)

        d = d.squeeze().cpu().numpy()
        pil = Image.fromarray((d * 255 / np.max(d)).astype("uint8"))
        results.append({"predicted_depth": d, "depth": pil})

    return results

# %% [markdown]
# ## Define models

# %%

# Define the base class
class DepthModel:
 
    def __init__(self, **config):
        self.config = config
        self.save_npy_flag = config['save_npy_flag']
        self.save_path = config['save_path']
        self.save_plt_flag = config['save_plt_flag']
        self.save_plt_path = config['save_plt_path']

        if self.save_npy_flag:
            os.makedirs(self.save_path, exist_ok=True)
        self.plot_flag = config['plot_flag']
        if self.save_plt_flag:
            os.makedirs(self.save_plt_path, exist_ok=True)
        if self.config['save_png_flag']:
            os.makedirs(self.config['save_png_path'], exist_ok=True)
        if self.config['get_depth_prediction_thresholded']:
            os.makedirs(self.config['save_thresholded_path'], exist_ok=True)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    def load_model(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def infer(self, image):
        raise NotImplementedError("Subclass must implement abstract method")
    def load_image(self, image):
        image = cv2.imread(filename)
        return image
    def save_npy(self, filename, image):
        if not self.save_npy_flag:
            return
        save_path = os.path.join(
            self.save_path,
            Path(filename).stem + ".npz"
        )
        # ic(save_path)
        # ic(image.shape)
        np.savez(save_path, image)

    def save_png(self, filename, image, extension=".png"):
        if not self.config['save_png_flag']:
            return
        save_path = os.path.join(
            self.config['save_png_path'],
            Path(filename).stem + extension
        )
        print("Saving to...",self.config['save_png_path'],
            Path(filename).stem + extension)
        cv2.imwrite(save_path, image * (255/20))

    def plot_results(self, image, depth_prediction, confidence=None,
        filename=None):

        """
        Plots the original image and depth prediction side by side.
        Optionally includes a confidence map if available.
        """

        if not self.plot_flag:
            return
        image = np.array(image)
        # ic(image.shape)
        # ic(depth_prediction.shape)
        # ic(np.min(depth_prediction), np.mean(depth_prediction), np.max(depth_prediction))

        fig, axes = plt.subplots(1, 3 if confidence is not None else 2, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct colors
        axes[0].axis('off')
        axes[0].set_title("Original Image")

        # Depth prediction
        im = axes[1].imshow(depth_prediction, cmap='inferno')
        axes[1].axis('off')
        axes[1].set_title("Depth Prediction")
        # fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar for depth prediction
        
        # Adjust layout to make axes tight
        plt.tight_layout()

        # Add colorbar for the third image
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.ax.set_ylabel('Difference', rotation=270, labelpad=15)

        # Adjust the main plot to make room for the colorbar
        plt.subplots_adjust(right=0.9)
        
        # Confidence map, if available
        if confidence is not None:
            axes[2].imshow(confidence)# , cmap='inferno')
            axes[2].axis('off')
            axes[2].set_title("Thresholded depth mask")

        if self.save_plt_flag:
            save_plt_path = os.path.join(
                self.save_plt_path,
                Path(filename).stem + ".png"
            )
            plt.savefig(save_plt_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()

    def apply_threshold(self, depth_prediction, threshold = 10):

        # depth_prediction_thresholded = (depth_prediction < threshold).astype(np.uint8) * 255
        depth_prediction_thresholded = np.zeros_like(depth_prediction)

        depth_prediction_thresholded[depth_prediction >= threshold] = 255
        depth_prediction_thresholded[depth_prediction < threshold] = 0
        # ic(np.min(depth_prediction_thresholded), np.mean(depth_prediction_thresholded),
        #   np.max(depth_prediction_thresholded))
        # ic(depth_prediction_thresholded.shape)
        # ic(threshold)        
        # ic(np.min(depth_prediction_thresholded), np.mean(depth_prediction_thresholded),
        #     np.max(depth_prediction_thresholded))
        # ic(np.unique(depth_prediction_thresholded, return_counts=True))
        return depth_prediction_thresholded
    def save_thresholded_mask(self, filename, depth_prediction_thresholded):
        save_path = os.path.join(
            self.config['save_thresholded_path'],
            Path(filename).stem + ".png"
        )
        cv2.imwrite(save_path, depth_prediction_thresholded)

import sys

# First, clone the DAv2 repository
# !git clone https://github.com/DepthAnything/Depth-Anything-V2
# Define the DAv2 class
class DAv2(DepthModel):
    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        # !git clone https://github.com/DepthAnything/Depth-Anything-V2
        print("os.chdir('Depth-Anything-V2/metric_depth')")
        os.chdir('Depth-Anything-V2/metric_depth')
        sys.path.append('/mnt/storage/jorge/calculate_depth/Depth-Anything-V2/metric_depth')
    def load_model(self):
        print("os.getcwd()",os.getcwd())
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
        }
        
        encoder = self.config['encoder']   # 'vitl' or 'vits', 'vitb'
        dataset = self.config['dataset']  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 80 if dataset == 'vkitti' else 20  # 20 for indoor model, 80 for outdoor model
        
        self.model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', 
            map_location='cpu'))
        self.model.eval()
    '''
    @torch.no_grad()
    def infer(self, raw_image):
        image, (h, w) = self.model.image2tensor(raw_image)
        image = image.to(self.DEVICE)
        ic(image.dtype)
        ic(next(self.model.parameters()).dtype)
        result = self.model(image).cpu().numpy()
        return result
    '''
    def infer(self, image):
        # print("np.min(image), np.mean(image), np.max(image)", np.min(image), np.mean(image), np.max(image))

        return self.model.infer_image(image)

# First, clone the Patchfusion repository
# !git clone https://github.com/zhyever/PatchFusion.git    
# Use export pythonpath as follows:
# export PYTHONPATH="${PYTHONPATH}:/path/to/the/folder/PatchFusion"
# export PYTHONPATH="${PYTHONPATH}:/path/to/the/folder/PatchFusion/external"
# Example:
# export PYTHONPATH="${PYTHONPATH}:/mnt/storage/jorge/calculate_depth/PatchFusion"
# export PYTHONPATH="${PYTHONPATH}:/mnt/storage/jorge/calculate_depth/PatchFusion/external"
# Define the Patchfusion class
class Patchfusion(DepthModel):
    def __init__(self, **config):     
        super().__init__(**config)
        # self.backbone = "DA" # DA or Zoedepth 
        self.config = config

        # First, clone the Patchfusion repository
        # !git clone https://github.com/zhyever/PatchFusion.git        
        os.chdir('PatchFusion')
        sys.path.append("external")
        # sys.path.append('/mnt/storage/jorge/calculate_depth/PatchFusion')
        # sys.path.append('/mnt/storage/jorge/calculate_depth/PatchFusion/external/zoedepth')
        print("os.getcwd()", os.getcwd())
        print("os.listdir()", os.listdir())

    def load_model(self):
        from estimator.models.patchfusion import PatchFusion

        # Valid model names:
        # 'Zhyever/patchfusion_depth_anything_vits14', 
        # 'Zhyever/patchfusion_depth_anything_vitb14', 
        # 'Zhyever/patchfusion_depth_anything_vitl14', 
        # 'Zhyever/patchfusion_zoedepth'
        if self.config['backbone'] == "DA":
            model_name = 'Zhyever/patchfusion_depth_anything_vitl14'
        elif self.config['backbone'] == "ZoeDepth":
            model_name = 'Zhyever/patchfusion_zoedepth'
        print("model_name", model_name)
        self.model = PatchFusion.from_pretrained(model_name).to(self.DEVICE).eval()

        self.image_raw_shape = self.config['image_raw_shape']  # use a customized value instead of default one in model.tile_cfg['image_raw_shape']
        self.image_resizer = self.model.resizer
        self.mode = self.config['mode'] # 'r128'  # use fewer patches
        self.process_num = 4
        self.tile_cfg = dict()
        self.tile_cfg['image_raw_shape'] = self.image_raw_shape
        self.tile_cfg['patch_split_num'] = self.config['patch_split_num']# [4, 4]  # use customized value instead of default [4, 4] for 4K images

    def infer(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        # print("image.shape", image.shape)
        image_tensor = transforms.ToTensor()(np.asarray(image))
        image_lr = self.image_resizer(image_tensor.unsqueeze(dim=0)).float().to(self.DEVICE)
        image_hr = F.interpolate(image_tensor.unsqueeze(dim=0), self.image_raw_shape, mode='bicubic', align_corners=True).float().to(self.DEVICE)
        
        depth_prediction, _ = self.model(
            mode='infer', cai_mode=self.mode, process_num=self.process_num,
            image_lr=image_lr, image_hr=image_hr, tile_cfg=self.tile_cfg
        )
        # print("depth_prediction.shape", depth_prediction.shape)
        return F.interpolate(depth_prediction, image_tensor.shape[-2:])[0, 0].detach().cpu().numpy()

# Define the Metric3D class
class Metric3D(DepthModel):
    def __init__(self, **config):
        super().__init__(**config)
        self.mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        self.std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    def load_model(self):
        self.model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_large', pretrain=True).cpu()

    def infer(self, image):
        im = np.transpose(image, (2, 0, 1))  # Transpose to (C, H, W)
        im = np.expand_dims(im, axis=0)  # Add batch dimension
        im = torch.from_numpy(im).float().cpu()

        # Normalize the image
        im = torch.div((im - self.mean), self.std)

        # Perform inference
        pred_depth, confidence, _ = self.model.inference({'input': im})
        pred_depth = pred_depth.squeeze().detach().cpu().numpy()
        confidence = confidence.squeeze().detach().cpu().numpy()

        # return pred_depth, confidence
        return pred_depth

# ZoeDepthHuggingface class inheriting from DepthModel
class ZoeDepthHuggingface(DepthModel):
    def __init__(self, **config):
        """
        Initialize the model and image processor from the Huggingface Transformers library.
        """
        super().__init__(**config)

        self.config = config

        if self.config['pretrained_dataset'] == "NYUv2":
            self.pretrained_id = "Intel/zoedepth-nyu"
        elif self.config['pretrained_dataset'] == "Kitti":
            self.pretrained_id = "Intel/zoedepth-kitti"
        elif self.config['pretrained_dataset'] == "NYUv2_Kitti":
            self.pretrained_id = "Intel/zoedepth-nyu-kitti"
        ic(self.pretrained_id)
        # self.image_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-kitti")
        # self.image_processor = ZoeDepthImageProcessor(do_resize=False).from_pretrained("Intel/zoedepth-nyu-kitti", size=None)
        self.image_processor = ZoeDepthImageProcessor(do_resize=False).from_pretrained(self.pretrained_id, size=None)

        self.DEVICE = self.get_device()


    def load_model(self):   
        self.model = ZoeDepthForDepthEstimation.from_pretrained(self.pretrained_id).to(self.get_device())        
    def get_device(self):
        """
        Determine if a GPU is available, otherwise use CPU.
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def infer(self, image):
        """
        Perform inference using ZoeDepth from Huggingface and return the depth prediction.
        """
        # Convert the OpenCV image (BGR) to PIL Image and preprocess
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


        inputs = self.image_processor(images=image_pil, 
            return_tensors="pt").to(self.DEVICE)
        # ic(inputs['pixel_values'].shape)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            # predicted_depth = outputs.predicted_depth
        
        depth_prediction = post_process_depth_estimation_zoedepth(outputs, 
            [image_pil.size[::-1]])[0]['predicted_depth']

        # ic(depth_prediction.shape)
        '''
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],  # Image size in (H, W)
            mode="bicubic",
            align_corners=False
        )
        '''

        return depth_prediction

# DepthAnythingV1 class inheriting from DepthModel
class DepthAnythingV1(DepthModel):
    def __init__(self, **config):
        """
        Initialize the DepthAnythingV1 model and image processor from Huggingface Transformers.
        """
        super().__init__(**config)
        self.image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        # self.image_processor = DPTImageProcessor(do_resize=False).from_pretrained("LiheYoung/depth-anything-large-hf")

        self.DEVICE = self.get_device()

    def load_model(self):
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(self.get_device())

    def get_device(self):
        """
        Determine if a GPU is available, otherwise use CPU.
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def infer(self, image):
        """
        Perform inference using DepthAnythingV1 and return the depth prediction.
        """
        # Convert the OpenCV image (BGR) to PIL Image
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Preprocess image
        inputs = self.image_processor(images=image_pil, return_tensors="pt").to(self.DEVICE)


        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        '''
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1],  # Image size in (H, W)
            mode="bicubic",
            align_corners=False
        )
        '''
        # Convert prediction to numpy for plotting
        depth_prediction = predicted_depth.squeeze().cpu().numpy()

        return depth_prediction

class ZoeDepth(DepthModel):
#    def __init__(self, from_local=False):
    def __init__(self, **config):
        super().__init__(**config)
        # !git clone https://github.com/isl-org/ZoeDepth.git
        os.chdir('ZoeDepth')
        self.config = config
    def load_image(self, filename):
        image = Image.open(filename)
        return image        
    def load_model(self):

        # ZoeD_K
        ic(os.getcwd())
        sys.path.append('/mnt/storage/jorge/calculate_depth/ZoeDepth')
        from zoedepth.models.builder import build_model
        from zoedepth.utils.config import get_config
        conf = get_config("zoedepth", "infer", config_version="kitti") # "nyu", "kitti", "mix"
        conf["input_height"]= self.config["input_height"]
        conf["input_width"] =  self.config["input_width"]
        self.model = build_model(conf)

        
        # model_id = "ZoeD_N" if self.config['pretrained_dataset'] == "NYUv2" else "ZoeD_K"
        # self.model = torch.hub.load(".", model_id, source="local", pretrained=True)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.DEVICE)
    def infer(self, image):
        # rgb_to_bgr = transforms.Lambda(lambda x: x[[2, 1, 0], :, :])

        transform = transforms.Compose([
        transforms.ToTensor()# ,
        # rgb_to_bgr
        ])

        # Apply the transformation to the image
        tensor_image = transform(image).unsqueeze(0).to(self.DEVICE)

        # ic(tensor_image.shape)

        depth_numpy = self.model(tensor_image)['metric_depth'].squeeze().detach().cpu().numpy()  # as numpy
        return depth_numpy




# %% [markdown]
# ## Select model

# %%

# Select model
models = {
    "DAv2": DAv2,
    "Patchfusion": Patchfusion,
    "Metric3D": Metric3D,
    "ZoeDepth": ZoeDepth,
    "ZoeDepthHuggingface": ZoeDepthHuggingface,
    "DepthAnythingV1": DepthAnythingV1
}

selected_model_id = args.model_id # "DAv2" "Metric3D" "ZoeDepth" "Patchfusion" "ZoeDepthHuggingface"


# %% [markdown]
# ## Set configuration

# %%


if selected_model_id == "Patchfusion":
    config = {
        "backbone": args.patchfusion_mode, # "DA", "ZoeDepth"
        "image_raw_shape": [1536, 1536],# [1024, 1024], # [1536, 1536], # [3360, 6720],
        "mode": 'r128',
        "patch_split_num": [4, 4],
    }
elif selected_model_id == "ZoeDepth":
    if args.pretrain_mode == "indoor":
        pretrained_dataset = "NYUv2"
    else:
        pretrained_dataset = "Kitti"
    config = {
        "pretrained_dataset": pretrained_dataset, # NYUv2, Kitti,
        "input_height": 1536,
        "input_width": 1536
    }
elif selected_model_id == "DAv2":
    if args.pretrain_mode == "indoor":
        pretrained_dataset = "hypersim"
        max_depth = 20
    else:
        pretrained_dataset = "vkitti"
        max_depth = 80
    config = {
        "encoder": 'vitl',  # or 'vits', 'vitb'
        "dataset": pretrained_dataset,  # 'hypersim' for indoor model, 'vkitti' for outdoor model
        "max_depth": max_depth  # 20 for indoor model, 80 for outdoor model
    }
elif selected_model_id == "ZoeDepthHuggingface":
    if args.pretrain_mode == "indoor":
        pretrained_dataset = "NYUv2"
    elif args.pretrain_mode == "outdoor":
        pretrained_dataset = "Kitti"
    elif args.pretrain_mode == "combined":
        pretrained_dataset = "NYUv2_Kitti"
    config = {
        "pretrained_dataset": pretrained_dataset # or NYUv2, Kitti, NYUv2_Kitti
    }
else:
    config = {}

base_save_path = os.path.join(args.path_output, f"results_{selected_model_id}")
os.makedirs(base_save_path, exist_ok=True)

config.update({
    'save_npy_flag': args.save_npy_flag,
    'save_plt_flag': False,
    'save_png_flag': False,
    'plot_flag': False,
    'get_depth_prediction_thresholded': args.get_depth_prediction_thresholded,
    'save_path': f"{base_save_path}/faces_depth",
    'save_plt_path': f"{base_save_path}/faces_depth",
    'save_thresholded_path': f"{base_save_path}/faces_depth_thresholded",
    'save_png_path': f"{base_save_path}/faces_depth_png"

})



print("==================================")
print(f"selected_model_id: {selected_model_id}")

# Instantiate the selected model
selected_model = models[selected_model_id](**config)  
selected_model.load_model()

# %% [markdown]


path_input = args.path_input

input_extension = "png"

# %%
glob_path = os.path.join(path_input, "*.png")


# %%
len(glob.glob(os.path.join(path_input, "*.png")))

# %% [markdown]
# ## Do inference

# %%
import tqdm
for idx, filename in enumerate(tqdm.tqdm(glob.glob(os.path.join(path_input, f"*.{input_extension}")))):

    # if idx > 4:
    #     continue
    ## ic(filename)
    image = selected_model.load_image(filename)

    depth_prediction = selected_model.infer(image)
    # print("np.min(depth_prediction), np.mean(depth_prediction), np.max(depth_prediction)", np.min(depth_prediction), np.mean(depth_prediction), np.max(depth_prediction))

    # ic(np.min(depth_prediction),np.mean(depth_prediction),np.max(depth_prediction))
    if config['get_depth_prediction_thresholded']:
        depth_prediction_thresholded = selected_model.apply_threshold(depth_prediction)
        # if idx % 5 == 0:
        if True:
            selected_model.plot_results(image, depth_prediction, 
                depth_prediction_thresholded, filename=filename)

        selected_model.save_thresholded_mask(filename, depth_prediction_thresholded)
        selected_model.save_npy(filename, depth_prediction)
    else:
        if idx <= 4:
            selected_model.plot_results(image, depth_prediction, filename=filename)
    
    selected_model.save_npy(filename, depth_prediction)
    ## ic(selected_model.config['save_png_flag'])
    selected_model.save_png(filename, depth_prediction)

    # pdb.set_trace()


