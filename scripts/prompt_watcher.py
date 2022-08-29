import argparse, os, sys, glob
from datetime import datetime
import cv2
import torch
import random
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from prompt.notion import Notion 
from prompt.artists import sanitize_for_filename

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

class StableSettings:
    def __init__(self):
        self.prompt = "a painting of a virus monster playing guitar"
        self.outdir = "output/artist_study"
        self.sampledir = "output/artist_study_samples"
        self.skip_grid = False
        self.skip_save = False
        self.ddim_steps = 50
        self.plms = False 
        self.laion400m = False 
        self.fixed_code = False
        self.n_iter = 1
        self.H = 512
        self.W = 512
        self.C = 4 # latent channels
        self.f = 8 # downsampling factor
        self.n_samples = 1
        self.n_rows = 0
        self.scale = 7.5
        self.ddim_eta = 0.0
        self.config = "configs/stable-diffusion/v1-inference.yaml"
        self.ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"
        self.seed = 42
        self.precision = "autocast" # ["full", "autocast"]
        self.from_file = False
        self.modelid = 'sd1p4'
class Stable:
    def __init__(self, params):

        # load safety model
        safety_model_id = "CompVis/stable-diffusion-safety-checker"
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

        self.opt = params
        print (f"Starting prompt: {params.prompt}")
        
        seed_everything(self.opt.seed)

        config = OmegaConf.load(f"{self.opt.config}")
        self.model = self.load_model_from_config(config, f"{self.opt.ckpt}")

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(self.device)

        if self.opt.plms:
            self.sampler = PLMSSampler(self.model)
        else:
            self.sampler = DDIMSampler(self.model)

        os.makedirs(self.opt.outdir, exist_ok=True)
        self.outpath = self.opt.outdir

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())


    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images


    def load_model_from_config(self, config, ckpt, verbose=False):
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.cuda()
        model.eval()
        return model


    def put_watermark(self, img, wm_encoder=None):
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img


    def load_replacement(self, x):
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y)/255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x


    def check_safety(self, x_image):
        safety_checker_input = self.safety_feature_extractor(self.numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = self.load_replacement(x_checked_image[i])
        return x_checked_image, has_nsfw_concept



    def generate(self, params):
        
        images = []
        self.opt = params
        seed_everything(self.opt.seed)
        
        batch_size = self.opt.n_samples
        n_rows = self.opt.n_rows if self.opt.n_rows > 0 else batch_size
        if not self.opt.from_file:
            prompt = self.opt.prompt
            assert prompt is not None
            data = [batch_size * [prompt]]

        else:
            print(f"reading prompts from {self.opt.from_file}")
            with open(self.opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = list(self.chunk(data, batch_size))

        sample_path = os.path.join(self.outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        grid_count = len(os.listdir(self.outpath)) - 1

        start_code = None
        if self.opt.fixed_code:
            start_code = torch.randn([self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f], device=self.device)

        precision_scope = autocast if self.opt.precision=="autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(self.opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if self.opt.scale != 1.0:
                                uc = self.model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = self.model.get_learned_conditioning(prompts)
                            shape = [self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f]
                            samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                            conditioning=c,
                                                            batch_size=self.opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=self.opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=self.opt.ddim_eta,
                                                            x_T=start_code)

                            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                            x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                            if not self.opt.skip_save:
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    images.append(img)
                                    base_count += 1
                    toc = time.time()

        print(f"Your samples are ready and waiting for you here: \n{self.outpath} \n"
            f" \nEnjoy.")
        return images
import numbers
def main():
    ndb = Notion("config.yaml")
    params = StableSettings()
    params.outdir = "/home/dsedov/Dropbox/sd_prompt_watcher/"

    stable = Stable(params)

    os.makedirs(params.outdir, exist_ok=True)

    while(True):
        prompts = ndb.queued_prompts()
        for prompt in prompts:
            print(type(prompt["iterations"]))
            if isinstance(prompt["iterations"], numbers.Number):
                if prompt["iterations"] > 0:
                    params.prompt = prompt["prompt"]
                    for i in range(prompt["iterations"]):
                        params.seed = random.randint(0, 2**32) 
                        images = stable.generate(params)
                        save_filename = sanitize_for_filename(f"{params.prompt}__S{params.seed}.png")
                        images[0].save(os.path.join(params.outdir, save_filename))
                ndb.mark_prompt_done(prompt)
        time.sleep(60)


if __name__ == "__main__":
    main()