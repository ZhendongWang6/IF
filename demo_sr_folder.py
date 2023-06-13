import os
from glob import glob

from PIL import Image

from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
from deepfloyd_if.pipelines import super_resolution

device = "cuda:0"
# if_I = IFStageI('IF-I-XL-v1.0', device=device)
if_II = IFStageII("IF-II-L-v1.0", device=device)
# if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device="cpu")

prompts = {
    "general": "detailed picture, 4k dslr, best quality",
    "none": "",
}
prompt_choose = "general"
# open method used to open different extension image file
img_paths = glob("testsets/Set5/LR_bicubic/X4/*.png")
for img_path in img_paths:
    raw_pil_image = Image.open(img_path)
    output_path = img_path.replace("LR_bicubic", f"Deep_IF_{prompt_choose}").replace("testsets", "results")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    middle_res = super_resolution(
        t5,
        if_III=if_II,
        seed=42,
        prompt=[prompts[prompt_choose]],
        support_pil_img=raw_pil_image,
        img_scale=4.0,
        img_size=64,
        if_III_kwargs={
            "sample_timestep_respacing": "smart100",
            "aug_level": 0.5,
            "guidance_scale": 6.0,
        },
        # if_III_kwargs={
        #     "guidance_scale": 4.0,
        #     "sample_timestep_respacing": "smart50",
        # },
        disable_watermark=True,
    )
    middle_res["III"][0].save(output_path)
