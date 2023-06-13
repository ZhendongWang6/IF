from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

device = "cuda:0"
# if_I = IFStageI('IF-I-XL-v1.0', device=device)
if_II = IFStageII("IF-II-L-v1.0", device=device)
# if_III = StableStageIII('stable-diffusion-x4-upscaler', device=device)
t5 = T5Embedder(device="cpu")
from deepfloyd_if.pipelines import super_resolution

from PIL import Image

# open method used to open different extension image file
raw_pil_image = Image.open("pics/baby.png")
middle_res = super_resolution(
    t5,
    if_III=if_II,
    seed=42,
    prompt=['detailed picture, 4k dslr, best quality'],
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
)
middle_res["III"][0].save("./baby_prompt_none.png")

