from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder
import pandas as pd
from tqdm import tqdm
import os

output_dir = "/mnt/blob/dataset"
result_dir = os.path.join(output_dir, "Deepfloyd-IF")

device = "cuda:0"
if_I = IFStageI("IF-I-XL-v1.0", device=device)
if_II = IFStageII("IF-II-L-v1.0", device=device)
t5 = T5Embedder(device="cpu")
if_III = StableStageIII("stable-diffusion-x4-upscaler", device=device)

from deepfloyd_if.pipelines import dream

xlsx_path = os.path.join(output_dir, 'DrawBench Prompts.csv')
Prompts = pd.read_csv(xlsx_path)['Prompts']

# prompt = "An astronaut riding a green horse"
for i, prompt in enumerate(tqdm(Prompts, desc='Generating Deepfloyd-IF images', total=len(Prompts))):
    os.makedirs(os.path.join(result_dir, "stage1"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "stage2"), exist_ok=True)
    os.makedirs(os.path.join(result_dir, "stage3"), exist_ok=True)
    result = dream(
        t5=t5,
        if_I=if_I,
        if_II=if_II,
        if_III=if_III,
        prompt=[prompt] * 1,
        seed=42,
        if_I_kwargs={
            "guidance_scale": 7.0,
            "sample_timestep_respacing": "smart100",
        },
        if_II_kwargs={
            "guidance_scale": 4.0,
            "sample_timestep_respacing": "smart50",
        },
        if_III_kwargs={
            "guidance_scale": 9.0,
            "noise_level": 20,
            "sample_timestep_respacing": "75",
        },
        disable_watermark=True,
    )
    result["I"][0].save(os.path.join(result_dir, f"stage1/{str(i).zfill(3)}.png"))
    result["II"][0].save(os.path.join(result_dir, f"stage2/{str(i).zfill(3)}.png"))
    result["III"][0].save(os.path.join(result_dir, f"stage3/{str(i).zfill(3)}.png"))
