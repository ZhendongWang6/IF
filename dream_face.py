import os
from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

device = "cuda:0"
if_I = IFStageI("IF-I-XL-v1.0", device=device)
if_II = IFStageII("IF-II-L-v1.0", device=device)
t5 = T5Embedder(device="cpu")
# if_III = StableStageIII("stable-diffusion-x4-upscaler", device=device)

from deepfloyd_if.pipelines import dream

prompt = "A photo of face"
count = 1

num_samples = 1000
result_dir = "/data3/wangzd/sr3/results/face/deep_if/generated"
os.makedirs(result_dir, exist_ok=True)
print("generating imagen face...")
for i in range(num_samples):
    # run pipeline in inference (sample random noise and denoise)
    print(f"{result_dir}/{i}.png")
    # save image
    result = dream(
        t5=t5,
        if_I=if_I,
        if_II=if_II,
        if_III=None,
        prompt=[prompt] * count,
        # seed=42,
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
    result["II"][0].save(f"{result_dir}/{i}.png")
