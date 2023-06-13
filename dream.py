from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
from deepfloyd_if.modules.t5 import T5Embedder

device = "cuda:0"
if_I = IFStageI("IF-I-XL-v1.0", device=device)
if_II = IFStageII("IF-II-L-v1.0", device=device)
t5 = T5Embedder(device="cpu")
if_III = StableStageIII("stable-diffusion-x4-upscaler", device=device)

from deepfloyd_if.pipelines import dream

prompt = "woman with a blue headscarf and a blue sweaterp, detailed picture, 4k dslr, best quality"
count = 1

result = dream(
    t5=t5,
    if_I=if_I,
    if_II=if_II,
    if_III=None,
    prompt=[prompt] * count,
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
result["I"][0].save("./dream_I.png")
result["II"][0].save("./dream_II.png")
