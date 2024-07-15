import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from accelerate import Accelerator
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.pose_guider import PoseGuider
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel

'''
f = safe_open("exp_output/stage1/checkpoint-10000/model.safetensors", 
				framework="pt", device="cpu")
reference_unet = {}
pose_guider = {}
denoising_unet = {}
for key in f.keys():
	if 'pose_guider' in key:
		pose_guider[key.removeprefix('pose_guider.')] = f.get_tensor(key)
	elif 'denoising_unet' in key:
		denoising_unet[key.removeprefix('denoising_unet.')] = f.get_tensor(key)
	elif 'reference_unet' in key:
		reference_unet[key.removeprefix('reference_unet.')] = f.get_tensor(key)
torch.save(denoising_unet, 'exp_output/stage1/denoising_unet-10000.pth')
torch.save(pose_guider, 'exp_output/stage1/pose_guider-10000.pth')
torch.save(reference_unet, 'exp_output/stage1/reference_unet-10000.pth')

'''
inference_config_path = "./configs/inference/inference_v2.yaml"
infer_config = OmegaConf.load(inference_config_path)

denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        './pretrained_weights/stable-diffusion-v1-5',
        "", subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
        },).to(device="cuda", dtype=torch.float16)

'''
denoising_unet = UNet3DConditionModel.from_pretrained_2d(
    './pretrained_weights/stable-diffusion-v1-5',
    './pretrained_weights/mm_sd_v15_v2.ckpt', subfolder="unet",
    unet_additional_kwargs=OmegaConf.to_container(
        infer_config.unet_additional_kwargs)).to(device="cuda", dtype=torch.float16)
'''
denoising_ckpt = torch.load("exp_output/stage1/denoising_unet-10000.pth", map_location="cpu")
count0, count1 = 0, 0
for name, param in denoising_unet.named_parameters():
	count0 += 1
	print('model', name)
for k, v in denoising_ckpt.items():
	print('weight', k)
	count1 += 1
print('model:', count0, 'ckpt:', count1)


'''
reference_unet = UNet2DConditionModel.from_pretrained(
        './pretrained_weights/stable-diffusion-v1-5',
        subfolder="unet",).to(device="cuda", dtype=torch.float16)
reference_ckpt = torch.load("exp_output/stage1/reference_unet-10000.pth", map_location="cpu")
reference_unet.load_state_dict(reference_ckpt, strict=True)

pose_guider = PoseGuider(conditioning_embedding_channels=320, 
    block_out_channels=(16, 32, 96, 256)).to(device="cuda", dtype=torch.float16)
pose_guider_ckpt = torch.load("exp_output/stage1//pose_guider-10000.pth", map_location="cpu")
pose_guider.load_state_dict(pose_guider_ckpt, strict=True)

'''


