from glob import glob 
import torch

import os 
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
import lpips
import torch.nn.functional as F

IMG_SIZE = 256
# from met3r import MEt3R
# Initialize MEt3R
# metric = MEt3R(
#     img_size=IMG_SIZE, # Default. Set to `None` to use the input resolution on the fly!
#     use_norm=True, # Default 
#     feat_backbone="dino16", # Default 
#     featup_weights="mhamilton723/FeatUp", # Default 
#     dust3r_weights="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric", # Default
#     use_mast3r_dust3r=True # Default. Set to `False` to use original DUSt3R. Make sure to also set the correct weights from huggingface.
# ).cuda()


from transformers import AutoImageProcessor, Dinov2Model
import torch

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base").cuda()
lpips_measure = lpips.LPIPS(net="alex").cuda()

@torch.no_grad()
def compute_dino_diversity(frames, target_frame):
    """Compute variance of DINO feature embeddings."""
    source = image_processor(frames, return_tensors="pt").to("cuda")
    source_embs = dino_model(**source).last_hidden_state[:, 0]
    target = image_processor(target_frame, return_tensors="pt").to("cuda")
    target_emb = dino_model(**target).last_hidden_state[:, 0]
    result = abs(F.cosine_similarity(source_embs, target_emb)).mean().item()

    return result

def compute_dino_div(frames):
    """Compute variance of DINO feature embeddings."""
    source = image_processor(frames, return_tensors="pt").to("cuda")
    source_embs = dino_model(**source).last_hidden_state[:, 0]
    # batch size variance    
    result = (torch.var(source_embs, dim=0)**0.5).mean().item()

    return result
# from common_metrics_on_video_quality.calculate_fvd import calculate_fvd
## Minecraft
# NUM_CONTEXT = 8
# gif_path_list = sorted(glob.glob("/shared/diff/diffusion-forcing/outputs/2025-02-03/13-16-12/wandb/latest-run/files/media/videos/validation_vis_pred/*.gif"))
# gif_path_list = sorted(glob.glob("/shared/diff/diffusion-forcing/outputs/2025-02-04/15-17-53/wandb/latest-run/files/media/videos/validation_vis_pred/*.gif"))
# gif_path_list = sorted(glob.glob("/shared/diff/diffusion-forcing/outputs/2025-02-04/15-24-19/wandb/latest-run/files/media/videos/validation_vis_pred/*.gif"))
# repeat_len = 4
# repeat_len = 8
# repeat_len = 16

## dmlab
NUM_CONTEXT = 4
# gif_path_list = sorted(glob.glob("/shared/diff/diffusion-forcing/outputs/2025-02-06/07-57-55/wandb/latest-run/files/media/videos/validation_vis_pred/*.gif")) # 2
# gif_path_list = sorted(glob.glob("/shared/diff/diffusion-forcing/outputs/2025-02-06/07-59-07/wandb/latest-run/files/media/videos/validation_vis_pred/*.gif")) # 4
# gif_path_list = sorted(glob.glob("/shared/diff/diffusion-forcing/outputs/2025-02-06/08-00-22/wandb/latest-run/files/media/videos/validation_vis_pred/*.gif")) # 8
# gif_path_list = sorted(glob.glob("/shared/diff/diffusion-forcing/outputs/2025-02-06/07-48-22/wandb/latest-run/files/media/videos/validation_vis_pred/*.gif")) # 16

paths = []
# 8-base
result_path = "/fsx/jeremy/DF/ws4/base_ws/wandb/latest-run/files/media/videos/validation_vis/"
paths.append(result_path)
# 8-aug
result_path = "/fsx/jeremy/DF/ws4/aug_ws/wandb/latest-run/files/media/videos/validation_vis/"
paths.append(result_path)
# 8-ours_wo
result_path = "/fsx/jeremy/DF/ws4/ours_ws/wandb/latest-run/files/media/videos/validation_vis/"
paths.append(result_path)

# 8-base
result_path = "/fsx/jeremy/DF/ws4/base_twice_ws/wandb/latest-run/files/media/videos/validation_vis/"
paths.append(result_path)
# 8-aug
result_path = "/fsx/jeremy/DF/ws4/aug_twice_ws/wandb/latest-run/files/media/videos/validation_vis/"
paths.append(result_path)
# 8-ours_wo
result_path = "/fsx/jeremy/DF/ws4/ours_twice_ws/wandb/latest-run/files/media/videos/validation_vis/"
paths.append(result_path)


for path in paths:
    print(path)
    inputs = []
    div_scores = []
    lpips_cons, lpips_div, lpips_ws = [], [], []
    dino_cons, dino_div, dino_ws = [], [], []
    
    gif_path_list = sorted(glob(f"{path}/*.gif"))

    for gif_path in tqdm(gif_path_list, total=len(gif_path_list)):

        gif = Image.open(gif_path)

        # Extract frames
        _frames, _gt_frames = [], []
        frame_index = 0

        while True:
            frame = gif.copy()  # Copy the current frame
            _frames.append(frame)
            try:
                gif.seek(gif.tell() + 1)  # Move to next frame
            except EOFError:
                break  # Stop if no more frames

        # Save frames as individual images
        frames = []
        for i, frame in enumerate(_frames[NUM_CONTEXT:NUM_CONTEXT+20]):
            frames.append(np.array(frame)[:, :64])
        # frames = sorted(glob(os.path.join(p, "*.png")), key=lambda x: int(x.split("frame_")[1].split(".png")[0]))
        frames_tensor = torch.stack([torch.from_numpy(np.array(Image.fromarray(frame).resize((IMG_SIZE, IMG_SIZE)))).permute(2, 0, 1) for frame in frames]).div(255) * 2 -1
        frames_tensor = frames_tensor.cuda()
        init_frame = frames_tensor[0]
        last_frame = frames_tensor[-1]
        mid_frame = frames_tensor[len(frames_tensor) // 2]
        
        # _input = torch.cat([init_frame.unsqueeze(0), last_frame.unsqueeze(0)], dim=0)
        # inputs.append(_input)

        # 중간이랑 비교
        # _target_frame = frames_tensor[len(frames) // 2]
        # inputs2 = [torch.cat([frames_tensor[0].unsqueeze(0), _target_frame.unsqueeze(0)], dim=0), torch.cat([frames_tensor[-1].unsqueeze(0), _target_frame.unsqueeze(0)], dim=0)]
        # inputs2 = torch.stack(inputs2, dim=0)
        # div_score, *_ = metric(
        #     images=inputs2, 
        # return_overlap_mask=False, # Default 
        # return_score_map=False, # Default 
        # return_projections=False # Default 
        # )
        # div_scores.append(div_score.mean().item())
        
        lpips_cons.append(lpips_measure(last_frame, init_frame).squeeze().item())
        # lpips_div.append(np.mean([lpips_measure(_s, _t).squeeze().item() for _s, _t in zip(_source_frames, _target_frames)]))
        # lpips_div.append(np.mean([v.item() for v in lpips_measure(_source_frames, _target_frames)]))
        lpips_div.append((lpips_measure(mid_frame, init_frame).squeeze().item() + lpips_measure(mid_frame, last_frame).squeeze().item())* 0.5)

        dino_cons.append(1-compute_dino_diversity([Image.fromarray(frames[0])], Image.fromarray(frames[-1])))

        # dino_div.append(compute_dino_div([Image.fromarray(v) for v in frames[:len(frames)//2]]))
        # dino_div.append(np.std([1-compute_dino_diversity([Image.fromarray(_s)], Image.fromarray(_t)) for _s, _t in zip(frames[1:len(frames)//2], frames[:len(frames)//2-1])]))
        dino_div.append(np.mean([1-compute_dino_diversity([Image.fromarray(frames[0])], Image.fromarray(frames[len(frames)//2])), 
                        1-compute_dino_diversity([Image.fromarray(frames[-1])], Image.fromarray(frames[len(frames)//2]))]
        ))
        
    print(path[0].split("episode")[0])
    print("LPIPS")
    print(round(np.mean(lpips_cons), 4))
    print(round(np.mean(lpips_div), 4))
    print(round(np.mean([con/div for con, div in zip(lpips_cons, lpips_div)]), 4))
    print()

    print("DINO")
    print(round(np.mean(dino_cons), 4))
    print(round(np.mean(dino_div), 4))
    print(round(np.mean([con/div for con, div in zip(dino_cons, dino_div)]), 4))
    print()
    
    # inputs = torch.stack(inputs, dim=0)
    # inputs = inputs.clip(-1, 1).cuda()
    
    # Evaluate MEt3R
    # sim_score, *_ = metric(
    #     images=inputs, 
    #     return_overlap_mask=False, # Default 
    #     return_score_map=False, # Default 
    #     return_projections=False # Default 
    # )
    # result = []
    # cands = inputs.chunk(4)
    # for v in cands:
    #     sim_score, *_ = metric(
    #         images=v, 
    #     return_overlap_mask=False, # Default 
    #     return_score_map=False, # Default 
    #     return_projections=False # Default 
    #     )
    #     result.append(sim_score)
    # sim_score = torch.cat(result, dim=0)

    # Should be between 0.25 - 0.35
    # print("MET3R")
    # print(round(sim_score.mean().item(), 4))
    # print(round(np.mean(div_scores), 4))
    # print(round(np.mean([con/div for con, div in zip(sim_score.cpu().numpy(), div_scores)]), 4))
    # Clear up GPU memory
    torch.cuda.empty_cache()