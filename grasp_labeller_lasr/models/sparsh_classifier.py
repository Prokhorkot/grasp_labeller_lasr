import torch
from huggingface_hub import hf_hub_download
from tactile_ssl.model import vit_base

ckpt_path = hf_hub_download(
    repo_id="facebook/sparsh-ijepa-base",
    filename="ijepa_vitbase.ckpt",
)

encoder = vit_base(
    img_size=(320, 240),
    in_chans=6,
    pos_embed_fn="sinusoidal",
    num_register_tokens=1,
)

ckpt = torch.load(ckpt_path, map_location="cuda")
state = {
    k.replace("target_encoder.", ""): v
    for k, v in ckpt["model"].items()
    if k.startswith("target_encoder.")
}

encoder.load_state_dict(state, strict=False)
encoder.eval()