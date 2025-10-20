# turbo_fgsm_cloak.py
import torch, clip
from PIL import Image
from torchvision import transforms


def turbo_fgsm_cloak(in_path, out_path, target_text="oil painting", eps=1/255):
    """
    One-step FGSM nudge to push the image embedding toward `target_text`.
    Fast demo: ~1–2s CPU for 224px; not robust like Glaze.
    """
    device = "cpu"
    model, _ = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()

    # Preprocess to 224 for speed
    x = Image.open(in_path).convert("RGB").resize((224, 224), Image.BICUBIC)
    to_tensor, to_pil = transforms.ToTensor(), transforms.ToPILImage()

    x = to_tensor(x).unsqueeze(0).to(device)
    x.requires_grad_(True)

    with torch.no_grad():
        txt = clip.tokenize([target_text]).to(device)
        tfeat = model.encode_text(txt)
        tfeat /= tfeat.norm(dim=-1, keepdim=True)

    # Encode image with CLIP normalization
    def encode(img_batch):
        # CLIP expects normalized 224x224
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)[None, :, None, None]
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)[None, :, None, None]
        z = (img_batch - mean) / std
        return model.encode_image(z)

    # Loss: maximize cosine similarity to target_text
    if x.grad is not None: x.grad.zero_()
    if hasattr(model, 'zero_grad'): model.zero_grad()

    if x.grad is not None: x.grad.zero_()
    img_feat = encode(x)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    loss = -torch.cosine_similarity(img_feat, tfeat).mean()
    loss.backward()

    # FGSM update
    x_adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()

    out = to_pil(x_adv.squeeze(0).cpu())
    out.save(out_path)
