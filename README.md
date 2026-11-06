# ComfyUI-Flat2VR

Turn a normal photo into a stereo **VR180** image in ComfyUI, with two LoRAs for **Qwen-Image 2.1** (image edit):
**flat2vr** builds the 180° left eye around your photo, **l2r** makes the matching right eye.

## Features

- **Place Photo**: drop, paste or upload a photo and place it in a live 180° preview (drag to move, scroll to resize). It is cut to the nearest trained shape automatically; hold **Shift** to crop by hand.
- **Prompt**: builds the prompt exactly as the training captions were written (optional setting, position and POV body).
- **Paste Photo**: puts your original photo pixels back after generation.
- **Refine Mask**: a seam-free mask for an optional detail pass.
- **Join Eyes**: left | right side by side (4096×2048).
- **180 Preview**: look around the result inside the node, with a left/right eye toggle.

## Requirements

- ComfyUI with the **Text Encode Qwen Image 2.1** node (recent ComfyUI).
- Models from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1):

  | File | Folder |
  |---|---|
  | `qwen_image_2.1_int8_convrot.safetensors` | `models/diffusion_models` |
  | `qwen3vl_8b_int8_convrot.safetensors` | `models/text_encoders` |
  | `qwen_image_2.1_vae_bf16.safetensors` | `models/vae` |

  For maximum quality use the `bf16` versions of the first two (the LoRAs were trained on them) and select them in the loaders.
- The two LoRAs (not included here) in `models/loras`: `vr180_flat2vr_v4_000001250.safetensors` and `vr180_l2r_v2_000002300.safetensors`.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/troysniper01/ComfyUI-Flat2VR
```

Restart ComfyUI. It only needs numpy and OpenCV, which ComfyUI already has.

## Usage

1. Open `workflows/flat2vr_v1.json`.
2. Drop your photo on **Flat2VR Place Photo** and place it. A red frame means it leaves the 180° view.
3. Optionally describe the scene in **Flat2VR Prompt**.
4. Run. The workflow makes the left eye, refines it, makes the right eye and joins them. The side-by-side result is saved as `output/flat2vr_180VR_*.png`; look around it in **180 Preview**.

If your VR player doesn't detect the format, add `_180_LR` to the file name.

## Tips

- Use real photos: the LoRAs were trained on photographs, and drawings or cartoons come back as plain grey.
- Keep the sampling as trained: 40 steps, cfg 1, euler / simple. flat2vr LoRA strength 0.8–1.0, l2r 1.0.
- Level or slightly downward placements (pitch −15° to +5°) work best.
