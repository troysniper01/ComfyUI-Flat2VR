# ComfyUI-Flat2VR

Nodes for turning a flat photo into a VR180 image: the flat2vr LoRA (MiniMax-H3-Image) makes the left eye around the
photo, the l2r LoRA (Qwen-Image 2.1) makes the right eye.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/troysniper01/ComfyUI-Flat2VR
```

Or install it from ComfyUI Manager (search "Flat2VR"). Needs a recent ComfyUI (the workflow uses subgraphs). GPL-3.0.

Models:

- [MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image): `minimax_h3_image_pruned_int8_convrot`, `qwen3vl_32b_minimax_h3_nvfp4_awq`, `minimax_h3_image_vae_fp16`
- [Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1): `qwen_image_2.1_int8_convrot`, `qwen3vl_8b_int8_convrot`, `qwen_image_2.1_vae_bf16`
- the flat2vr and l2r LoRAs in `models/loras`

## Workflow

`workflows/flat_to_180vr.json`. Place the photo in Flat2VR Place Photo, optionally fill in Flat2VR Prompt, and run.
It goes Left Eye → Hi-res Fix (re-rendered at 4096, back to 2048) → Right Eye → Soften → Join Eyes → Vignette.
Output is 2048 per eye, side by side. If a VR player doesn't detect it, add `_180_LR` to the file name.

Sampling as trained: H3 6 steps, l2r 40 steps, both cfg 1, euler / simple.

## Nodes

- **Place Photo**: puts the photo on the 180° canvas (drag to move, scroll to resize, Shift to crop).
- **Prompt**: builds the prompt in the training-caption format.
- **Paste Photo**, **Refine Mask**: put the original pixels back; mask for the hi-res pass.
- **Right Eye (best of N)**: makes the right eye and retries seeds until it passes the stereo checks.
- **Join Eyes**: left | right, with brightness and divergence fixes.
- **Soften**: a slight blur that brings H3's sharpness down to the right eye's.
- **Vignette**: fades each eye's rim to black.
- **180 Preview**: look around the result.
