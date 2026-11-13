# ComfyUI-Flat2VR

Turn a normal photo into a stereo **VR180** image in ComfyUI: a **flat2vr** LoRA builds the 180° left eye around your
photo, the **l2r** LoRA (Qwen-Image 2.1) makes the matching right eye. Two workflows:

- `workflows/h3-image.json`: flat2vr on **MiniMax-H3-Image**, with a hi-res fix (the left eye is re-rendered and decoded
  at 4096, then scaled back to 2048, which keeps the fine detail H3's compressed VAE loses at 2048), then the Qwen l2r
  right eye. Output 2048 per eye.
- `workflows/qwen.json`: flat2vr on **Qwen-Image 2.1** with a light refine pass, then the l2r right eye.

## Features

- **Place Photo**: drop, paste or upload a photo and place it in a live 180° preview (drag to move, scroll to resize). It is cut to the nearest trained shape automatically; hold **Shift** to crop by hand.
- **Prompt**: builds the prompt exactly as the training captions were written (optional setting, position and POV body). Turn `label_no_body` on for the H3 LoRA: its captions say "POV body: none." when there is no body.
- **Paste Photo**: puts your original photo pixels back after generation.
- **Refine Mask**: a seam-free mask for an optional detail pass.
- **Right Eye (best of N)**: makes the right eye (KSampler + VAE Decode in one), checks it against the left eye and tries the next seed until one passes every check, between `min_tries` and `max_tries` rolls, keeping the best. Each check has its own limit (0 = off): pose changes, zoom, relief (too little = cardboard), smeared depth edges, background pop-outs and background beyond infinity. The defaults are what 90% of real stereo photos stay within, a little looser for halo and beyond-infinity (Join Eyes fixes the latter). Its `report` output lists every try.
- **Join Eyes**: left | right side by side (4096×2048). It matches the right eye's brightness to the left and, if part of the view sits beyond infinity (eyes would splay outward), turns the right eye back a few pixels. Both can be switched off.
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
- For `h3-image.json`, also the models from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) (needs a ComfyUI with the **MiniMax H3 Image Conditioning** node):

  | File | Folder |
  |---|---|
  | `minimax_h3_image_pruned_int8_convrot.safetensors` | `models/diffusion_models` |
  | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders` |
  | `minimax_h3_image_vae_fp16.safetensors` | `models/vae` |
- The LoRAs (not included here) in `models/loras`: flat2vr `vr180_flat2vr_v6_*.safetensors` (H3) or `vr180_flat2vr_v4_000001250.safetensors` (Qwen), and l2r `vr180_l2r_v2_000002300.safetensors`.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/troysniper01/ComfyUI-Flat2VR
```

Restart ComfyUI. It only needs numpy and OpenCV, which ComfyUI already has.

## Usage

1. Open `workflows/h3-image.json` (or `workflows/qwen.json`).
2. Drop your photo on **Flat2VR Place Photo** and place it. A red frame means it leaves the 180° view.
3. Optionally describe the scene in **Flat2VR Prompt**.
4. Run. The workflow makes the left eye, refines it (H3: re-rendered at 4096 and scaled back to 2048), makes the right eye and joins them. The side-by-side result is saved as `output/flat2vr_h3_180_LR_*.png` (H3) or `output/flat2vr_180VR_*.png` (Qwen); look around it in **180 Preview**. The H3 workflow can also save the 4096 left eye: its save node is bypassed, select it and press Ctrl+B.

If your VR player doesn't detect the format, add `_180_LR` to the file name.

## Tips

- Use real photos: the LoRAs were trained on photographs, and drawings or cartoons come back as plain grey.
- Keep the sampling as trained. H3: 6 steps, cfg 1, euler / simple; its hi-res pass enters that 6-step grid at `start_at_step` 4 (3 = stronger, 5 = lighter). Qwen: 40 steps, cfg 1, euler / simple. flat2vr LoRA strength 0.8–1.0, l2r 1.0.
- Level or slightly downward placements (pitch −15° to +5°) work best.
