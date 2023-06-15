# MiniGPT-4 Caption Generation

[MiniGPT-4](https://minigpt-4.github.io/) is a multi-modal LLM. It combines a visual encoder and Vicuna as its LLM.
To use MiniGPT-4 for caption generation, the model must be interacted with as a chatbot. Some prompts that the authors 
MiniGPT-4 used are available in `MiniGPT-4/prompts/alignment.txt`. The prompt used to generate the included results is
"Describe this image in detail."

See `Use_minigpt4.ipynb` to make use of the model. Running `demo.py` will run the main demo for MiniGPT-4. The 
`generate_captions.py` script adapts the demo to read in a list of URLs, generate captions for each image, and
save the results. The model in its current configuration uses ~16GB of VRAM.

## Future Work

MiniGPT-4 was observed to suffer from significant hallucinations. It seems that the training of the model was 
focused more on natural responses rather than accurate responses. As [noted by others](https://github.com/Vision-CAIR/MiniGPT-4/issues/61),
the training set includes many captions that are not accurate, providing extra detail that does not exist. Our 
observations are aligned with this: many generated captions go into significant detail, often including nonexistent 
detail. In particular, many captions mention the sky, even when it is not present in the photo. [Other users](
https://github.com/Vision-CAIR/MiniGPT-4/issues/57) have experienced similar issues.

To get MiniGPT-4 to reduce its bias toward giving significant detail, the prompt "Is this a blank image? Describe it 
succintly." was experimented with. The first question was included, because without it, the model would not recognize 
uniform images. Asking the model to describe the image *succinctly* may have had some success with producing shorter 
captions. Other variations of asking to caption the image still resulted in lengthy captions. Prompt engineering can be 
explored further, as is following for updates on a trained model with a better quality dataset.

Currently, the repo for MiniGPT-4 has been included, as it was the easiest path for experimentation. Future work should 
try installing the repo with `pip` and making necessary modifications outside the repo. Two config files were modified, 
and the image loading code was modified to allow for URLs. The code for generating captions is as close to the demo code
as possible, which is not necessary, as this hacks the UI a bit to feed in the data programmatically. A more direct 
interface with the model should be explored.

Some code has been included but unused for threading the caption generation. This has not been tested due to VRAM limits
at the time. Caption generation takes 4-8s on a V100 GPU with 16GB of VRAM. Similar performance was observed on an RTX 
4090 with 24GB of VRAM running the model in 8-bit mode.

More work should be done to make caption generation more generic and flexible, regardless of system resources.

## Known Issues

The `bitsandbytes` library tends to cause issues when trying to load the model in 8-bit, which is `low_resource`. The 
8-bit model uses 12GB of VRAM, but dev2 was the only machine able to run it. There are other users with similar problems,
so perhaps this may be fixed in the future. There are several issues. The cause is unknown.

As mentioned before, the 16-bit model uses 16GB of VRAM. It is recommended to use a GPU with more than 16GB of VRAM, as 
some images can cause `CUDA Out of memory` errors. Chat conversations of more than length 2 can also easily push the GPU
beyond its limits. However, 16GB can suffice for most of the time.

## Summary of Changes

- Added and modified `Use_minigpt4.ipynb`, which is referenced in [MiniGPT-4's README](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/README.md).
- Added `generate_captions.py`
- Added `requirements.txt`
- Modified `minigpt4/conversation/conversation.py` to allow for URLs and some time reporting
- Modified `eval_configs/minigpt4_eval.yml`
- Modified `minigpt4/configs/models/minigpt4.yaml`
