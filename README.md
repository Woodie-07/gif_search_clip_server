# CLIP search server
This is a Flask API server using video CLIP models to index images/GIFs/MP4s and provide search functionality, intended for CLIP-powered Discord favourited GIF search.

[Stats for my hosted instance](https://stats.woodie.dev/public-dashboards/b616106008c7484eacfb869eccf7b2f6)

## Features
- Can download media, process them through CLIP models, and index them in per-user FAISS indexes
- Multiple models supported, easy to add more
- Multiple workers can be connected over the internet to balance load and accelerate processing. Not all workers have to support all models.
- Provides status information for users, both per media item and overall
- Allows exporting detailed stats in Prometheus format

## Installing
You must follow these steps only if you wish to run your own instance of the server. Otherwise, you can use my hosted instance, which is the default in the Vencord plugin.
### Server
Inside the `server` working directory:
- Create an activate a venv
- Install the requirements from requirements.txt with pip.
- Run the server using the command in run.sh

### Worker
Inside the `worker` working directory
- Create an activate a venv
- Install the requirements from requirements.txt with pip.
- Download [VideoCLIP-XL-v2.bin](https://huggingface.co/alibaba-pai/VideoCLIP-XL-v2/blob/main/VideoCLIP-XL-v2.bin) into models/videoclip_xl_v2
- If you wish to use the Frozen-in-Time model, download [cc-webvid2m-4f_stformer_b_16_224.pth.tar](https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/models/cc-webvid2m-4f_stformer_b_16_224.pth.tar) into models/frozen_in_time, otherwise comment it out of models/\_\_init\_\_.py. I'd recommend commenting it out and possibly X-CLIP too as VideoCLIP-XL-v2 seems far superior in my testing.

## Usage
Install the [Vencord plugin](https://github.com/Woodie-07/clipFavGifSearch)


<small>SoM: GitHub Copilot autocomplete was used in this project.</small>
