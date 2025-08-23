# CLIP search server
This is a Flask API server using video CLIP models to index images/GIFs/MP4s and provide search functionality, intended for CLIP-powered Discord favourited GIF search.

# Installing
- Install the requirements from requirements.txt with pip.
- Create an empty directory named 'indexes'
- Download [VideoCLIP-XL-v2.bin](https://huggingface.co/alibaba-pai/VideoCLIP-XL-v2/blob/main/VideoCLIP-XL-v2.bin) into models/videoclip_xl_v2
- Run the server using the command in run.sh

# Usage
Install the [Vencord plugin](https://github.com/Woodie-07/clipFavGifSearch)
