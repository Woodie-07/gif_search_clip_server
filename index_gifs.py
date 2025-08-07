import random
import requests
import base64
from discord_protos import FrecencyUserSettings

def random_string(length, chars="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
    return ''.join(random.choice(chars) for _ in range(length))

with open("frecency_user_settings.txt", "r") as f:
    data = f.read().strip()

fus = FrecencyUserSettings.FromString(base64.b64decode(data))

user_key = random_string(32)

print(f"Using user key: {user_key}")
names = []
srcs = []
for name, gif in fus.favorite_gifs.gifs.items():
    if len(name) > 512:
        print(f"skipping {name} as {len(name)} > 512 characters")
        continue

    if len(gif.src) > 2000:
        print(f"skipping {name} as {len(gif.src)} > 2000 characters")
        continue

    if gif.src.startswith("http://"): domain_offset = 7
    elif gif.src.startswith("https://"): domain_offset = 8
    else:
        print(f"skipping {name} as invalid src: {gif.src}")
        continue

    path_idx = gif.src.find('/', domain_offset)
    if path_idx == -1:
        path_idx = len(gif.src)

    domain = gif.src[domain_offset:path_idx]

    if not domain or len(domain) > 256 or (not domain.endswith(".discordapp.net") and not domain == "media.tenor.co"):
        print(f"skipping {name} as invalid domain: {domain}")
        continue

    names.append(name)
    srcs.append(gif.src)

r = requests.post(f"http://127.0.0.1:5002/{user_key}/index", json={"names": names, "media_srcs": srcs, "models": [0, 1]})
if r.status_code != 202:
    print(f"Failed to index: {r.status_code} {r.text}")
else:
    print(f"Successfully queued the indexing of {len(names)} GIFs")
