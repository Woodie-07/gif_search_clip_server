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
for name, gif in fus.favorite_gifs.gifs.items():
    r = requests.post(f"http://127.0.0.1:5002/{user_key}/index", json={"name": name, "media_src": gif.src})
    if r.status_code != 202:
        print(f"Failed to index {name}: {r.status_code} {r.text}")
    else:
        print(f"Successfully queued the indexing of {name}")
