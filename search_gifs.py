import requests

user_key = input("Enter your user key: ").strip()
search_query = input("Enter your search query: ").strip()

r = requests.get(f"http://127.0.0.1:5002/{user_key}/search", params={"text": search_query})
if r.status_code == 200:
    results = r.json()["results"]
    for model_id, gifs in results.items():
        print(f"Model ID {model_id}:")
        for name, distance in gifs:
            print(f"  Name: {name}, Distance: {distance}")
else:
    print(f"Search failed: {r.status_code} {r.text}")
