import csv
import zipfile
import requests
from bs4 import BeautifulSoup
import re
import json
import os
import random
import shutil
from urllib.parse import quote
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import color
from sklearn.cluster import KMeans
import numpy as np

# constants

COMPRESSED_QUALITY = 60
MAX_FILMS = 300
MAX_WORKERS = 8  # adjust based on your connection
SAMPLE = 2  # number of images to sample per film
NEIGHBORS = 5  # number of nearest neighbors for force graph
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

# color helpers

def rgb_to_lab(rgb):
    rgb = np.array(rgb)/255
    lab = color.rgb2lab([[rgb]])[0][0]
    return lab

def color_distance(c1, c2):
    return np.linalg.norm(np.array(c1) - np.array(c2))

# letterboxd parsing

def load_watched(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('watched.csv') as f:
            reader = csv.DictReader(f.read().decode('utf-8').splitlines())
            movies = [row for row in reader]
    return movies

def normalize_title(title):
    return re.sub(r'\s*\(\d{4}\)$', '', title).strip().lower()

# film-grab scraping and caching

def scrape_filmgrab_index():
    r = requests.get("https://film-grab.com/movies-a-z/", headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    films = soup.select("li.listing-item a.title:first-of-type")
    return [(item.text.strip(), item["href"]) for item in reversed(films)]

def match_films(watched, filmgrab_index):
    lookup = {
        normalize_title(m["Name"]): {"title": m["Name"], "year": m["Year"]}
        for m in watched
    }
    matched = {}
    for title, url in filmgrab_index:
        normalized = normalize_title(title)
        if normalized not in lookup or normalized in matched:
            continue
        year_match = re.search(r'\((\d{4})\)$', title)
        year = year_match.group(1) if year_match else None
        if year is None or year == lookup[normalized]["year"]:
            matched[normalized] = {
                "url": url,
                "title": lookup[normalized]["title"],
                "year":  lookup[normalized]["year"],
                "filepaths": []
            }
    return matched

def get_image_urls(film_url):
    r = requests.get(film_url, headers=HEADERS)
    soup = BeautifulSoup(r.text, "html.parser")
    links = soup.select("a.bwg-a")
    selected = random.sample(links, min(SAMPLE, len(links)))
    return [link['href'] for link in selected]

def cache_film(normalized, info, cache_dir):
    film_dir = os.path.join(cache_dir, normalized)
    os.makedirs(film_dir, exist_ok=True)
    image_urls = get_image_urls(info["url"])
    filepaths = []
    for i, url in enumerate(image_urls):
        encoded = quote(url, safe=":/?=&")
        img_data = requests.get(encoded, headers=HEADERS).content
        img = Image.open(BytesIO(img_data)).convert("RGB")
        path = os.path.join(film_dir, f"{i+1}.jpg")
        img.save(path, "JPEG", quality=COMPRESSED_QUALITY, optimize=True)
        rel = os.path.relpath(path, cache_dir).replace(os.sep, "/")
        filepaths.append(rel)
    return filepaths

# build force graph JSON

def build_force_graph(sampled, cache_dir, output_path, neighbors=NEIGHBORS):
    images = []
    for normalized, info in sampled.items():
        for filepath in info.get("filepaths", []):
            try:
                full_path = os.path.join(cache_dir, filepath)
                img = Image.open(full_path).convert("RGB")
                avg_rgb = np.array(img).reshape(-1, 3).mean(axis=0)
                lab = rgb_to_lab(avg_rgb)
                images.append({
                    "id":   filepath,
                    "file": filepath,
                    "lab":  lab,
                    "film": info["title"],
                    "year": info["year"],
                    "url":  info["url"]
                })
            except Exception as e:
                print("Skipped:", filepath, e)

    links = []
    for i, img_a in enumerate(images):
        distances = sorted(
            [(j, color_distance(img_a["lab"], img_b["lab"]))
             for j, img_b in enumerate(images) if i != j],
            key=lambda x: x[1]
        )
        nearest  = distances[:neighbors]
        max_dist = nearest[-1][1] if nearest else 1
        for j, dist in nearest:
            links.append({
                "source":   img_a["file"],
                "target":   images[j]["file"],
                "strength": round(1 - dist / max_dist, 3)
            })

    nodes = [{k: v if k != "lab" else v.tolist() for k, v in img.items()} for img in images]

    with open(output_path, "w") as f:
        json.dump({"nodes": nodes, "links": links}, f)

# main pipeline

def run_pipeline(zip_path, session_dir, progress):
    """
    progress(pct, message) — called throughout so Flask can stream updates.
    All outputs are written into session_dir.
    """
    cache_dir  = os.path.join(session_dir, "cached_images")
    graph_path = os.path.join(session_dir, "force_graph.json")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Parse Letterboxd zip
    progress(5, "Parsing Letterboxd data...")
    watched = load_watched(zip_path)

    # 2. Scrape film-grab index
    progress(10, "Scraping film-grab index...")
    filmgrab_index = scrape_filmgrab_index()

    # 3. Match films
    progress(15, "Matching films...")
    matched = match_films(watched, filmgrab_index)
    sampled = dict(random.sample(list(matched.items()), min(MAX_FILMS, len(matched))))

    # 4. Scrape + cache images (threaded, progress per film)
    total  = len(sampled)
    done   = 0

    def process(item):
        normalized, info = item
        info["filepaths"] = cache_film(normalized, info, cache_dir)
        return normalized, info

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process, item): item for item in sampled.items()}
        for future in as_completed(futures):
            try:
                normalized, info = future.result()
                sampled[normalized] = info
            except Exception as e:
                print("Film failed:", e)
            done += 1
            pct = 15 + int((done / total) * 70)  # 15→85%
            progress(pct, f"Scraped {future.result()[1]['title']} ({done}/{total})")

    # 5. Build graph JSON
    progress(94, "Building graph...")
    build_force_graph(sampled, cache_dir, graph_path)

    progress(100, "Done!")
    return graph_path