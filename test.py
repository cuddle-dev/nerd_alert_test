import os
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict

BASE_URL = "https://inversionrecruitment.blob.core.windows.net/find-the-code/"

N_COLS = 40
N_ROWS = 30
TOTAL = 1200

os.makedirs("tiles", exist_ok=True)

tiles = {}

for i in range(1, TOTAL + 1):
    url = f"{BASE_URL}{i}.png"
    r = requests.get(url)
    img = Image.open(BytesIO(r.content)).convert("L")
    tiles[i] = np.array(img)
    img.save(f"tiles/{i}.png")

def edges(img):
    top = img[0, :]
    bottom = img[-1, :]
    left = img[:, 0]
    right = img[:, -1]
    return top, right, bottom, left

def edge_diff(e1, e2):
    return np.mean(np.abs(e1 - e2))

tile_edges = {}
for k, img in tiles.items():
    tile_edges[k] = edges(img)

def is_black(edge):
    return np.mean(edge) < 5

corners = []
for k, (t, r, b, l) in tile_edges.items():
    black_sides = sum([
        is_black(t), is_black(r),
        is_black(b), is_black(l)
    ])
    if black_sides == 2:
        corners.append(k)

def best_match(tile_id, side, used):
    best = None
    best_score = float("inf")
    e1 = tile_edges[tile_id][side]
    
    for k in tiles:
        if k in used:
            continue
        for s2 in range(4):
            e2 = tile_edges[k][s2]
            score = edge_diff(e1, e2[::-1])
            if score < best_score:
                best_score = score
                best = k
    return best

grid = [[None]*N_COLS for _ in range(N_ROWS)]
used = set()

start = corners[0]
grid[0][0] = start
used.add(start)

for c in range(1, N_COLS):
    prev = grid[0][c-1]
    nxt = best_match(prev, 1, used)
    grid[0][c] = nxt
    used.add(nxt)

for r in range(1, N_ROWS):
    for c in range(N_COLS):
        above = grid[r-1][c]
        nxt = best_match(above, 2, used)
        grid[r][c] = nxt
        used.add(nxt)

tile_h, tile_w = tiles[1].shape
final = np.zeros((tile_h*N_ROWS, tile_w*N_COLS), dtype=np.uint8)

for r in range(N_ROWS):
    for c in range(N_COLS):
        final[r*tile_h:(r+1)*tile_h,
              c*tile_w:(c+1)*tile_w] = tiles[grid[r][c]]

Image.fromarray(final).save("reconstructed.png")