#!/usr/bin/env python3
import argparse
import pathlib
import random


def blank(width, height, value=255):
    return [[value, value, value] for _ in range(width * height)]


def put_px(buf, width, height, x, y, rgb):
    if 0 <= x < width and 0 <= y < height:
        buf[y * width + x] = [rgb[0], rgb[1], rgb[2]]


def draw_rect(buf, width, height, x0, y0, x1, y1, rgb, fill=False):
    x0, x1 = sorted((max(0, x0), min(width - 1, x1)))
    y0, y1 = sorted((max(0, y0), min(height - 1, y1)))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if fill or y in (y0, y1) or x in (x0, x1):
                put_px(buf, width, height, x, y, rgb)


def draw_line(buf, width, height, x0, y0, x1, y1, rgb):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        put_px(buf, width, height, x0, y0, rgb)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def draw_checker(buf, width, height, block, c0, c1):
    for y in range(height):
        for x in range(width):
            use_c0 = ((x // block) + (y // block)) % 2 == 0
            put_px(buf, width, height, x, y, c0 if use_c0 else c1)


def write_ppm(path: pathlib.Path, width: int, height: int, buf):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        raw = bytearray()
        for px in buf:
            raw.extend(px)
        f.write(raw)


def gen_image(idx: int, width: int, height: int, rng: random.Random):
    buf = blank(width, height, 255)
    mode = idx % 5

    if mode == 0:
        draw_checker(buf, width, height, rng.choice([4, 8, 16, 32]), (0, 0, 0), (255, 255, 255))
    elif mode == 1:
        for _ in range(rng.randint(40, 120)):
            x0 = rng.randrange(width)
            y0 = rng.randrange(height)
            x1 = rng.randrange(width)
            y1 = rng.randrange(height)
            color = rng.choice([(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 128, 0)])
            draw_line(buf, width, height, x0, y0, x1, y1, color)
    elif mode == 2:
        for _ in range(rng.randint(10, 40)):
            x0 = rng.randrange(width)
            y0 = rng.randrange(height)
            x1 = rng.randrange(width)
            y1 = rng.randrange(height)
            fill = rng.random() < 0.2
            color = rng.choice([(0, 0, 0), (255, 255, 255), (20, 20, 20), (230, 230, 230)])
            draw_rect(buf, width, height, x0, y0, x1, y1, color, fill=fill)
    elif mode == 3:
        stripe = rng.choice([1, 2, 3, 4, 6, 8])
        for y in range(height):
            on = ((y // stripe) % 2) == 0
            c = (0, 0, 0) if on else (255, 255, 255)
            for x in range(width):
                put_px(buf, width, height, x, y, c)
    else:
        # Step edges and boxed "UI-like" regions.
        split_x = rng.randrange(width // 4, 3 * width // 4)
        split_y = rng.randrange(height // 4, 3 * height // 4)
        for y in range(height):
            for x in range(width):
                c = (245, 245, 245)
                if x > split_x:
                    c = (20, 20, 20)
                if y > split_y:
                    c = (220, 220, 220) if x <= split_x else (40, 40, 40)
                put_px(buf, width, height, x, y, c)
        for _ in range(18):
            x0 = rng.randrange(width)
            y0 = rng.randrange(height)
            w = rng.randrange(8, 72)
            h = rng.randrange(8, 36)
            draw_rect(buf, width, height, x0, y0, x0 + w, y0 + h, (0, 0, 0), fill=False)

    return buf


def main():
    ap = argparse.ArgumentParser(description="Generate deterministic text/UI/line-art stress set (PPM)")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        buf = gen_image(i, args.width, args.height, rng)
        write_ppm(args.out_dir / f"stress_{i:04d}.ppm", args.width, args.height, buf)

    print(f"Generated {args.count} stress images in {args.out_dir}")


if __name__ == "__main__":
    main()
