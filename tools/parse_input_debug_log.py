#!/usr/bin/env python3
from argparse import ArgumentParser
from base64 import b64decode
from pathlib import Path
from struct import pack
from zlib import compress, crc32
import json


MASK_COLOR = (32, 220, 255)
BOX_COLOR = (255, 216, 0)
POSITIVE_COLOR = (42, 255, 96)
NEGATIVE_COLOR = (255, 64, 64)


def main():
    args = parser().parse_args()
    data_paths = find_data_paths(args.path)
    if not data_paths:
        raise SystemExit(f"No data.json files found under {args.path}")
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for data_path in data_paths:
        output_path = output_for(data_path, args.output_dir, args.output_name)
        image, stats = render_input_prompts(load_json(data_path))
        write_png(output_path, image)
        print(f"{output_path} points={stats['points']} box={stats['box']} mask_pixels={stats['mask_pixels']}")


def parser():
    parser = ArgumentParser(description="Render SAMM debug-log input prompts as PNG images.")
    parser.add_argument("path", nargs="?", type=Path, default=Path("logs/predictions"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-name", default="input_prompts.png")
    return parser


def find_data_paths(path):
    if path.is_file():
        return [path]
    if (path / "data.json").exists():
        return [path / "data.json"]
    return sorted(path.glob("*/data.json"))


def output_for(data_path, output_dir, output_name):
    if output_dir:
        return output_dir / f"{data_path.parent.name}_{output_name}"
    return data_path.parent / output_name


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def render_input_prompts(data):
    prompts = data.get("prompts", {})
    height, width = canvas_shape(data)
    image = blank_image(width, height)
    mask_pixels = draw_mask(image, prompts.get("mask"))
    if prompts.get("box"):
        draw_rect(image, prompts["box"], BOX_COLOR)
    for point, label in zip(prompts.get("points", []), prompts.get("labels", [])):
        draw_point(image, point, POSITIVE_COLOR if label else NEGATIVE_COLOR)
    return image, {
        "points": len(prompts.get("points", [])),
        "box": bool(prompts.get("box")),
        "mask_pixels": mask_pixels,
    }


def canvas_shape(data):
    prompts = data.get("prompts", {})
    mask = prompts.get("mask")
    if mask:
        return shape(mask)
    response_shape = data.get("response", {}).get("shape")
    if response_shape:
        return response_shape
    points = prompts.get("points", [])
    box = prompts.get("box")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if box:
        xs.extend([box[0], box[2]])
        ys.extend([box[1], box[3]])
    return int(max(ys, default=255)) + 16, int(max(xs, default=255)) + 16


def shape(mask):
    height, width = mask["shape"]
    return int(height), int(width)


def blank_image(width, height):
    return [[(0, 0, 0) for _ in range(width)] for _ in range(height)]


def draw_mask(image, mask):
    if not mask:
        return 0
    height, width = shape(mask)
    mask_bytes = b64decode(mask["data"], validate=True)
    if len(mask_bytes) != height * width:
        raise ValueError(f"Mask byte count {len(mask_bytes)} does not match shape {height}x{width}")
    count = 0
    for y in range(min(height, len(image))):
        row = image[y]
        for x in range(min(width, len(row))):
            if mask_bytes[y * width + x]:
                row[x] = MASK_COLOR
                count += 1
    return count


def draw_rect(image, box, color):
    height, width = len(image), len(image[0])
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    x0, x1 = sorted((clip(x0, 0, width - 1), clip(x1, 0, width - 1)))
    y0, y1 = sorted((clip(y0, 0, height - 1), clip(y1, 0, height - 1)))
    for offset in range(2):
        image[clip(y0 + offset, 0, height - 1)][x0:x1 + 1] = [color] * (x1 - x0 + 1)
        image[clip(y1 - offset, 0, height - 1)][x0:x1 + 1] = [color] * (x1 - x0 + 1)
        for y in range(y0, y1 + 1):
            image[y][clip(x0 + offset, 0, width - 1)] = color
            image[y][clip(x1 - offset, 0, width - 1)] = color


def draw_point(image, point, color):
    height, width = len(image), len(image[0])
    x, y = [int(round(value)) for value in point]
    radius = 5
    for row in range(max(0, y - radius), min(height, y + radius + 1)):
        for column in range(max(0, x - radius), min(width, x + radius + 1)):
            if (row - y) ** 2 + (column - x) ** 2 <= radius ** 2:
                image[row][column] = color


def clip(value, low, high):
    return max(low, min(high, value))


def write_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = len(image), len(image[0])
    raw = b"".join(bytes([0]) + b"".join(bytes(pixel) for pixel in row) for row in image)
    payload = chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", compress(raw))
    payload += chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + payload)


def chunk(kind, data):
    return pack(">I", len(data)) + kind + data + pack(">I", crc32(kind + data) & 0xFFFFFFFF)


if __name__ == "__main__":
    main()
