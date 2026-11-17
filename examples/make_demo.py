"""Regenerate the small disconnected-stroke demo image used in the README."""

from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    image = Image.new("L", (320, 320), 255)
    draw = ImageDraw.Draw(image)
    ink = 22
    width = 6
    draw.ellipse((42, 30, 278, 266), outline=ink, width=width)
    draw.ellipse((101, 104, 119, 122), outline=ink, width=width)
    draw.ellipse((201, 104, 219, 122), outline=ink, width=width)
    draw.arc((102, 133, 218, 225), start=22, end=158, fill=ink, width=width)
    target = Path(__file__).with_name("demo_smiley.png")
    image.save(target)
    print(target)


if __name__ == "__main__":
    main()
