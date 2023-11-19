import sys
import re


def add_white_background_to_svg(svg_file_path):
    with open(svg_file_path, 'r') as file:
        svg_content = file.read()

    # SVGタグの後に白い背景の長方形を追加
    svg_content = re.sub(
        r'(<svg[^>]*>)', r'\1<rect width="100%" height="100%" fill="white"/>', svg_content, 1)

    with open(svg_file_path, 'w') as file:
        file.write(svg_content)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_bg_to_svg.py [SVG_FILE_PATH]")
        sys.exit(1)

    svg_file_path = sys.argv[1]
    add_white_background_to_svg(svg_file_path)
