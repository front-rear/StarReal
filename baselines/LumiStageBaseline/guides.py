"""Build a review sheet from the exact conditions and project P/Q in the original annotation canvas."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from common import bound, request, write, sha
from geometry import project


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("request", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    m = request(a.request)
    base = a.request.resolve().parent
    projection = project(m, base)
    a.output.mkdir(parents=True, exist_ok=False)
    write(a.output / "projections.json", projection)
    tiles = []
    for ref in m["condition_images"]:
        with Image.open(bound(ref,base)) as im:
            tiles.append((ref["role"], ImageOps.contain(im.convert("RGB"), (400,216))))
    for i in (0,24,48):
        canvas = Image.new("RGB",(400,216),"#232323")
        mask_p = Image.new("1",canvas.size)
        mask_q = mask_p.copy()
        for role, mask in (("P",mask_p),("Q",mask_q)):
            box = projection["B"].get(role, [None]*49)[i]
            if box:
                w,h = m.get("annotation_canvas_wh", [1280,704])
                bounds = tuple(round(x*y) for x,y in zip(box,[400/w,216/h]*2))
                ImageDraw.Draw(mask).rectangle(bounds,fill=1)
        canvas.paste((218,49,49),mask=mask_p)
        canvas.paste((35,97,246),mask=mask_q)
        from PIL import ImageChops
        canvas.paste((247,232,29),mask=ImageChops.logical_and(mask_p,mask_q))
        tiles.append((f"B target / frame {i}",canvas))
    board = Image.new("RGB",(800,244*((len(tiles)+1)//2)),"#202020")
    draw = ImageDraw.Draw(board)
    for n,(title,tile) in enumerate(tiles):
        x,y=(n%2)*400,(n//2)*244
        board.paste(tile,(x+(400-tile.width)//2,y+26));draw.text((x+4,y+5),title[:62],fill="white")
    board.save(a.output / "conditions_and_B.jpg",quality=92)
    write(a.output / "binding.json",dict(request_sha256=sha(a.request),conditions=m["condition_images"],
          board_sha256=sha(a.output / "conditions_and_B.jpg"),reviewed=False))


if __name__ == "__main__":
    main()
