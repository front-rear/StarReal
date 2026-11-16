from pathlib import Path
import argparse,json
import numpy as np
from .core import render_pair,mesh_supports,export_control_fields
def main():
    parser=argparse.ArgumentParser(description="Frozen StarReal core, explicit inputs, no automatic upload")
    sub=parser.add_subparsers(dest="command",required=True)
    render=sub.add_parser("render")
    render.add_argument("--scene",type=Path,required=True)
    render.add_argument("--out",type=Path,required=True)
    masks=sub.add_parser("mesh-masks")
    masks.add_argument("--mesh",type=Path,required=True)
    masks.add_argument("--camera",type=Path,required=True)
    masks.add_argument("--operation",required=True,choices=["insert","remove","move","replace","move_replace"])
    masks.add_argument("--view",choices=["A","B"],default="B")
    masks.add_argument("--source-color-linear",type=float,nargs=3)
    masks.add_argument("--frame",type=int,default=0)
    masks.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="render":
        import cv2
        scene=json.loads(args.scene.read_text())
        old,new,aux=render_pair(np.array(scene["K"]),np.array(scene["w2c"]),
            scene["boxes"],width=scene.get("width",1280),height=scene.get("height",704),
            aa=scene.get("aa",2),first_w2c_A=np.array(scene.get("first_w2c_A",np.eye(4).tolist())))
        args.out.mkdir(parents=True,exist_ok=False)
        for name,image in [("ordinary",old),("depth_aware",new)]:
            assert cv2.imwrite(str(args.out/(name+".png")),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        np.savez_compressed(args.out/"depth_evidence.npz",**aux)
        print(json.dumps(dict(out=str(args.out),world_unit=scene.get("world_unit","relative estimated units"),GPU_used=False)))
    else:
        with np.load(args.mesh,allow_pickle=False) as mesh,np.load(args.camera,allow_pickle=False) as camera:
            f=args.frame;K=camera["K"][f];E=camera["w2c_"+args.view][f]
            supports=mesh_supports(mesh["vertices"],mesh["faces"],mesh["rotation"],K,E)
            rows=export_control_fields(supports,args.out,args.operation,frame=f,source_color_linear=args.source_color_linear)
        print(json.dumps(rows))
if __name__=="__main__":main()
