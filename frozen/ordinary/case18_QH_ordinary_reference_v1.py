"""Unmodified legacy bbox math and AST native704 joint adapter, local paths."""
from pathlib import Path
import ast,json,sys
import numpy as np
from PIL import Image
R=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/'runtime/case18_ordinary_refs_v1')]
from bbox import OrientedBoundingBox,render_bbox_conditions

def bbox_mask(center,size,rot,K,w2c):
    box=OrientedBoundingBox('box',center,size,rot,np.array([255,255,255],np.uint8))
    lines,_,mask,_=render_bbox_conditions([box],K,w2c,704,1280,edge_width=3,fill_alpha=.28)
    wire=((lines.max(-1)>100)&(mask>0)).astype(np.uint8)*255
    return mask,wire

def load_joint():
    class Native704(ast.NodeTransformer):
        def visit_Constant(self,node):
            if type(node.value) is int and node.value in (352,640):return ast.copy_location(ast.Constant(value={352:704,640:1280}[node.value]),node)
            return node
    p=R/'runtime/case18_ordinary_refs_v1/CASE18_bbox_dataset_original_UNMODIFIED.py'
    tree=Native704().visit(ast.parse(p.read_text()));ast.fix_missing_locations(tree)
    ns={'__name__':'local_original_joint_reader'};exec(compile(tree,str(p),'exec'),ns)
    class DatasetPath(type(Path())):
        def glob(self,pattern):
            return (p for p in super().glob(pattern) if not p.name.startswith('._'))
    ns['Path']=DatasetPath
    return ns['joint_frames']
