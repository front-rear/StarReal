#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import triton as tr
import triton.language as tl
import triton.language.extra.cuda.libdevice as lib
from scipy.spatial.transform import Rotation, Slerp


@tr.jit
def field(x, y, z, ball, rod):
    # Nodes at integer grid coordinates; no finite array or moving window.
    qx = x - tl.floor(x + 0.5)
    qy = y - tl.floor(y + 0.5)
    qz = z - tl.floor(z + 0.5)
    sx, sy, sz = qx*qx, qy*qy, qz*qz
    ds = tl.sqrt(sx + sy + sz) - ball
    dxy = tl.sqrt(sx + sy) - rod
    dxz = tl.sqrt(sx + sz) - rod
    dyz = tl.sqrt(sy + sz) - rod
    return tl.minimum(ds, tl.minimum(dxy, tl.minimum(dxz, dyz)))


@tr.jit
def render_kernel(CAM, OUT, DEPTH, W: tl.constexpr, H: tl.constexpr,
                  BALL: tl.constexpr, ROD: tl.constexpr, FAR: tl.constexpr,
                  STEPS: tl.constexpr, BLOCK: tl.constexpr,
                  DIRECTION_RGB: tl.constexpr, BACKGROUND_ONLY: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    valid = i < W*H
    px = (i % W - tl.load(CAM+2)) / tl.load(CAM)
    py = (i // W - tl.load(CAM+3)) / tl.load(CAM+1)
    inv = tl.rsqrt(px*px + py*py + 1.)
    px, py, pz = px*inv, py*inv, inv
    dx = tl.load(CAM+4)*px + tl.load(CAM+5)*py + tl.load(CAM+6)*pz
    dy = tl.load(CAM+7)*px + tl.load(CAM+8)*py + tl.load(CAM+9)*pz
    dz = tl.load(CAM+10)*px + tl.load(CAM+11)*py + tl.load(CAM+12)*pz
    ox, oy, oz = tl.load(CAM+13), tl.load(CAM+14), tl.load(CAM+15)
    t = tl.full((BLOCK,), .002, tl.float32)
    hit = tl.full((BLOCK,), False, tl.int1)
    active = valid & (not BACKGROUND_ONLY)
    step = 0
    while (tl.sum(active.to(tl.int32), 0) > 0) & (step < STEPS):
        dist = field(ox+t*dx, oy+t*dy, oz+t*dz, BALL, ROD)
        # abs also handles an origin inside a scaffold solid: find the exit.
        new_hit = active & (tl.abs(dist) < .00015)
        hit = hit | new_hit
        active = active & ~new_hit
        t = tl.where(active, t + tl.maximum(tl.abs(dist)*.95, .000075), t)
        active = active & (t < FAR)
        step += 1
    x, y, z = ox+t*dx, oy+t*dy, oz+t*dz
    e = .0003
    nx = field(x+e,y,z,BALL,ROD)-field(x-e,y,z,BALL,ROD)
    ny = field(x,y+e,z,BALL,ROD)-field(x,y-e,z,BALL,ROD)
    nz = field(x,y,z+e,BALL,ROD)-field(x,y,z-e,BALL,ROD)
    nn = tl.rsqrt(nx*nx+ny*ny+nz*nz+1.e-12)
    diffuse = .55 + .45*tl.maximum(0., (nx*.36-ny*.80-nz*.48)*nn)
    qx, qy, qz = x-tl.floor(x+.5), y-tl.floor(y+.5), z-tl.floor(z+.5)
    sphere = tl.sqrt(qx*qx+qy*qy+qz*qz)-BALL < .0005
    # Directional hue map: independent of translation and scaffold spacing.
    lon = lib.atan2(dx, dz)
    lat = lib.asin(tl.minimum(1., tl.maximum(-1., -dy)))
    hue = lon/6.283185307179586+.5
    rr = tl.minimum(1., tl.maximum(0., tl.abs(hue*6.-3.)-1.))
    gg = tl.minimum(1., tl.maximum(0., 2.-tl.abs(hue*6.-2.)))
    bb = tl.minimum(1., tl.maximum(0., 2.-tl.abs(hue*6.-4.)))
    sat = .68 * (1.-.35*tl.abs(lat)/1.5707963267948966)
    val = .86 + .10*tl.cos(lat*2.)
    er, eg, eb = val*(1.-sat+sat*rr), val*(1.-sat+sat*gg), val*(1.-sat+sat*bb)
    if DIRECTION_RGB:
        # Continuous world-direction map, including at the poles. The base
        # RGB map distinguishes signed latitude; graticule is decorative.
        er, eg, eb = (dx+1.)*.5, (dy+1.)*.5, (dz+1.)*.5
    # Fine white angular grid matches the visual motif of Wonder Fig. 4.
    angle = 0.2617993877991494
    glon = tl.abs(lon/angle-tl.floor(lon/angle+.5))*angle*tl.cos(lat)
    glat = tl.abs(lat/angle-tl.floor(lat/angle+.5))*angle
    aa = .75/tl.load(CAM)
    grid = 1.-tl.minimum(1., tl.maximum(0., (tl.minimum(glon,glat)-aa)/(aa+1.e-7)))
    if DIRECTION_RGB:
        # Longitude is undefined at the poles: fade the angular overlay there.
        grid *= tl.minimum(1., tl.sqrt(dx*dx+dz*dz)/.1)
    er, eg, eb = er+(1.-er)*grid*.55, eg+(1.-eg)*grid*.55, eb+(1.-eb)*grid*.55
    cr = tl.where(sphere, .92, .26)*diffuse
    cg = tl.where(sphere, .28, .29)*diffuse
    cb = tl.where(sphere, .16, .31)*diffuse
    # Far attenuation reduces subpixel aliasing; world geometry remains fixed.
    alpha = tl.where(hit, tl.exp(-tl.maximum(t-2.,0.)/8.), 0.)
    cr, cg, cb = er*(1.-alpha)+cr*alpha, eg*(1.-alpha)+cg*alpha, eb*(1.-alpha)+cb*alpha
    tl.store(OUT+3*i, (255.*cr).to(tl.uint8), valid)
    tl.store(OUT+3*i+1, (255.*cg).to(tl.uint8), valid)
    tl.store(OUT+3*i+2, (255.*cb).to(tl.uint8), valid)
    tl.store(DEPTH+i, tl.where(hit,t,tl.where(active,-2.,-1.)),valid)


class Renderer:
    def __init__(self, width=832, height=480, aa=2, far=64., steps=4096, environment='direction_rgb'):
        self.width,self.height,self.aa,self.far,self.steps=width,height,aa,far,steps
        self.environment=environment
        self.rgb=torch.empty((height*aa,width*aa,3),device='cuda',dtype=torch.uint8)
        self.depth=torch.empty((height*aa,width*aa),device='cuda')

    def frame(self, c2w, K, spacing, radius=.05, rod=.008, background_only=False):
        # The initial camera lies at a cell center, for every spacing variant.
        # This phase is fixed for the whole trajectory, never recentered.
        # Reduce only the evaluation coordinate of the periodic renderer.
        # This is algebraically identical to the fixed infinite lattice, and
        # avoids losing precision in X-round(X). Keep actual memory addresses
        # unwrapped; this modulo is not a camera-following grid translation.
        center=np.remainder(c2w[:3,3].astype(np.float64)/spacing + .5, 1.)
        k=K.copy();k[:2,:]*=self.aa
        k[0,2]+=(self.aa-1)*.5;k[1,2]+=(self.aa-1)*.5
        packed=np.r_[k[0,0],k[1,1],k[0,2],k[1,2],c2w[:3,:3].ravel(),center].astype('float32')
        cam=torch.tensor(packed,device='cuda')
        w,h=self.width*self.aa,self.height*self.aa
        render_kernel[(tr.cdiv(w*h,128),)](cam,self.rgb,self.depth,w,h,radius,rod,self.far,self.steps,128,
                                         self.environment=='direction_rgb',background_only)
        rgb=self.rgb.cpu().numpy()
        if self.aa>1: rgb=cv2.resize(rgb,(self.width,self.height),interpolation=cv2.INTER_AREA)
        dep=self.depth[::self.aa,::self.aa].cpu().numpy()
        return rgb,dep


def make_trajectories(root, out, items=(4,10,15)):
    out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in (root/'test/000000/index.jsonl').read_text().splitlines()]
    paths=[]
    for item in items:
        row=next(x for x in rows if x['source_item_index']==item)
        d=np.load(root/row['cameras']); t=d['timestamps_us'].astype(float)/1e6;t-=t[0]
        assert t[-1]>=5., (item,t[-1])
        target=np.arange(81)/16.;poses=d['c2w'].astype(float)
        result=np.repeat(np.eye(4)[None],81,axis=0)
        result[:,:3,:3]=Slerp(t,Rotation.from_matrix(poses[:,:3,:3]))(target).as_matrix()
        for j in range(3):result[:,j,3]=np.interp(target,t,poses[:,j,3])
        # One rigid gauge transform only; preserve metric scale.
        result=np.linalg.inv(result[0])[None]@result
        K=np.stack([np.interp(target,t,d['intrinsics'][:,i,j]) for i in range(3) for j in range(3)],axis=-1).reshape(81,3,3)
        h,w=d['image_size_hw'];scale=max(832/w,480/h)
        cropx=(w*scale-832)/2; cropy=(h*scale-480)/2
        K[:,:2,:]*=scale
        K[:,0,2]+=(scale-1)*.5-cropx;K[:,1,2]+=(scale-1)*.5-cropy
        name=f're10k_{item:02d}_{row["source_key"]}'
        path=out/(name+'.npz')
        np.savez(path,c2w=result.astype('float32'),K=K.astype('float32'),times_s=target)
        meta={'source_camera_file':str(root/row['cameras']),'source_metadata':str(root/row['metadata']),
              'source_image_hw':d['image_size_hw'].tolist(),'resize_scale':scale,'crop_xy':[cropx,cropy],
              'units':'official ViPE metric estimate, no translation scale fitting','sampling':'first 5 seconds, rotation Slerp, translation/K linear interpolation at 16fps',
              'translation_path_m':float(np.linalg.norm(np.diff(result[:,:3,3],axis=0),axis=1).sum())}
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2));paths.append(path)
    K=np.repeat(np.array([[600,0,415.5],[0,600,239.5],[0,0,1]],dtype='float32')[None],81,axis=0)
    u=np.linspace(0,1,81)
    for name in ('translate_1m','rotate_60deg','closed_loop'):
        C=np.repeat(np.eye(4)[None],81,axis=0)
        if name=='translate_1m':C[:,0,3]=u
        elif name=='rotate_60deg':C[:,:3,:3]=Rotation.from_euler('y',60*u,degrees=True).as_matrix()
        else:
            C[:,0,3]=.4*np.sin(2*np.pi*u);C[:,2,3]=.3*(1-np.cos(2*np.pi*u))
            C[:,:3,:3]=Rotation.from_euler('y',20*np.sin(2*np.pi*u),degrees=True).as_matrix()
        path=out/(name+'.npz');np.savez(path,c2w=C.astype('float32'),K=K,times_s=u*5);paths.append(path)
    return paths


def writer(path,w=832,h=480):
    ffmpeg=shutil.which('ffmpeg') or '/opt/tiger/ss_bin/ffmpeg'
    return subprocess.Popen([ffmpeg,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}',
                             '-r','16','-i','-','-an','-c:v','libx264','-crf','17','-preset','fast',
                             '-threads','4','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    paths=make_trajectories(Path(args.annotations),out/'trajectories',args.scene_items)
    renderer=Renderer(aa=args.aa,far=args.far,steps=args.steps,environment=args.environment)
    tasks=[]
    if args.mode=='preview':
        for spacing in (.25,.5,1.,2.):
            rgb,dep=renderer.frame(np.eye(4),np.array([[600,0,415.5],[0,600,239.5],[0,0,1.]]),spacing,args.radius)
            cv2.imwrite(str(out/f'preview_{spacing}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            print(spacing,{'hit':float((dep>0).mean()),'unresolved':float((dep==-2).mean())},flush=True)
        return
    for path in paths[:len(args.scene_items)]:
        for spacing in (.25,.5,1.,2.):tasks.append((path,spacing,args.radius))
    for radius in args.radius_extras:tasks.append((paths[0],.5,radius))
    if not args.no_synthetic_videos:
        for path in paths[len(args.scene_items):]:tasks.append((path,.5,args.radius))
    (out/'videos').mkdir(exist_ok=True);(out/'stills').mkdir(exist_ok=True)
    summaries=[]
    for path,spacing,radius in tasks:
        name=f'{path.stem}_d{spacing:g}_r{radius:g}';video=out/'videos'/(name+'.mp4')
        d=np.load(path);stats=[];start=time.perf_counter();proc=writer(video)
        try:
            for t in range(81):
                rgb,dep=renderer.frame(d['c2w'][t],d['K'][t],spacing,radius)
                proc.stdin.write(rgb.tobytes())
                if t in (0,20,40,60,80):cv2.imwrite(str(out/'stills'/f'{name}_{t:03d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                stats.append({'hit':float((dep>0).mean()),'unresolved':float((dep==-2).mean()),
                              'beyond_4_cells':float((dep>4).mean()),'beyond_8_cells':float((dep>8).mean()),
                              'near_half_cell':float(((dep>0)&(dep<.5)).mean())})
        finally:proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
        summary={'name':name,'video':str(video),'trajectory':str(path),'spacing_m':spacing,'sphere_radius_over_spacing':radius,
                 'rod_radius_over_spacing':.008,'seconds':time.perf_counter()-start,
                 'mean_hit':float(np.mean([s['hit'] for s in stats])),
                 'max_unresolved':max(s['unresolved'] for s in stats),
                 'max_near_half_cell':max(s['near_half_cell'] for s in stats),
                 'mean_beyond_4_cells':float(np.mean([s['beyond_4_cells'] for s in stats])),
                 'mean_beyond_8_cells':float(np.mean([s['beyond_8_cells'] for s in stats]))}
        summaries.append(summary)
        (out/'metrics.json').write_text(json.dumps(summaries,indent=2))
        print(json.dumps(summary),flush=True)
    manifest={'worker':'1094200','shape':[81,480,832],'fps':16,'aa':args.aa,'far_cells':args.far,'max_steps':args.steps,
              'geometry':'infinite periodic sphere/cylinder union, finite ray distance','origin':'(-Delta/2,-Delta/2,-Delta/2), fixed per trajectory',
              'renderer':'custom CUDA Triton sphere tracing; not official Wonder OpenGL code','device':torch.cuda.get_device_name(),
              'environment_mode':args.environment,'base_radius_ratio':args.radius,
              'environment':('E(d)=(d+1)/2; 15 degree white graticule with polar fade' if args.environment=='direction_rgb'
                             else 'world longitude hue + latitude modulation + 15 degree white graticule'),
              'distance_fog':'exp(-max(ray_distance_cells-2,0)/8), blends into directional environment','variants':summaries}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--annotations',default='/mnt/hdfs/data_ssd/tianqi/re10k_annotations/vipe_dav3_v1')
    p.add_argument('--scene-items',type=int,nargs='+',default=[4,10,15])
    p.add_argument('--radius',type=float,default=.05)
    p.add_argument('--radius-extras',type=float,nargs='*',default=[])
    p.add_argument('--environment',choices=['direction_rgb','legacy_hsv'],default='direction_rgb')
    p.add_argument('--no-synthetic-videos',action='store_true')
    p.add_argument('--aa',type=int,default=2);p.add_argument('--far',type=float,default=64.)
    p.add_argument('--steps',type=int,default=4096);p.add_argument('--mode',choices=['preview','sweep'],default='sweep')
    run(p.parse_args())
