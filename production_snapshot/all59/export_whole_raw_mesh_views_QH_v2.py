"""Read-only CPU GLB decoding bridge on authorized Qinghai Python3.12 runtime.

Input/output archives live in process memory; no raw payload extracted on any server.
Whole scene graph transforms are applied by unchanged installed trimesh.
"""
from pathlib import Path
import hashlib,io,json,sys,zipfile
N=Path('/tmp/starreal_uid1007_case19_cpu_v1');runtime=N/'runtime/sam2_private_cpu_py312_v1';rt=json.loads((runtime/'RESULT.json').read_text());sys.path[:0]=[str(N/'inputs/case74_native_CPU_v1'),rt['dependencies_root']]
import cv2,numpy as np,trimesh
from real_geometry import Rasterizer
def camera(az):
    eye=np.array([4*np.sin(az),-4*np.cos(az),1.7]);forward=-eye/np.linalg.norm(eye);right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right);E=np.eye(4);E[:3,:3]=np.stack([right,down,forward]);E[:3,3]=-E[:3,:3]@eye;return E
def main():
    cv2.setNumThreads(1);payload=sys.stdin.buffer.read(32*1024**2+1);assert len(payload)<=32*1024**2
    inp=zipfile.ZipFile(io.BytesIO(payload));proof=json.loads(inp.read('ORIGINAL_LIBRARY_RAW_CANDIDATES_PROOF.json'));assert proof['schema']=='original_user_lamp_library_exact_raw_candidates_v1' and proof['qualified500_claimed'] is False
    result=io.BytesIO();raster=Rasterizer();K=np.array([[650.,0.,256.],[0.,650.,256.],[0.,0.,1.]]);records=[]
    with zipfile.ZipFile(result,'w',compression=zipfile.ZIP_DEFLATED) as out:
        for row in proof['actual_exact_raw_SHA_matched']:
            ix=row['source_index'];raw=inp.read('raw_assets/lamp_'+ix+'.glb');assert hashlib.sha256(raw).hexdigest()==row['actual_raw_SHA256']
            scene=trimesh.load(io.BytesIO(raw),file_type='glb',force='scene');mesh=scene.to_geometry();v=np.asarray(mesh.vertices,float);v=np.stack([v[:,0],-v[:,2],v[:,1]],axis=1);f=np.asarray(mesh.faces,np.int32)
            b=io.BytesIO();np.savez_compressed(b,vertices=v,faces=f);out.writestr('mesh_arrays/'+ix+'.npz',b.getvalue())
            tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2;normal=cross/np.maximum(2*area[:,None],1e-15);span=np.ptp(v[:,2]);flat=(np.abs(normal[:,2])>.99999)&(np.ptp(tri[:,:,2],axis=1)<max(span*1e-7,1e-9))&(tri[:,:,2].mean(1)<v[:,2].min()+.28*span)&(area>0);levels=np.round(tri[:,:,2].mean(1),6);planes=[]
            for z in np.unique(levels[flat]):
                mask=flat&(levels==z);planes.append(dict(z=float(z),faces=int(mask.sum()),area=float(area[mask].sum()),centroid=np.average(tri[mask].mean(1),axis=0,weights=area[mask]).tolist(),xy_bounds=[tri[mask].reshape(-1,3)[:,:2].min(0).tolist(),tri[mask].reshape(-1,3)[:,:2].max(0).tolist()]))
            out.writestr('mount_diagnostics/'+ix+'.json',json.dumps(dict(index=ix,whole_mesh_not_modified=True,planes=sorted(planes,key=lambda x:-x['area'])[:30]),indent=2))
            lo,hi=v.min(0),v.max(0);vn=(v-(lo+hi)/2)*2/max(hi-lo);canvas=np.full((550,1536,3),25,np.uint8)
            for j,az in enumerate([0.,np.pi/2,np.pi/4]):
                dep=raster.depth(vn,f,K,camera(az),width=512,height=512);hit=np.isfinite(dep);image=np.full((512,512,3),25,np.uint8)
                if hit.any():values=dep[hit];t=(values-values.min())/max(float(np.ptp(values)),1e-9);image[hit]=np.rint((220-110*t)[:,None]).astype(np.uint8)
                canvas[38:,j*512:(j+1)*512]=image
            cv2.putText(canvas,f'Original lamp library {ix}: WHOLE raw GLB front/side/oblique; no cropping',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);ok,jpg=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok;out.writestr('actual_mesh_views/'+ix+'.jpg',jpg.tobytes())
            assert np.isfinite(v).all() and len(v)>0 and len(f)>0 and f.min()>=0 and f.max()<len(v)
            records.append(dict(index=ix,role=row['role'],raw_SHA256=row['actual_raw_SHA256'],vertices=len(v),faces=len(f),converted_raw_bounds=[lo.tolist(),hi.tolist()],geometry_nodes=len(scene.graph.nodes_geometry),all_scene_nodes_preserved=True))
        out.writestr('MESH_PARSE_PROOF.json',json.dumps(dict(case=proof['case'],records=records,trimesh_version=trimesh.__version__,source_input_SHA256=hashlib.sha256(payload).hexdigest(),raw_payload_bytes_written_to_disk=0,GPU_used=False),indent=2))
    assert result.tell()<32*1024**2;sys.stdout.buffer.write(result.getvalue())
if __name__=='__main__':main()
