"""Independent readers and actual palette review artifacts, not acceptance."""
from pathlib import Path
import hashlib,json,sys,time
R=Path('/tmp/starreal_uid1007_case19_cpu_v1')
import os
CASE=os.environ['STARREAL_CASE'];assert CASE.isdecimal()
sys.path[:0]=[str(R/'runtime/qh_cpu_py39_v1'),str(R/'runtime/qh_pcf_cpu_v1'),str(R/'runtime/qh_ffmpeg_py39_v1'),str(R/'runtime/case18_ordinary_refs_v1'),str(R/'code')]
import cv2,numpy as np
from realcase_condition_reader import RealcaseConditionReader
from original_joint_frame_stream_v1 import OriginalJointStream
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cv2.setNumThreads(1);start=time.time();c=R/f'outputs/{CASE}/geometry_masks49_cached_QH_v1';j=R/f'outputs/{CASE}/two_pcf49_fixed_world_yellow_QH_v1'
    out=R/f'outputs/{CASE}/actual_condition_palette_review_QH_v1';out.mkdir(exist_ok=False)
    m=json.loads((c/'MANIFEST.json').read_text());reader=RealcaseConditionReader(c,R/f'inputs/case{CASE}_source_video_HF_v1/source_A.mkv',j)
    joint_reader=OriginalJointStream(R/'runtime/case18_ordinary_refs_v1/CASE18_bbox_dataset_original_UNMODIFIED.py');rgbrecords=[]
    for name in m['operations']:
        expected=json.loads((j/name/'RGB_PIXEL_READBACK.json').read_text())['records'];book={(x['version'],x['frame']):x['RGB_SHA256'] for x in expected}
        for i in range(49):
            old=np.asarray(reader.joint_frame(name,'old_joint',i));np.testing.assert_array_equal(old,joint_reader.frame(j/name/'old_style',i))
            for version in ['old_joint','new_joint']:
                rgb=np.asarray(reader.joint_frame(name,version,i));digest=hashlib.sha256(rgb.tobytes()).hexdigest();assert rgb.shape==(704,1280,3) and digest==book[(version,i)]
                rgbrecords.append(dict(variant=name,version=version,frame=i,RGB_SHA256=digest))
        print(json.dumps(dict(variant=name,old_and_new_RGB49_read=True,original_ordinary_joint_math_exact=True)),flush=True)
    sheets=[];colors=list(m['colors_linear_rgb'])
    for family in ['lamp_silhouette','bbox_silhouette']:
        for frame in [0,24,48]:
            canvas=np.full((5*114,8*160,3),20,np.uint8)
            for row,(name,meta) in enumerate(m['operations'].items()):
                level=next(iter(m['gamma_control_pairs_by_operation'][meta['operation']]))
                for col,color in enumerate(colors):
                    v=reader.condition_frame(name,family,frame,color,level)
                    rgb=np.asarray(v['color_mask_rgb']);support=v['support']
                    assert np.array_equal(np.any(rgb>0,axis=2),support)
                    thumb=cv2.cvtColor(cv2.resize(rgb,(160,88),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
                    y,x=row*114,col*160;canvas[y+26:y+114,x:x+160]=thumb
                    cv2.putText(canvas,name+' '+color,(x+2,y+10),cv2.FONT_HERSHEY_SIMPLEX,.24,(255,255,255),1)
                    cv2.putText(canvas,f'{family} B{frame:02d}',(x+2,y+22),cv2.FONT_HERSHEY_SIMPLEX,.23,(255,255,255),1)
            p=out/f'{family}_eight_colors_B{frame:02d}.jpg';assert cv2.imwrite(str(p),canvas,[cv2.IMWRITE_JPEG_QUALITY,96]);sheets.append(dict(path=str(p.relative_to(R)),SHA256=sha(p)))
    # Source appearance hints use unchanged RGB; this is not measured emission.
    for family in ['lamp_silhouette','bbox_silhouette']:
        rows=[reader.condition_frame('move',family,i,'source_hint','unchanged') for i in range(49)]
        assert all(x['normalized_gamma_for_model']==0 for x in rows)
    result=dict(case=CASE,state='all490_joint_RGB_pixels_original_ordinary_reference_exact_and_palette_artifacts_ready_for_actual_review',joint_RGB_frames_fully_read=len(rgbrecords),original_joint_streaming=True,Original_file_SHA256=joint_reader.original_file_SHA256,records=rgbrecords,review_sheets=sheets,source_hint_both_families49_read=True,quality_accepted=False,GPU_used=False,elapsed_seconds=time.time()-start)
    (R/f'receipts/CASE{CASE}_JOINT_RGB_AND_PALETTE_INDEPENDENT_READBACK_v1.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
