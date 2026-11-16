"""Export only exact delivery evidence after the server receipt exists."""
from pathlib import Path
import datetime,hashlib,json,os
N=Path('/tmp/starreal_uid1007_case19_cpu_v1')
Q=Path('/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal')
def main():
    pairs={}
    def add(source,dest):
        text=source.read_text();pairs[dest]=dict(text=text,sha256=hashlib.sha256(text.encode()).hexdigest())
    base='releases/37/revisions/fixed_world_yellow_depth_v5/'
    for name in ['RELEASE.json','HF_RECEIPT.json','README.txt']:add(Q/base/name,base+name)
    receipt=json.loads(pairs[base+'HF_RECEIPT.json']['text']);assert receipt['state']=='published_and_server_verified' and receipt['case']=='37'
    add(Q/'receipts/CASE37_DURABLE_ARCHIVE_RECEIPT_v1.json','logs/CASE37_DURABLE_QH_ARCHIVE_RECEIPT_v1.json')
    for name in ['MANIFEST.json','PORTABLE_FINAL_AUDIT.json','BUILD_STATE.json']:add(Q/'remote_package_metadata/37'/name,'remote_package_metadata/37/'+name)
    for name in ['CASE37_NATIVE_FAST5_SIGNED_ORIGINAL_ENTRY_READBACK_v1.json','CASE37_NATIVE_FAST5_SOURCE_HINT_ORIGINAL_ENTRY_READBACK_v1.json','CASE37_BOUNDED_MEMORY_ORIGINAL_ENTRY_STREAM_v1.json','CASE37_HISTORICAL_CAMERAS_AND_SOURCE49_READBACK_v1.json','CASE37_JOINT_RGB_AND_PALETTE_INDEPENDENT_READBACK_v1.json']:add(N/'receipts'/name,'logs/'+name)
    add(N/'outputs/37/geometry_masks49_cached_QH_v1/INDEPENDENT_CONDITION_READBACK.json','logs/CASE37_CONDITIONS_FULL_READBACK_v1.json')
    for name in ['RESULT.json','JOB.json','NATIVE_FULL_VS_SPARSE_ALL5_EQUALITY.json']:add(N/'outputs/37/two_pcf49_fixed_world_yellow_QH_v1'/name,'remote_component_metadata/37/DELIVERY_v1/'+name)
    add(N/'outputs/37/historical_joint_comparison49_QH_v1/MANIFEST.json','remote_component_metadata/37/DELIVERY_v1/HISTORICAL_MANIFEST.json')
    for name in ['CASE37_WHOLE_SOURCE_AND_THREE_TARGET_PILOT_ACTUAL_REVIEW_v1.json','CASE37_FIVE_ROLE_ALL49_GEOMETRY_ACTUAL_REVIEW_v1.json','CASE37_FIXED_WORLD_YELLOW_FINAL_ALL49_ACTUAL_REVIEW_v2.json','CASE37_HISTORICAL10_ALL49_ACTUAL_REVIEW_v1.json']:add(N/'reviews'/name,'reviews/'+name)
    add(N/'inputs/case37_three_fresh_whole_models_v2/ORIGINAL_LIBRARY_RAW_CANDIDATES_PROOF.json','provenance/CASE37_THREE_ORIGINAL_LIBRARY_RAW_HASH_MATCH_v1.json')
    checks=[]
    for rel in ['outputs/37/five_role_geometry49_QH_v1/JOB.json','outputs/37/geometry_masks49_cached_QH_v1/JOB.json','outputs/37/two_pcf49_fixed_world_yellow_QH_v1/JOB.json','packages/37/fixed_world_yellow_depth_v5_staging/BUILD_STATE.json']:
        job=json.loads((N/rel).read_text());assert job['exit_code']==0
        p=Path('/proc')/str(job['pid']);identity_live=None if p.exists() else False
        if p.exists() and job.get('startticks'):
            identity_live=p.stat().st_uid==job['uid'] and job['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and job['startticks']==(p/'stat').read_text().split(') ',1)[1].split()[19]
        checks.append(dict(host='cscg-qh00',job=rel,exit_code=0,pid=job['pid'],identity_live=identity_live))
    livepath=N/'receipts/CASE37_TERMINAL_CPU_LIVE_READBACK_v1.json';livepath.write_text(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,GPU_used=False),indent=2));add(livepath,'logs/'+livepath.name)
    out=Q/'exports/37/CANONICAL_METADATA_BUNDLE_v1.json';out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists();out.write_text(json.dumps(pairs,indent=2));print(json.dumps(dict(metadata_files=len(pairs),bytes=out.stat().st_size,SHA256=hashlib.sha256(out.read_bytes()).hexdigest(),path=str(out))),flush=True)
if __name__=='__main__':main()
