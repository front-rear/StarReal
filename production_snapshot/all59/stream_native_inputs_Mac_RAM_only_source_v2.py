"""Finite READ-ONLY SFTP via native SSH; stream original inputs to QH, no Mac payload files."""
from pathlib import Path
import argparse, hashlib, json, os, signal, struct, subprocess, sys, time

S = Path('/Volumes/T9/LampAssetPipeline_handoff/StarReal')
Q = '/mnt/atlas/fangrui/LampEdit/native_project_production_20261027_v1/results/new50_sig_256_v1/root_coordinator/StarReal'
U = Q + '/all59_reconstruction_20261115_v1'
pack = lambda n: struct.pack('>I', n)
string = lambda b: pack(len(b)) + b

class ReadOnlySFTP:
    def __init__(self):
        self.p = subprocess.Popen(['ssh', '-oBatchMode=yes', '-oConnectTimeout=20', '-oServerAliveInterval=30', '-oServerAliveCountMax=3', '-s', '6000D', 'sftp'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        self.seq = 0
        self.send(b'\x01' + pack(3))
        reply = self.recv()
        assert reply[0] == 2 and struct.unpack('>I', reply[1:5])[0] == 3

    def send(self, b):
        self.p.stdin.write(pack(len(b)) + b)
        self.p.stdin.flush()

    def exact(self, n):
        parts = []
        while n:
            b = self.p.stdout.read(n)
            if not b: raise EOFError('Read-only SFTP channel ended')
            parts.append(b); n -= len(b)
        return b''.join(parts)

    def recv(self):
        n = struct.unpack('>I', self.exact(4))[0]
        assert 1 <= n <= 2 * 1024**2
        return self.exact(n)

    def request(self, kind, body):
        self.seq += 1; self.send(bytes([kind]) + pack(self.seq) + body)
        return self.seq

    def response(self, req, kind):
        b = self.recv(); assert struct.unpack('>I', b[1:5])[0] == req
        if b[0] == 101:
            code = struct.unpack('>I', b[5:9])[0]
            assert kind == 101 and code == 0, ('Read-only SFTP status', code)
        else: assert b[0] == kind, ('Unexpected SFTP response', b[0], kind)
        return b[5:]

    def stat(self, handle):
        b = self.response(self.request(8, string(handle)), 105)
        flags = struct.unpack('>I', b[:4])[0]; offset = 4; d = {}
        if flags & 1: d['size'] = struct.unpack('>Q', b[offset:offset+8])[0]; offset += 8
        if flags & 2: offset += 8
        if flags & 4: d['permissions'] = struct.unpack('>I', b[offset:offset+4])[0]; offset += 4
        if flags & 8: d['atime'], d['mtime'] = struct.unpack('>II', b[offset:offset+8])
        assert 'size' in d
        return d

    def open(self, path):
        # SSH_FXF_READ=1 only; never CREATE/WRITE/APPEND/TRUNCATE.
        b = self.response(self.request(3, string(path.encode()) + pack(1) + pack(0)), 102)
        n = struct.unpack('>I', b[:4])[0]; return b[4:4+n]

    def blocks(self, handle, size):
        offset = 0
        while offset < size:
            requests = []
            for _ in range(16):
                pos = offset + sum(n for _, n in requests)
                if pos >= size: break
                n = min(65536, size-pos)
                requests.append((self.request(5, string(handle) + struct.pack('>Q', pos) + pack(n)), n))
            replies = {}
            for _ in requests:
                b = self.recv(); assert b[0] == 103, ('Missing original input bytes', b[0])
                req, n = struct.unpack('>II', b[1:9]); data = b[9:]; assert len(data) == n
                replies[req] = data
            for req, n in requests:
                data = replies[req]; assert len(data) == n
                yield data; offset += n

    def close(self, handle):
        self.response(self.request(4, string(handle)), 101)

    def finish(self):
        self.p.stdin.close(); assert self.p.wait(timeout=30) == 0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cases', nargs='+', required=True); a = ap.parse_args()
    assert 1 <= len(a.cases) <= 2 and all(c.isdecimal() for c in a.cases)
    signal.alarm(1800); start = time.time()
    board = json.loads((S/'MASTER_WORKBOARD.json').read_text()); specs = []
    base = '/root/autodl-tmp/kimi_realcase_20261109'
    for case in a.cases:
        row = next(x for x in board['cases'] if str(x['case']) == case); assert not row.get('full_case_accepted', False)
        v = row['versions'][0]; cid = v['case_id']
        pairs = [(v['selected_local_source'], 'source_A.mkv'), (v['version_root']+'/orbit_10deg/cameras_and_boxes.npz', 'cameras.npz'), (v['version_root']+'/geometry_plan.json', 'HISTORICAL_GEOMETRY_PLAN.json'), (v['version_root']+'/manifest.json', 'HISTORICAL_MANIFEST.json'), (base+'/samples/'+cid+'/step3_worldpoints/vggt_raw.npz', 'vggt_raw.npz'), ('/root/autodl-tmp/StarReal/code/real_geometry.py', 'real_geometry.py'), ('/root/autodl-tmp/StarReal/code/mesh_zbuffer.so', 'mesh_zbuffer.so')]
        for i in range(49):
            pairs += [(base+'/cases/'+cid+'/frames/'+f'{i:03d}.png', f'original_frames49/{i:06d}.png'), (base+'/samples/'+cid+'/step8_sam2_candidate/mask49/'+f'{i:06d}.png', f'original_mask49/{i:06d}.png')]
        specs.append(dict(case=case, version=v, read_only_original_paths=[dict(remote=r, local=l) for r,l in pairs]))
    name = '_'.join(a.cases); ctl = Path('/private/tmp/StarReal_stream_control_v2')/name; ctl.mkdir(parents=True, exist_ok=False)
    spec = dict(cases=specs, remote_execution=False, remote_writes=False, Mac_payload_files=False)
    (ctl/'SPEC.json').write_text(json.dumps(spec, indent=2))
    receiver = subprocess.Popen(['ssh', 'cscg-qh00', 'python3', U+'/receive_readonly_SFTP_stream_QH_v2.py', '--name', name], stdin=subprocess.PIPE, stdout=(ctl/'QH_RECEIVER.log').open('xb'), stderr=subprocess.STDOUT)
    def emit(x):
        b = json.dumps(x).encode(); receiver.stdin.write(pack(len(b))+b); receiver.stdin.flush()
    emit(spec); client = ReadOnlySFTP(); records = []
    try:
        for cfg in specs:
            for row in cfg['read_only_original_paths']:
                handle = client.open(row['remote']); before = client.stat(handle)
                emit(dict(path=cfg['case']+'/'+row['local'], bytes=before['size'], source_stat_before=before))
                h = hashlib.sha256()
                for b in client.blocks(handle, before['size']): receiver.stdin.write(b); h.update(b)
                after = client.stat(handle); client.close(handle)
                assert before['size'] == after['size'] and before.get('mtime') == after.get('mtime'), 'Original changed while reading'
                emit(dict(SHA256=h.hexdigest(), source_stat_after=after))
                records.append(dict(case=cfg['case'], path=row['local'], bytes=before['size'], SHA256=h.hexdigest()))
                if len(records)%10 == 0:
                    (ctl/'CURRENT.json').write_text(json.dumps(dict(state='streaming_read_only_originals_to_QH',files=len(records),bytes=sum(x['bytes'] for x in records),Mac_payload_files=False,seconds=time.time()-start),indent=2))
        client.finish(); receiver.stdin.close(); assert receiver.wait(timeout=60) == 0
        result = dict(state='all_original_inputs_streamed_QH_pending_array_RGB_verification',files=len(records),bytes=sum(x['bytes'] for x in records),records=records,seconds=time.time()-start,Mac_payload_files=False,source_remote_execution=False,source_remote_writes=False)
        (ctl/'TRANSFER_RESULT.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    finally:
        if client.p.poll() is None: client.p.stdin.close()
        if receiver.poll() is None: receiver.stdin.close()
    transfer='/tmp/starreal_uid1007_case19_cpu_v1/inputs/StarReal_native_readonly_SFTP_'+name+'_stream_v2'
    with (ctl/'QH_ARRAY_READBACK.log').open('xb') as f:
        r=subprocess.run(['ssh','cscg-qh00','python3',U+'/prepare_native_from_readonly_SFTP_retained_runtime_v2.py','--transfer',transfer,'--spec',transfer+'/SPEC.json','--cases']+a.cases,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    assert r.returncode==0, 'Exact transferred inputs preserved; inspect QH_ARRAY_READBACK.log'
    (ctl/'FINAL.json').write_text(json.dumps(dict(state='exact_Native_inputs_verified_on_QH',cases=a.cases,Mac_payload_files=False,source_remote_execution=False,source_remote_writes=False,seconds=time.time()-start),indent=2));print('QH full Native RGB/camera/depth readback passed',flush=True)

if __name__ == '__main__': main()
