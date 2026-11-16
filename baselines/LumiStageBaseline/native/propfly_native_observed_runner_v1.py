"""Outer observer around the unchanged original full49 wrapper and author demo."""
import argparse,hashlib,json,os,runpy,sys
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(add_help=False)
 parser.add_argument('--observer-manifest',required=True)
 parser.add_argument('--observer-manifest-sha',required=True)
 x,args=parser.parse_known_args()
 if sha(x.observer_manifest)!=x.observer_manifest_sha:raise ValueError('Observed PropFly manifest pin differs')
 m=json.loads(Path(x.observer_manifest).read_text())
 for z in m['observer']['files']:
  if sha(z['path'])!=z['SHA256']:raise ValueError('Observed runtime helper source pin differs')
 for z in m['propfly']['native_source_files']:
  if sha(z['path'])!=z['SHA256']:raise ValueError('Original PropFly source/writer differs')
 from diffusers import WanVACETransformer3DModel,AutoencoderKLWan
 from transformers import UMT5EncoderModel
 from pipeline.pipeline_vace_video import WanVACEPipeline
 from qwen21_propfly_native_load_observer_v1 import NativeLoadObserver
 out=Path(args[args.index('--output_dir')+1])
 classes={'transformer':WanVACETransformer3DModel,'vae':AutoencoderKLWan,'text_encoder':UMT5EncoderModel}
 saved_argv=sys.argv
 try:
  sys.argv=[m['propfly']['full49_entry']]+args
  with NativeLoadObserver(m,out,'propfly',classes,WanVACEPipeline):
   runpy.run_path(m['propfly']['full49_entry'],run_name='__main__')
 finally:sys.argv=saved_argv
if __name__=='__main__':main()
