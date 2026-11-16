"""Build only the optional CPU kernels; no GPU or model download."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
jobs=[("frozen/render/pcf_trace_cpu.cpp","frozen/render/pcf_trace_cpu.so",["-fopenmp"]),
      ("frozen/geometry/mesh_zbuffer.cpp","frozen/geometry/mesh_zbuffer.so",[])]
for source,target,extra in jobs:
    subprocess.run(["g++","-O3","-std=c++17","-shared","-fPIC",*extra,str(ROOT/"src/starreal/_frozen"/Path(source).relative_to("frozen")),"-o",str(ROOT/"src/starreal/_frozen"/Path(target).relative_to("frozen"))],check=True)
    print(str((ROOT/"src/starreal/_frozen"/Path(target).relative_to("frozen")).relative_to(ROOT)))
