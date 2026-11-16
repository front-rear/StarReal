import subprocess
for path in ['/tmp/starreal_uid1007_case19_cpu_v1/code/build_native_cached_conditions49_QH_v3.py', '/tmp/starreal_uid1007_case19_cpu_v1/code/verify_native_conditions49_QH_v2.py', '/tmp/starreal_uid1007_case19_cpu_v1/code/prepare_native_source_appearance_reference_QH_v2.py']:
 r=subprocess.run(["python3","-u",path])
 if r.returncode:raise SystemExit(r.returncode)
