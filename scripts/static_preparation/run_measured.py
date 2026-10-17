"""Run an official entrypoint unchanged, recording time and process GPU memory."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
import traceback
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--cwd',type=Path,required=True)
parser.add_argument('--module')
parser.add_argument('command',nargs=argparse.REMAINDER)
args=parser.parse_args()
command=args.command[1:] if args.command[:1]==['--'] else args.command
args.output=args.output.resolve(); args.output.parent.mkdir(parents=True,exist_ok=True)
os.chdir(args.cwd)
sys.path.insert(0,str(args.cwd.resolve()))
peak_process_mib=0
gpu_pid_resolution='container_pid_exact_match'
stopping=threading.Event()


def monitor():
    global peak_process_mib,gpu_pid_resolution
    while not stopping.is_set():
        try:
            result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            lines=[line for line in result.stdout.splitlines() if line.strip()]
            for line in lines:
                fields=[v.strip() for v in line.split(',')]
                if len(fields)==2 and fields[0]==str(os.getpid()):
                    peak_process_mib=max(peak_process_mib,int(fields[1]))
                elif len(fields)==2 and len(lines)==1:
                    # NVML reports host PID in this container. GPU jobs are serialized.
                    peak_process_mib=max(peak_process_mib,int(fields[1]))
                    gpu_pid_resolution='sole_GPU_compute_process_host_pid_namespace'
        except Exception:
            pass
        stopping.wait(1)


thread=threading.Thread(target=monitor,daemon=True)
thread.start()
torch.cuda.reset_peak_memory_stats()
started=time.time()
record=dict(status='running',pid=os.getpid(),started_unix=started,command=command,module=args.module,cwd=str(args.cwd),torch=torch.__version__,gpu=torch.cuda.get_device_name())
args.output.write_text(json.dumps(record,indent=2))
try:
    if args.module:
        sys.argv=[args.module,*command]
        runpy.run_module(args.module,run_name='__main__')
    else:
        sys.argv=command
        runpy.run_path(command[0],run_name='__main__')
    torch.cuda.synchronize()
    record['status']='complete'
except BaseException as error:
    record['status']='failed'
    record['error']=repr(error)
    record['traceback']=traceback.format_exc()
    raise
finally:
    record.update(elapsed_seconds=time.time()-started,peak_torch_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_torch_reserved_mib=torch.cuda.max_memory_reserved()/2**20,peak_process_gpu_mib=peak_process_mib or None,gpu_pid_resolution=gpu_pid_resolution)
    stopping.set();thread.join(timeout=6)
    args.output.write_text(json.dumps(record,indent=2))
