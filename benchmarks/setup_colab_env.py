import argparse
import subprocess, os, sys

def run(cmd, check=True):
    print(f"--> Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    res = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)
    if res.stdout:
        print(res.stdout[-500:])
    if res.stderr:
        print("STDERR:", res.stderr[-500:])
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}")
    return res

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="/content/PyRPMD")
parser.add_argument("--install-ml", action="store_true",
                    help="also install optional MACE/DeepMD benchmark packages")
args = parser.parse_args()
repo = os.path.abspath(args.repo)

print("=== 1. Check GPU and CUDA runtime ===")
run("nvidia-smi")

print("=== 2. Clone / Check PyRPMD ===")
if not os.path.exists(repo):
    raise RuntimeError(f"repository does not exist: {repo}; upload or clone it first")

print("=== 3. Install PyRPMD ===")
run([sys.executable, "-m", "pip", "install", "-e", repo])

print("=== 4. Install ML Packages via uv ===")
run([sys.executable, "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
     "-r", os.path.join(repo, "tools/requirements-jaxpip-gpu.txt")])
if args.install_ml:
    run([sys.executable, "-m", "pip", "install", "ase", "mace-torch", "deepmd-kit",
         "vesin[torch]"])

print("=== 5. Optional ONNX/ML setup ===")
if args.install_ml:
    run("apt-get update -y && apt-get install -y nlohmann-json3-dev cmake build-essential")
if args.install_ml and not os.path.exists("/content/onnxruntime-sdk"):
    url = "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-gpu-1.20.1.tgz"
    run(f"curl -sL {url} -o /content/ort.tgz")
    os.makedirs("/content/onnxruntime-sdk", exist_ok=True)
    run("tar -xzf /content/ort.tgz -C /content/onnxruntime-sdk --strip-components=1")

if args.install_ml:
    os.makedirs(os.path.join(repo, "build/onnx"), exist_ok=True)
    run(["cmake", "-S", os.path.join(repo, "native/onnx"), "-B", os.path.join(repo, "build/onnx"),
         "-DCMAKE_BUILD_TYPE=Release", "-DONNXRUNTIME_ROOT=/content/onnxruntime-sdk"])
    run(["cmake", "--build", os.path.join(repo, "build/onnx"), "--parallel", "2"])

print("=== 6. Patch DeepMD cxx_op.py for compatibility ===")
if args.install_ml:
    cxx_file = "/usr/local/lib/python3.13/dist-packages/deepmd/pt/cxx_op.py"
else:
    cxx_file = None
if cxx_file and os.path.exists(cxx_file):
    with open(cxx_file, "r") as f:
        c = f.read()
    if 'ENABLE_CUSTOMIZED_OP = load_library("deepmd_op_pt")' in c:
        c = c.replace(
            'ENABLE_CUSTOMIZED_OP = load_library("deepmd_op_pt")',
            'try:\n    ENABLE_CUSTOMIZED_OP = load_library("deepmd_op_pt")\nexcept Exception:\n    ENABLE_CUSTOMIZED_OP = False'
        )
        with open(cxx_file, "w") as f:
            f.write(c)
        print("Patched deepmd cxx_op.py successfully!")

print("=== 7. Download MACE Models ===")
if args.install_ml:
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device="cuda")
    print("MACE-MP medium ready!")

print("=== SETUP COMPLETE! ===")
