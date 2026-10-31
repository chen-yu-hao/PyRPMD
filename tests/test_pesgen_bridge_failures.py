"""Compile and exercise C ABI failure handling without an upstream PES build."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Caracal bridge is Linux-only")
def test_native_failed_context_is_safe_for_concurrent_callers(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    native = Path(__file__).resolve().parents[1] / "src/pyrpmd/pesgen/_native"
    (tmp_path / "model_data.h").write_text('''
#include <cstdint>
static const char FORCE_MODE[]="analytic", FORCE_STEP[]="1e-4";
static const int64_t NATOMS=2, TIMEOUT_MS=1;
static const uint64_t WORKER_FNV=0, WORKER_BYTES=0;
static const struct {const char* key; const char* value;} IDENTITIES[]={{"natoms","2"}};
static const struct {const char* name; const char* content;} MODEL_FILES[]={{"model.key",""}};
static const char METADATA[]="{}";
''')
    source = tmp_path / "failure.cpp"
    source.write_text('#include '+json.dumps(str(native / "bridge.cpp"))+'''
#include <atomic>
#include <thread>
int main() {
    // A disconnected worker causes a real transfer failure without a socket
    // server or an external Caracal binary. Call the exported ABI directly,
    // bypassing Python's outer provider lock.
    pyrpmd_pes_context_v1 context;
    std::atomic<int> ready{0}, failures{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    for (int thread=0; thread<8; ++thread) threads.emplace_back([&] {
        double q[6]={0,0,0,0,0,1.4}, energy[1], force[6];
        int32_t status[1];
        char message[256];
        pyrpmd_error_v1 error{message,256,0};
        pyrpmd_pes_input_v1 input{1,sizeof(input),1,1,1,2,q,6};
        pyrpmd_pes_output_v1 output{1,sizeof(output),energy,1,force,6,status,1,nullptr,0,0};
        ++ready;
        while (!go.load()) std::this_thread::yield();
        for (int i=0; i<64; ++i) {
            const int rc=pyrpmd_pes_evaluate_v1(&context,&input,&output,&error);
            if (rc!=PYRPMD_BACKEND_FAILURE_V1 || error.required_capacity<2 || !message[0])
                ++failures;
        }
    });
    while (ready.load()!=8) std::this_thread::yield();
    go=true;
    for (auto& thread:threads) thread.join();
    return failures.load()!=0 || !context.failed;
}
''')
    executable = tmp_path / "failure"
    subprocess.run([compiler, "-std=c++17", "-O2", "-pthread", "-I", str(tmp_path),
                    "-I", str(native), str(source), "-ldl", "-o", str(executable)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(executable)], check=True, timeout=30, capture_output=True, text=True)
