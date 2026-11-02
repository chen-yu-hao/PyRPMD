#!/usr/bin/env python3
"""Bounded, fixed odd/even queues for the registered eight classical chains.

Run at most one odd and one even queue concurrently. Start only after any
manual native_reference invocation for those chains has exited. Every child
has its own wall budget; failures stop the queue without replacing a seed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from benchmarks.water.campaign import sha256
from benchmarks.water.native_reference import Config,source_identity,verify_manifest


def committed_prefix(directory):
    manifest_path=directory/"manifest.json"
    if not manifest_path.exists():
        return {"committed_steps":0,"completed_NVE_segments":0,"chunk_count":0,
                "chunk_prefix_sha256":hashlib.sha256(b"[]").hexdigest(),"complete":False},[]
    raw=manifest_path.read_bytes();manifest=json.loads(raw)
    if not {"model.xml","integrator.xml"}.issubset(manifest.get("artifacts",{})):
        raise ValueError("serialized model/integrator identity records are required")
    verify_manifest(directory,manifest)
    config=manifest["config"]
    chunks=manifest["chunks"]
    encoded=json.dumps(chunks,sort_keys=True,separators=(",",":")).encode()
    return {"manifest_sha256":hashlib.sha256(raw).hexdigest(),
            "committed_steps":manifest["committed_steps"],"chunk_count":len(chunks),
            "completed_NVE_segments":max(0,manifest["committed_steps"]-config["equilibration_steps"])//config["segment_steps"],
            "chunk_prefix_sha256":hashlib.sha256(encoded).hexdigest(),
            "last_chunk":chunks[-1] if chunks else None,"complete":manifest["complete"]},chunks


def schedule(output,protocol_path,queue,*,wall_seconds=7200,invocation_seconds=1800):
    output,protocol_path=Path(output).resolve(),Path(protocol_path).resolve()
    if (queue not in ("odd","even") or not math.isfinite(invocation_seconds)
            or not 0<invocation_seconds<=1800 or not math.isfinite(wall_seconds) or wall_seconds<=0):
        raise ValueError("require odd/even queue, positive outer budget, and child budget <=1800 s")
    protocol=json.loads(protocol_path.read_text())
    revision_path=protocol_path.parent/"checkpoint-contract-v2.json"
    revision=json.loads(revision_path.read_text())
    registration_path=protocol_path.parent/"native-queue-registration.json"
    registration=json.loads(registration_path.read_text())
    expected=dict(protocol["source_sha256"])
    native="benchmarks/water/native_reference.py"
    if revision["from_source_sha256"]!=expected[native]:
        raise ValueError("checkpoint revision does not identify the registered source")
    expected[native]=revision["to_source_sha256"]
    if source_identity()!=expected:
        raise ValueError("current source differs from the explicitly registered/revised source")
    if (registration["protocol_sha256"]!=sha256(protocol_path)
            or registration["checkpoint_revision_sha256"]!=sha256(revision_path)
            or registration["native_source_sha256"]!=expected
            or registration["scheduler_source_sha256"]!=sha256(Path(__file__))):
        raise ValueError("queue registration differs from source/protocol/checkpoint revision")
    if sha256(protocol_path.parent/"checkpoint-contract-v2.patch")!=revision["patch_sha256"]:
        raise ValueError("checkpoint source patch checksum mismatch")
    seeds=protocol["fixed_stage"]["seeds"][0 if queue=="odd" else 1::2]
    identity={"protocol_sha256":sha256(protocol_path),"checkpoint_revision_sha256":sha256(revision_path),
              "native_source_sha256":expected,"scheduler_source_sha256":sha256(Path(__file__)),
              "queue_registration_sha256":sha256(registration_path),
              "queue":queue,"registered_seeds":seeds,"outer_wall_budget_seconds":wall_seconds,
              "child_wall_budget_seconds":invocation_seconds,"propagator_protocol_changes":False}
    output.mkdir(parents=True,exist_ok=True)
    log_path=output/f"scheduler-{queue}.jsonl"
    def journal(event,**values):
        row={"utc":datetime.now(timezone.utc).isoformat(),"event":event,**values}
        with log_path.open("a") as stream:
            stream.write(json.dumps(row,sort_keys=True)+"\n");stream.flush();os.fsync(stream.fileno())
        print(json.dumps(row,sort_keys=True),flush=True)
    with (output/f"scheduler-{queue}.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError("this seed queue already has an active scheduler") from error
        started=time.perf_counter()
        journal("queue_started",identity=identity)
        try:
            for seed in seeds:
                directory=output/"runs"/f"classical-{seed}"
                config_path=protocol_path.parent/"configs"/f"classical-seed-{seed}.json"
                config_bytes=config_path.read_bytes();config=Config(**json.loads(config_bytes)).validate()
                config_sha=hashlib.sha256(config_bytes).hexdigest()
                if (config.seed!=seed or config_sha!=registration["config_sha256"][str(seed)]
                        or sha256(ROOT/config.initial_checkpoint)!=registration["initial_checkpoint_sha256"]):
                    raise ValueError("registered seed/config mapping mismatch")
                while True:
                    remaining=wall_seconds-(time.perf_counter()-started)
                    if remaining<=0:
                        journal("outer_budget_reached",seed=seed);return
                    if (source_identity()!=expected or sha256(protocol_path)!=identity["protocol_sha256"]
                            or sha256(config_path)!=config_sha or sha256(Path(__file__))!=identity["scheduler_source_sha256"]
                            or sha256(registration_path)!=identity["queue_registration_sha256"]
                            or sha256(ROOT/config.initial_checkpoint)!=registration["initial_checkpoint_sha256"]):
                        raise RuntimeError("source, protocol or registered configuration changed during the queue")
                    before,prefix=committed_prefix(directory)
                    if before["complete"]:
                        journal("chain_complete",seed=seed,committed=before);break
                    exists=(directory/"manifest.json").exists()
                    if exists:
                        manifest=json.loads((directory/"manifest.json").read_text())
                        if manifest["config"]!=asdict(config) or manifest["source_sha256"]!=expected:
                            raise ValueError("existing run does not match the fixed registered source/configuration")
                    budget=min(invocation_seconds,remaining)
                    command=[sys.executable,str(ROOT/native),"--output",str(directory),"--wall-seconds",str(budget)]
                    command+=["--resume"] if exists else ["--config",str(config_path)]
                    journal("invocation_started",seed=seed,committed=before,registered_config_sha256=config_sha,
                            child_wall_budget_seconds=budget)
                    logs=output/"logs";logs.mkdir(exist_ok=True)
                    environment=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",OPENMM_CPU_THREADS="1")
                    with (logs/f"classical-{seed}.log").open("a") as log:
                        code=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
                    after,updated=committed_prefix(directory)
                    if updated[:len(prefix)]!=prefix:
                        raise RuntimeError("previously committed trajectory prefix was changed")
                    journal("invocation_finished",seed=seed,exit_code=code,committed=after)
                    if code!=0:
                        raise RuntimeError(f"native invocation for seed {seed} failed; queue stopped without replacement")
                    if after["committed_steps"]<=before["committed_steps"]:
                        raise RuntimeError("successful child made no progress")
            journal("all_registered_queue_chains_complete")
        except BaseException as error:
            journal("queue_stopped_on_error",error_type=type(error).__name__,message=str(error))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--queue",choices=("odd","even"),required=True)
    parser.add_argument("--wall-seconds",type=float,default=7200)
    parser.add_argument("--invocation-seconds",type=float,default=1800)
    args=parser.parse_args()
    schedule(args.output,args.protocol,args.queue,wall_seconds=args.wall_seconds,invocation_seconds=args.invocation_seconds)


if __name__=="__main__":main()
