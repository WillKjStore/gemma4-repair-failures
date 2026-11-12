"""Run the pinned harness in separately mounted agent and verifier phases."""

import argparse
import asyncio
import hashlib
import inspect
import json
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch


ALIAS = "gemma-4-e4b-it-qat-w4a16-ct"
SEED = 20261109
GRAPH_TOOLS = {"get_code_neighbors", "search_similar_code", "get_code_subgraph"}
PUBLIC_TASK_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement", "hints_text", "created_at")
STUDY = Path("/path/to/study")
INPUTS = Path("/var/tmp")


def save_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)


def prepare(bundle, output, pilot_index, task_id=None):
    if STUDY.resolve(strict=True) != STUDY or not bundle.resolve(strict=True).is_relative_to(STUDY):
        raise ValueError("invalid study bundle")
    if output.exists() or output.is_symlink() or not output.parent.resolve(strict=True).is_relative_to(STUDY):
        raise ValueError("new private preparation directory required")
    manifest = (bundle / "tasks.jsonl").resolve(strict=True)
    manifest.relative_to(STUDY)
    with manifest.open(encoding="utf-8") as stream:
        candidates = [json.loads(line) for line in stream]
    if task_id is None:
        candidates = [row for row in candidates if row["repo"].rsplit("/", 1)[-1].lower() in {"rich", "fastapi"}
                      and row.get("patch") and row.get("test_patch")]
        candidates.sort(key=lambda row: (row["repo"].rsplit("/", 1)[-1].lower() != "rich",
                                        hashlib.sha256(f'{SEED}:gate:{row["instance_id"]}'.encode()).hexdigest()))
    else:
        candidates = [row for row in candidates if row["instance_id"] == task_id and row.get("test_patch")]
        if len(candidates) != 1:
            raise ValueError("frozen task does not match one manifest entry")
    for task in candidates:
        task_id = task["instance_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            continue
        snapshot = bundle / "snapshots" / f"{task_id}.tgz"
        commit = task["base_commit"]
        if not re.fullmatch(r"[a-f0-9]{40}", commit):
            continue
        matching = [list((bundle / folder).glob(f"*{commit}*.{suffix}"))
                    for folder, suffix in (("graphs", "json"), ("embeddings", "npz"))]
        if (snapshot.is_file() and snapshot.resolve(strict=True).is_relative_to(STUDY)
                and all(len(files) == 1 and files[0].is_file() and files[0].resolve(strict=True).is_relative_to(STUDY)
                        for files in matching)):
            if pilot_index:
                pilot_index -= 1
                continue
            break
    else:
        raise ValueError("no eligible direct-snapshot pilot task")

    output.mkdir(mode=0o700)
    agent = {key: task[key] for key in PUBLIC_TASK_FIELDS if key in task}
    verifier = {key: task[key] for key in (*PUBLIC_TASK_FIELDS[:3], "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS")
                if key in task}
    verifier["problem_statement"] = ""
    for phase, row in (("agent", agent), ("verify", verifier)):
        with (output / f"{phase}.jsonl").open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
    if task_id is None:
        with (output / "reference.patch").open("x", encoding="utf-8", newline="") as stream:
            stream.write(task["patch"])
    for name, source in zip(("snapshot", "graph", "embedding"), (snapshot, matching[0][0], matching[1][0])):
        with (output / f"{name}.path").open("x", encoding="utf-8") as stream:
            stream.write(str(source.resolve(strict=True)) + "\n")


def configuration(tasks, output, models):
    from swegemma.budget import EvaluationBudget, HarnessLimits
    from swegemma.config import EvalConfig

    return EvalConfig(tasks_path=tasks, snapshots_dir=INPUTS / "snapshots", results_dir=output,
                      submission_dir=INPUTS / "submission", models=models, sandbox="subprocess",
                      wheels_dir=INPUTS / "wheels", graph_dir=str(INPUTS / "graphs"),
                      embeddings_dir=str(INPUTS / "embeddings"), display_mode="quiet", concurrency=1,
                      budget=EvaluationBudget(time_minutes=30, tool_calls=50, turns=150),
                      harness=HarnessLimits(command_timeout_seconds=300, verification_timeout_seconds=300))


def one_task(path):
    from swegemma.models.task import load_tasks

    [task] = load_tasks(path)
    return task


def require_container():
    import socket

    if (STUDY.exists() or (INPUTS / "reference.patch").exists() or
            {name for _, name in socket.if_nameindex()} != {"lo"}):
        raise RuntimeError("isolation preflight failed")


def snapshot_setup_timeout(manager, output):
    original = manager.exec

    def execute(sandbox_id, command, *, timeout=None):
        snapshot_unpack = command.startswith("(tar --no-same-owner -I pigz -xf /tmp/") and "-C /workspace)" in command
        result = original(sandbox_id, command, timeout=1800 if snapshot_unpack else timeout)
        if snapshot_unpack:
            save_json(output / "snapshot-extraction.json", {"exit_code": result.exit_code,
                      "status": result.status, "duration_seconds": result.duration_seconds,
                      "timeout_seconds": 1800})
        return result

    return patch.object(manager, "exec", execute)


async def agent_phase(output):
    import torch
    from adk_submission.resolvers.generation import install_litellm_reasoning_patch
    from adk_submission.server import VllmConfig, VllmServer
    from swegemma import evaluate
    from swegemma.context import SwegemmaContext
    from swegemma.harness.verification import save_trace_artifact
    from swegemma.models.task import TaskResult

    class SplitVllmServer(VllmServer):
        def build_cmd(self):
            command = super().build_cmd()
            interpreter = Path("/opt/server-venv/bin/python")
            if (command[:3] != [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
                    or not interpreter.is_file()):
                raise RuntimeError("unexpected official server command or missing inference interpreter")
            return [str(interpreter), *command[1:]]

        def stop(self, timeout=10.0):
            saved_log = output / "vllm-server.log"
            try:
                if self.log_path:
                    original_log = Path(self.log_path)
                    if (original_log.is_file() and original_log.resolve(strict=True).is_relative_to(Path("/tmp"))
                            and not saved_log.exists()):
                        with original_log.open("rb") as source, saved_log.open("xb") as destination:
                            shutil.copyfileobj(source, destination)
            finally:
                super().stop(timeout=timeout)

    require_container()
    tasks = INPUTS / "task.jsonl"
    task = one_task(tasks)
    if (task.patch or task.test_patch or getattr(task, "FAIL_TO_PASS", ()) or getattr(task, "PASS_TO_PASS", ())
            or (INPUTS / "candidate.patch").exists()):
        raise RuntimeError("verification-only data reached agent phase")
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected exactly one H100 visible in the agent phase")
    output.mkdir()
    original_tools = SwegemmaContext.create_tools
    tools_initialized = set()
    called = False

    def record_tools(context):
        result = original_tools(context)
        names = result.keys() if isinstance(result, dict) else (
            getattr(tool, "name", getattr(tool, "__name__", "")) for tool in result)
        tools_initialized.update(names)
        return result

    async def capture(_docker, config, task, _snapshot_path, *, agent_patch="", agent_error=None,
                      trace=None, start_time=0.0, **_kwargs):
        nonlocal called
        called = True
        with (output / "agent.patch").open("x", encoding="utf-8", newline="") as stream:
            stream.write(agent_patch)
        trace_path = save_trace_artifact(trace, output, task.instance_id)
        return TaskResult(instance_id=task.instance_id, repo=task.repo, resolved=False,
                          agent_patch=agent_patch, error=agent_error, trace=trace,
                          trace_json_path=trace_path, duration_seconds=time.monotonic() - started)

    config = VllmConfig(model=str(INPUTS / "model"), tool_call_parser="gemma4", reasoning_parser="gemma4",
                       reasoning_config={"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"},
                       max_model_len=32768, gpu_memory_utilization=0.90, enable_lora=False,
                       startup_timeout=3600, tensor_parallel_size=1,
                       extra_args=["--served-model-name", ALIAS])
    install_litellm_reasoning_patch()
    server = SplitVllmServer(config)
    save_json(output / "vllm-log-location.json", {"path": str(server.log_path)})
    try:
        server.start()
        registry = server.create_model_registry(aliases={ALIAS: ALIAS}, model_prefix="openai/")
        evaluator = evaluate.Evaluator(configuration(tasks, output, registry))
        started = time.monotonic()
        with (patch.object(evaluate, "verify_task", capture), patch.object(SwegemmaContext, "create_tools", record_tools),
              snapshot_setup_timeout(evaluator.docker, output)):
            result = await evaluator.evaluate_task(task)
    finally:
        server.stop()
    save_json(output / "agent-status.json", {"deferred_verification": called,
              "inference_healthy": True, "initialized_tools": sorted(tools_initialized),
              "tool_calls": result.tool_calls, "elapsed_seconds": time.monotonic() - started,
              "has_agent_error": bool(result.error), "trace_saved": bool(result.trace_json_path),
              "gpu_name": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count()})
    if not called or not GRAPH_TOOLS <= tools_initialized or not result.trace_json_path:
        raise RuntimeError("agent phase did not produce required artifacts")


async def verify_phase(output, patch_path, control):
    from adk_submission.server import ModelRegistry
    from swegemma.deduplication import resolve_task_snapshot_paths
    from swegemma.evaluate import Evaluator
    from swegemma.harness import verification

    require_container()
    tasks = INPUTS / "task.jsonl"
    task = one_task(tasks)
    if task.patch or task.problem_statement:
        raise RuntimeError("verifier input contains unexpected fields")
    candidate = patch_path.read_text(encoding="utf-8") if patch_path else ""
    output.mkdir()
    config = configuration(tasks, output, ModelRegistry())
    evaluator = Evaluator(config)
    snapshot, base, delta = resolve_task_snapshot_paths(config.snapshots_dir, task.instance_id, task.repo)
    if not snapshot.is_file():
        raise RuntimeError("missing direct pilot snapshot")
    original_exec = verification.sandbox_exec
    junit_capture = None

    async def record_junit(manager, sandbox_id, command, *, timeout=None):
        nonlocal junit_capture
        match = re.search(r"--junitxml[=\s]+['\"]?(/tmp/_swegemma_junit_[a-f0-9]+\.xml)", command)
        if match:
            pytest_install = await original_exec(
                manager, sandbox_id,
                "(python3 -I -m pip --version || python3 -I -m ensurepip --default-pip) && "
                "python3 -I -m pip install --no-index --find-links=/mnt pytest && "
                "python3 -I -c 'import pytest'", timeout=300,
            )
            detail = (pytest_install.stdout + "\n" + pytest_install.stderr).lower()
            labels = {"no_matching_distribution": "no matching distribution found", "missing_pip": "no module named pip",
                      "read_only": "read-only file system", "permission_denied": "permission denied",
                      "ignored_links": "is ignored", "timeout": "timed out", "missing_pytest": "no module named pytest"}
            save_json(output / "sandbox-deps.json", {"exit_code": pytest_install.exit_code,
                      "diagnostic_categories": [name for name, marker in labels.items() if marker in detail]})
            if pytest_install.exit_code != 0:
                raise RuntimeError("offline pytest installation failed in the task sandbox")
            package = task.repo.rsplit("/", 1)[-1].lower()
            code = ("import importlib.util, pathlib; "
                    f"spec = importlib.util.find_spec({package!r}); "
                    "assert spec and spec.origin and pathlib.Path(spec.origin).resolve().is_relative_to(pathlib.Path.cwd().resolve())")
            probe = await original_exec(manager, sandbox_id,
                                        "cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -c "
                                        + shlex.quote(code), timeout=20)
            if probe.exit_code != 0:
                raise RuntimeError("the verification import does not resolve into the task snapshot")
        value = await original_exec(manager, sandbox_id, command, timeout=timeout)
        if match:
            xml = Path(match[1])
            if xml.is_file():
                junit_capture = xml.read_text(encoding="utf-8")
            else:
                captured = await original_exec(manager, sandbox_id, f"cat {shlex.quote(match[1])}", timeout=10)
                junit_capture = getattr(captured, "stdout", None)
        return value

    if not inspect.iscoroutinefunction(original_exec):
        raise RuntimeError("unsupported official sandbox_exec interface")
    with patch.object(verification, "sandbox_exec", record_junit), snapshot_setup_timeout(evaluator.docker, output):
        result = await verification.verify_task(evaluator.docker, config, task, snapshot,
                                                base_snapshot_path=base, patch_path=delta,
                                                fast_path=False, agent_patch=candidate, start_time=time.monotonic())
    with (output / "verification.log").open("x", encoding="utf-8") as stream:
        stream.write(result.test_output or "")
        stream.write("\n" + (result.error or ""))
    if junit_capture:
        (output / "junit.xml").write_text(junit_capture, encoding="utf-8")
    save_json(output / "status.json", {"control": control, "resolved": result.resolved,
              "exit_code": result.test_exit_code, "junit_captured": bool(junit_capture),
              "duration_seconds": result.duration_seconds})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "agent", "verify"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--pilot-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--task-id")
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--control", choices=("reference", "empty", "agent"))
    args = parser.parse_args()
    if args.phase == "prepare":
        if not args.bundle:
            parser.error("prepare requires --bundle")
        os.umask(0o077)
        prepare(args.bundle, args.output, args.pilot_index, args.task_id)
    elif args.phase == "agent":
        asyncio.run(agent_phase(args.output))
    else:
        if not args.control or (args.control != "empty") != bool(args.patch):
            parser.error("verify requires --control and requires --patch for reference and agent cases")
        asyncio.run(verify_phase(args.output, args.patch, args.control))


if __name__ == "__main__":
    main()
