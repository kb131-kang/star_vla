#!/usr/bin/env python3
"""Step 6: SimplerEnv WidowX CI.
  python3 scripts/run_ci.py --ckpt <steps_XXXX_pytorch_model.pt> [--port 6678] [--tasks spoon,carrot,stack,eggplant]
                            [--episodes 24] [--sequential] [--run-id ID] [--notes "..."]
동작: 정책 서버 기동 → 추론 지연 측정 → WidowX 4과제 평가 → reports/{run_id}.json + reports/summary.csv 적재 → 서버 종료.
stdlib 만 사용 (base python3 로 실행 가능). 실제 작업은 scripts/lib/*.sh 에 위임한다.
"""
import argparse, csv, datetime as dt, json, os, re, subprocess, sys, time
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
LIB = WS / "scripts" / "lib"
TASK_ORDER = ["spoon", "carrot", "stack", "eggplant"]
CSV_COLS = ["run_id", "datetime", "ckpt_id", "framework", "train_steps", "spoon", "carrot", "stack", "eggplant",
            "mean", "n_episodes", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "chunk_size",
            "server_load_s", "eval_wall_s", "mode", "workspace_commit", "notes"]


def sh(cmd, **kw):
    return subprocess.run(cmd, text=True, capture_output=True, **kw)


def ckpt_meta(ckpt: Path):
    run_dir = ckpt.parents[1]
    cfg = (run_dir / "config.yaml").read_text() if (run_dir / "config.yaml").exists() else ""
    fw = re.search(r"^framework:\s*\n\s+name:\s*(\S+)", cfg, re.M)
    steps = re.search(r"steps_(\d+)", ckpt.name)
    return {"ckpt_id": f"{run_dir.name}/{ckpt.name}", "framework": fw.group(1) if fw else "?",
            "train_steps": int(steps.group(1)) if steps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--port", type=int, default=6678)
    ap.add_argument("--tasks", default=",".join(TASK_ORDER)); ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--sequential", action="store_true", help="과제를 순차 실행 (기본: 4과제 동시 실행, 원본 스크립트와 동일)")
    ap.add_argument("--run-id", default=None); ap.add_argument("--notes", default="")
    ap.add_argument("--skip-latency", action="store_true")
    a = ap.parse_args()
    ckpt = Path(a.ckpt).resolve(); assert ckpt.is_file(), f"ckpt not found: {ckpt}"
    meta = ckpt_meta(ckpt)
    now = dt.datetime.now()
    run_id = a.run_id or f"{now:%Y%m%d_%H%M%S}_{meta['framework']}_{meta['train_steps']}"
    out = WS / "logs" / "ci" / run_id; out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in a.tasks.split(",") if t]
    commit = sh(["git", "-C", str(WS), "rev-parse", "--short", "HEAD"]).stdout.strip()
    print(f"[ci] run_id={run_id} ckpt={ckpt} tasks={tasks} episodes={a.episodes} mode={'seq' if a.sequential else 'parallel'}")

    report = {"run_id": run_id, "datetime": now.isoformat(timespec="seconds"), "ckpt_path": str(ckpt), **meta,
              "n_episodes": a.episodes, "tasks": {}, "latency": None, "mode": "sequential" if a.sequential else "parallel",
              "workspace_commit": commit, "notes": a.notes, "log_dir": str(out)}
    try:
        t0 = time.time()
        r = sh(["bash", str(LIB / "policy_server.sh"), "start", str(ckpt), str(a.port), str(out / "server.log")])
        print(r.stdout.strip()); 
        if r.returncode != 0: print(r.stderr); sys.exit(1)
        report["server_load_s"] = round(time.time() - t0, 1)

        if not a.skip_latency:
            env = dict(os.environ, PYTHONPATH=str(WS / "third_party" / "starVLA"))
            sim_py = os.environ.get("SIM_PY", str(Path(os.environ.get("CONDA_BASE", Path.home() / "miniconda3")) / "envs/simpler_env/bin/python"))
            r = sh([sim_py, str(WS / "scripts" / "latency_probe.py"), "--port", str(a.port)], env=env, cwd=WS)
            try:
                report["latency"] = json.loads(r.stdout.strip().splitlines()[-1]); print(f"[ci] latency {report['latency']}")
            except Exception:
                print("[ci] latency probe failed:\n", r.stdout[-2000:], r.stderr[-2000:])

        t1 = time.time(); procs = {}
        def launch(t):
            log = out / f"{t}.log"
            cmd = ["bash", str(LIB / "run_task.sh"), str(ckpt), str(a.port), t, "0", str(a.episodes), str(out / "videos"), str(log)]
            return subprocess.Popen(cmd, stdout=open(out / f"{t}.summary", "w"), stderr=subprocess.STDOUT), time.time()
        for t in tasks:
            procs[t] = launch(t)
            if a.sequential:
                procs[t][0].wait(); report["tasks"][t] = {"wall_s": round(time.time() - procs[t][1], 1)}
            else:
                time.sleep(6)
        if not a.sequential:
            for t in tasks:
                procs[t][0].wait(); report["tasks"][t] = {"wall_s": round(time.time() - procs[t][1], 1)}
        report["eval_wall_s"] = round(time.time() - t1, 1)
    finally:
        sh(["bash", str(LIB / "policy_server.sh"), "stop", str(a.port)])

    rates = []
    for t in tasks:
        log = (out / f"{t}.log").read_text(errors="ignore") if (out / f"{t}.log").exists() else ""
        m = re.findall(r"Average success ([0-9.]+)", log)
        d = report["tasks"].setdefault(t, {})
        if m:
            sr = float(m[-1]); d.update(success_rate=round(sr, 4), n_success=round(sr * a.episodes), n_episodes=a.episodes, status="ok")
            rates.append(sr)
        else:
            d.update(success_rate=None, status="FAILED", log=str(out / f"{t}.log"))
    report["mean"] = round(sum(rates) / len(rates), 4) if rates else None
    report["all_tasks_ok"] = len(rates) == len(tasks)

    rep_dir = WS / "reports"; rep_dir.mkdir(exist_ok=True)
    (rep_dir / f"{run_id}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    lat = report["latency"] or {}
    row = {"run_id": run_id, "datetime": report["datetime"], "ckpt_id": meta["ckpt_id"], "framework": meta["framework"],
           "train_steps": meta["train_steps"], "mean": report["mean"], "n_episodes": a.episodes,
           "latency_mean_ms": lat.get("mean_ms"), "latency_p50_ms": lat.get("p50_ms"), "latency_p95_ms": lat.get("p95_ms"),
           "chunk_size": lat.get("chunk_size"), "server_load_s": report.get("server_load_s"), "eval_wall_s": report.get("eval_wall_s"),
           "mode": report["mode"], "workspace_commit": commit, "notes": a.notes}
    for t in TASK_ORDER:
        row[t] = report["tasks"].get(t, {}).get("success_rate")
    csv_path = rep_dir / "summary.csv"; new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS); new and w.writeheader(); w.writerow(row)

    print(f"\n== {run_id} ({meta['ckpt_id']}, {meta['framework']} @ {meta['train_steps']} steps) ==")
    for t in tasks:
        d = report["tasks"][t]
        print(f"  {t:9s} {('%5.1f%%' % (d['success_rate']*100)) if d.get('success_rate') is not None else ' FAIL '}  "
              f"({d.get('n_success','?')}/{a.episodes})  wall={d.get('wall_s')}s")
    print(f"  mean      {report['mean']*100 if report['mean'] is not None else float('nan'):5.1f}%   latency={lat.get('mean_ms')}ms/chunk"
          f"  server_load={report.get('server_load_s')}s eval_wall={report.get('eval_wall_s')}s")
    print(f"  report: {rep_dir / (run_id + '.json')}  |  summary: {csv_path}")
    sys.exit(0 if report["all_tasks_ok"] else 2)


if __name__ == "__main__":
    main()
