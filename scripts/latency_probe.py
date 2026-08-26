"""정책 서버 추론 지연(청크당 ms) 측정. simpler_env env + PYTHONPATH=third_party/starVLA 에서 실행.
   python scripts/latency_probe.py --port 6678 [--n 20] [--warmup 3]  → JSON 1줄 출력"""
import argparse, json, time
import numpy as np
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

p = argparse.ArgumentParser()
p.add_argument("--port", type=int, required=True); p.add_argument("--host", default="127.0.0.1")
p.add_argument("--n", type=int, default=20); p.add_argument("--warmup", type=int, default=3)
p.add_argument("--unnorm-key", default="oxe_bridge")
a = p.parse_args()
client = WebsocketClientPolicy(a.host, a.port)
meta = client.get_server_metadata()
rng = np.random.default_rng(0)
img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
req = {"examples": [{"image": [img], "lang": "put the spoon on the towel"}], "do_sample": False,
       "use_ddim": True, "num_ddim_steps": 10, "unnorm_key": a.unnorm_key}
for _ in range(a.warmup): client.predict_action(req)
ts = []
for _ in range(a.n):
    t = time.perf_counter(); r = client.predict_action(req); ts.append((time.perf_counter() - t) * 1000)
ts = np.array(ts)
print(json.dumps({"chunk_size": meta.get("action_chunk_size"), "n": a.n, "mean_ms": round(float(ts.mean()), 1),
                  "p50_ms": round(float(np.median(ts)), 1), "p95_ms": round(float(np.percentile(ts, 95)), 1),
                  "action_shape": list(np.asarray(r["data"]["actions"]).shape)}))
client.close()
