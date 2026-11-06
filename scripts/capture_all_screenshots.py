import asyncio
import base64
import json
import os
import sys
import time
import urllib.request
import websockets

OUT_DIR = os.path.abspath("docs/screenshots")
os.makedirs(OUT_DIR, exist_ok=True)

class CDPClient:
    def __init__(self, ws_url):
        self.ws_url = ws_url
        self.ws = None
        self._id = 0

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, max_size=200_000_000)

    async def send(self, method, params=None):
        self._id += 1
        msg_id = self._id
        req = {"id": msg_id, "method": method, "params": params or {}}
        await self.ws.send(json.dumps(req))
        while True:
            raw = await self.ws.recv()
            resp = json.loads(raw)
            if resp.get("id") == msg_id:
                if "error" in resp:
                    raise RuntimeError(f"CDP Error in {method}: {resp['error']}")
                return resp.get("result", {})

    async def eval_js(self, expression):
        res = await self.send("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True
        })
        return res.get("result", {}).get("value")

    async def capture(self, filename, clip=None):
        params = {"format": "png"}
        if clip:
            params["clip"] = clip
        res = await self.send("Page.captureScreenshot", params)
        data = base64.b64decode(res["data"])
        path = os.path.join(OUT_DIR, filename)
        with open(path, "wb") as f:
            f.write(data)
        print(f"Captured {filename} ({len(data):,} bytes)")
        return path

    async def close(self):
        if self.ws:
            await self.ws.close()

def inject_fault(fault_type="db_slowdown", target="postgres", duration=300, intensity=0.85):
    url = "http://localhost:8001/api/faults/inject"
    req = urllib.request.Request(
        url,
        data=json.dumps({"type": fault_type, "target": target, "duration": duration, "intensity": intensity}).encode(),
        headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req)

def clear_faults():
    url = "http://localhost:8001/api/faults/clear"
    req = urllib.request.Request(url, data=b"{}", headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req)

def reset_session():
    url = "http://localhost:8001/api/session/control"
    req = urllib.request.Request(url, data=json.dumps({"action": "reset"}).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req)

async def main():
    # Make sure session is reset and healthy
    try:
        reset_session()
    except Exception as e:
        print("Warning on reset:", e)

    req = urllib.request.Request("http://localhost:9222/json/new", method="PUT")
    res = urllib.request.urlopen(req)
    target = json.loads(res.read().decode())
    ws_url = target["webSocketDebuggerUrl"]
    target_id = target["id"]

    client = CDPClient(ws_url)
    await client.connect()
    await client.send("Page.enable")
    await client.send("Runtime.enable")

    # High-resolution device viewport (1680x1050 with 2x retina scale = 3360x2100)
    await client.send("Emulation.setDeviceMetricsOverride", {
        "width": 1680,
        "height": 1050,
        "deviceScaleFactor": 2,
        "mobile": False
    })

    print("=== 1. Capturing Live Operations (Normal) ===")
    await client.send("Page.navigate", {"url": "http://localhost:8001/"})
    await asyncio.sleep(4)
    await client.capture("01_live_operations_normal.png")

    print("=== 2. Capturing Chaos Console ===")
    # Scroll to chaos console
    await client.eval_js("window.scrollTo({top: 650, behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("02_chaos_console.png")

    print("=== 3. Injecting Fault & Capturing Active Incident ===")
    try:
        inject_fault("db_slowdown", "postgres", 360, 0.9)
        print("Injected db_slowdown on postgres, waiting 12s for ML pipeline inference & lead-time warning...")
        await asyncio.sleep(12)
        await client.eval_js("window.scrollTo({top: 0, behavior: 'instant'});")
        await asyncio.sleep(1)
        await client.capture("03_live_incident_active_rca.png")
    finally:
        clear_faults()
        reset_session()
        await asyncio.sleep(2)

    print("=== 4. Capturing Incidents Page ===")
    await client.send("Page.navigate", {"url": "http://localhost:8001/incidents"})
    await asyncio.sleep(3)
    await client.capture("04_incidents_heldout_eval.png")

    print("=== 5. Capturing Model Lab - Scorecard ===")
    await client.send("Page.navigate", {"url": "http://localhost:8001/lab#summary"})
    await asyncio.sleep(3)
    await client.eval_js("document.getElementById('summary')?.scrollIntoView({behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("05_model_lab_scorecard.png")

    print("=== 6. Capturing Model Lab - Dimensionality Reduction (PCA) ===")
    await client.eval_js("document.getElementById('dimred')?.scrollIntoView({behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("06_model_lab_dimred.png")

    print("=== 7. Capturing Model Lab - Anomaly Detection ===")
    await client.eval_js("document.getElementById('anomaly')?.scrollIntoView({behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("07_model_lab_anomaly.png")

    print("=== 8. Capturing Model Lab - Fault Classification & Confusion Matrix ===")
    await client.eval_js("document.getElementById('classification')?.scrollIntoView({behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("08_model_lab_classification.png")

    print("=== 9. Capturing Model Lab - Unsupervised Clustering ===")
    await client.eval_js("document.getElementById('clustering')?.scrollIntoView({behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("09_model_lab_clustering.png")

    print("=== 10. Capturing Model Lab - Early Warning LSTM ===")
    await client.eval_js("document.getElementById('warning')?.scrollIntoView({behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("10_model_lab_early_warning.png")

    print("=== 11. Capturing Model Lab - Root Cause Analysis (RCA) ===")
    await client.eval_js("document.getElementById('rca')?.scrollIntoView({behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("11_model_lab_root_cause.png")

    print("=== 12. Capturing Model Lab - Real Docker Campaigns ===")
    # Click or toggle to 'Real Docker data'
    await client.eval_js("""
    (() => {
        const btns = Array.from(document.querySelectorAll('button'));
        const realBtn = btns.find(b => b.textContent && b.textContent.includes('Real Docker data'));
        if (realBtn) realBtn.click();
    })()
    """)
    await asyncio.sleep(2)
    await client.eval_js("document.getElementById('summary')?.scrollIntoView({behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("12_model_lab_real_docker.png")

    print("=== 13. Capturing Testbed Architecture ===")
    await client.send("Page.navigate", {"url": "http://localhost:8001/testbed"})
    await asyncio.sleep(3)
    await client.capture("13_testbed_architecture.png")

    print("=== 14. Capturing Testbed Dataset & Drain Templates ===")
    await client.eval_js("window.scrollTo({top: 480, behavior: 'instant'});")
    await asyncio.sleep(1)
    await client.capture("14_testbed_dataset_drain.png")

    print("=== 15. Capturing FastAPI Swagger Interactive Docs ===")
    await client.send("Page.navigate", {"url": "http://localhost:8001/docs"})
    await asyncio.sleep(3)
    await client.capture("15_api_swagger_docs.png")

    print("=== All screenshots captured successfully! ===")
    await client.close()

    try:
        urllib.request.urlopen(f"http://localhost:9222/json/close/{target_id}")
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(main())
