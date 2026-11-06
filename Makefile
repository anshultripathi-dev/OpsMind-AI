PY = .venv/bin/python

.PHONY: all dataset train loghub dashboard api test clean real-up real-down real-run real-medium real-ramped real-collect real-collect2 real-eval real-train
DOCKER_BIN = /Applications/Docker.app/Contents/Resources/bin
COMPOSE = PATH="$$PATH:$(DOCKER_BIN)" docker compose -f testbed/docker-compose.yml
all: dataset train loghub dashboard api

dataset:
	$(PY) scripts/generate_dataset.py
train:
	$(PY) scripts/train.py
loghub:
	mkdir -p data/loghub && [ -f data/loghub/HDFS_2k.log_structured.csv ] || curl -sSL -o data/loghub/HDFS_2k.log_structured.csv https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log_structured.csv
	$(PY) scripts/validate_loghub.py
dashboard:
	cd dashboard && npm install --no-audit --no-fund && npm run build
api:
	$(PY) -W ignore -m uvicorn api.main:app --port 8000
test:
	$(PY) -W ignore -m pytest tests -q
clean:
	rm -rf data artifacts dashboard/dist

# ---- real Docker testbed (needs Docker Desktop running) -----------------------------------------
real-up:
	$(COMPOSE) --profile load up -d --build
real-down:
	$(COMPOSE) --profile load down
real-run:        # ~72 min: 2 control runs + one episode of every fault type (9 min each)
	PATH="$$PATH:$(DOCKER_BIN)" $(PY) -u testbed/injector/injector.py --plan normal,db_slowdown,memory_leak,network_latency,service_crash,cache_failure,cpu_saturation,normal --run-sec 540 --warmup 170 230 --dur 150 240 --imin 0.6 --out data/real/episodes.jsonl
real-medium:     # ≈ 8 h overnight: 8 episodes of every fault + 6 control runs (keep the Mac awake and plugged in)
	PATH="$$PATH:$(DOCKER_BIN)" caffeinate -dis $(PY) -u testbed/injector/injector.py --plan "$$(cat data/real/plan.txt)" --run-sec 540 --warmup 170 230 --dur 150 240 --imin 0.4 --seed 21 --out data/real/episodes.jsonl
real-ramped:     # ≈ 9 h: campaign 2 — ramped faults, 12 min runs, 170-user bursts, stack reset before every run
	BURST_USERS=170 BURST_FIRST=120 $(COMPOSE) --profile load up -d --force-recreate locust
	PATH="$$PATH:$(DOCKER_BIN)" caffeinate -dis $(PY) -u testbed/injector/injector.py --plan "$$(cat data/real2/plan.txt)" --run-sec 720 --warmup 240 300 --dur 270 330 --imin 0.4 --ramp-frac 0.55 --reset --seed 33 --out data/real2/episodes.jsonl
real-collect2:
	PATH="$$PATH:$(DOCKER_BIN)" $(PY) testbed/collector/collect.py --episodes data/real2/episodes.jsonl --out data/real2 --prefix rb
real-collect:
	PATH="$$PATH:$(DOCKER_BIN)" $(PY) testbed/collector/collect.py --episodes data/real/episodes.jsonl --out data/real
real-eval:
	$(PY) scripts/evaluate_real.py
real-train:      # train every module on the real runs; writes artifacts/real/ (Model lab 'Real Docker data')
	$(PY) scripts/train_real.py
