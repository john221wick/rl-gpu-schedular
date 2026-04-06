UV := uv
PYTHON := python

.PHONY: init sync lock test smoke api inference baseline-medium baseline-hard docker-build clean clean-venv

init:
	$(UV) venv
	$(UV) sync

sync:
	$(UV) sync

lock:
	$(UV) lock

test:
	$(UV) run $(PYTHON) -m unittest discover -s tests -v

smoke:
	$(UV) run $(PYTHON) -m compileall gpu_scheduler server tests

api:
	$(UV) run server --host 0.0.0.0 --port 7860 --reload

inference:
	$(UV) run $(PYTHON) inference.py

baseline-medium:
	$(UV) run $(PYTHON) -c "from gpu_scheduler.baseline import run_baseline; import json; print(json.dumps(run_baseline('smart', 'medium_001'), indent=2))"

baseline-hard:
	$(UV) run $(PYTHON) -c "from gpu_scheduler.baseline import run_baseline; import json; print(json.dumps(run_baseline('smart', 'hard_001'), indent=2))"

docker-build:
	docker build -t gpu-scheduler-ml .

clean:
	rm -rf __pycache__ gpu_scheduler/__pycache__ tests/__pycache__

clean-venv:
	rm -rf .venv
