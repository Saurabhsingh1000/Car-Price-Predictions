.PHONY: install install-dev train evaluate test run docker-build docker-run lint clean

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

test:
	pytest tests/ -v

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

lint:
	flake8 src app tests scripts --max-line-length=100

docker-build:
	docker build -t car-price-prediction .

docker-run:
	docker run --rm -p 8000:8000 car-price-prediction

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
