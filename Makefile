.PHONY: install train serve test dashboard docker-build docker-run

# -------------------------------------------------------------------
# FraudShield Makefile
# -------------------------------------------------------------------

install:
	pip install --upgrade pip
	pip install -r requirements.txt

train:
	python -m src.train_pipeline

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

dashboard:
	streamlit run app/dashboard.py

docker-build:
	docker compose build

docker-run:
	docker compose up -d
