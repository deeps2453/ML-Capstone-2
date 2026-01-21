.PHONY: help setup train run test docker-build docker-run clean

help:
	@echo "Football Player Salary Prediction - Available Commands:"
	@echo ""
	@echo "  make setup          - Install dependencies and setup environment"
	@echo "  make train          - Train the model"
	@echo "  make run            - Run the Flask API locally"
	@echo "  make test           - Run API tests"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-run     - Run Docker container"
	@echo "  make docker-compose - Run with docker-compose"
	@echo "  make clean          - Clean generated files"
	@echo ""

setup:
	@echo "Setting up environment..."
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "Setup complete! Activate with: source venv/bin/activate"

setup-pipenv:
	@echo "Setting up environment with Pipenv..."
	pipenv install
	@echo "Setup complete! Activate with: pipenv shell"

train:
	@echo "Training model..."
	python src/train.py

run:
	@echo "Starting Flask API..."
	python src/predict.py

test:
	@echo "Running API tests..."
	python test_api.py

docker-build:
	@echo "Building Docker image..."
	docker build -t salary-prediction:latest .

docker-run:
	@echo "Running Docker container..."
	docker run -p 5000:5000 salary-prediction:latest

docker-compose:
	@echo "Starting with docker-compose..."
	docker-compose up -d
	@echo "Service started! Access at http://localhost:5000"

docker-compose-down:
	@echo "Stopping docker-compose services..."
	docker-compose down

clean:
	@echo "Cleaning generated files..."
	rm -rf models/*.pkl models/*.json models/*.png
	rm -rf __pycache__ src/__pycache__
	rm -rf .pytest_cache
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

deploy-aws:
	@echo "Deploying to AWS..."
	chmod +x deployment/cloud/deploy.sh
	./deployment/cloud/deploy.sh

deploy-k8s:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f deployment/kubernetes/deployment.yaml

all: setup train run
