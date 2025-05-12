# Variables
PYTHON=python
PIP=$(PYTHON) -m pip
FLAKE8=flake8
PYTEST=pytest
COVERAGE=coverage_html
REQUIREMENTS=docs/requirements.txt
ENV_FILE=.env
TEST_DIR=tests
SRC_DIR=src

# Default target
all: install lint test coverage deploy clean

# Install dependencies
install:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r $(REQUIREMENTS)

# Linting
lint:
	@echo "Running flake8 linting..."
	$(PIP) install --quiet flake8
	$(PYTHON) -m $(FLAKE8) --config .flake8

# Testing
test:
	@echo "Running tests with pytest..."
	$(PIP) install --quiet pytest pytest-cov
	$(PYTHON) -m pytest $(TEST_DIR) --cov-report=term-missing

# Coverage report
coverage:
	@echo "Generating coverage report..."
	$(PIP) install --quiet coverage pytest pytest-cov
	$(PYTHON) -m $(COVERAGE) run -m pytest --cov=${SRC_DIR}
	$(PYTHON) -m $(COVERAGE) report
	$(PYTHON) -m $(COVERAGE) html

# Deploy (TODO)
deploy:
	@echo "Deploying to Render..."

# Clean
clean:
	@echo "Cleaning up..."
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf .pytest_cache htmlcov

# Help
help:
	@echo "Makefile for FastAPI project"
	@echo ""
	@echo "Usage:"
	@echo "  make install      Install dependencies"
	@echo "  make lint         Run flake8 linting"
	@echo "  make test         Run tests with pytest"
	@echo "  make coverage     Generate test coverage report"
	@echo "  make deploy       Deploy application to Render"
	@echo "  make clean        Clean up temporary files"
	@echo "  make help         Show this help message"
