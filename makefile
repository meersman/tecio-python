PYTHON ?= python3
PACKAGE = tecio-python
IMPORT = tecio

.PHONY: help \
	install uninstall venv \
	format lint typecheck \
	test coverage testclean \
	build distclean clean \
	versions \

# Default target
help:
	@echo ""
	@echo "                      Tecio Development Commands"
	@echo "    ================================================================="
	@echo "    Setup:"
	@echo "      make install          Install tecio package and dependencies"
	@echo "      make uninstall        Uninstall tecio  package"
	@echo "      make venv             Create python env and install tecio package and dependencies"
	@echo "    Code Quality:"
	@echo "      make format           Format code ruff"
	@echo "      make lint             Run code linting with ruff"
	@echo "      make typecheck        Run mypy type checking"
	@echo "      make check            Run all checks (format + lint)"
	@echo "    Testing:"
	@echo "      make test             Run unit tests"
	@echo "      make coverage         Run unit tests and create coverage report"
	@echo "    Cleanup:"
	@echo "      make clean            Remove generated files"
	@echo "      make testclean        Remove test artefacts"
	@echo "      make distclean        Remove distribution build artifacts"
	@echo "    Distribution:"
	@echo "      make build            Build wheel and sdist into dist/"
	@echo ""

# Install package dependencies
install:
	$(PYTHON) -m pip install -e ".[all]"

# Uninstall package
uninstall:
	$(PYTHON) -m pip uninstall -y $(PACKAGE)

# Create venv and install tecio and dependencies
venv:
	$(PYTHON) -m venv .venv
	.venv/bin/$(PYTHON) -m pip install -e ".[all]"

# Format code with ruff
format:
	ruff format --check

# Run linter
lint:
	ruff check

# Run type checker
typecheck:
	ty check tecio

# Run all checks
check: format lint typecheck
	@echo ""
	@echo "✓ All checks passed"

# Run unit tests
test:
	pytest -v --junitxml=junit.xml; EXIT=$$?; \
	exit $$EXIT
	@echo "✓ Tests complete"

# Run coverage report
coverage:
	pytest -q --cov=$(IMPORT)  --cov-report=term; EXIT=$$?; \
	exit $$EXIT
	@echo "✓ Tests complete"

# Clean test artifacts
testclean:
	@echo "✓ Cleaning test generated files"
	@find . -type f -name "test*.plt" -delete
	@find . -type f -name "test*.szplt" -delete
	@find . -type f -name "test*.dat" -delete
	@find . -type f -name "*.xml" -delete
	@find . -type f -name "tp?*" -delete

# Clean up generated files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "test*.plt" -delete
	find . -type f -name "test*.szplt" -delete
	find . -type f -name "test*.dat" -delete
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "✓ Cleanup complete"

# Build wheel and sdist distribution artifacts
build:
	$(PYTHON) -m build
	twine check dist/*
	@echo "✓ Distribution artifacts built in dist/"

# Remove distribution build artifacts only
distclean:
	rm -rf build/ dist/ *.egg-info
	@echo "✓ Distribution artifacts removed"

# Show current versions of tools
versions:
	@echo "Tool Versions:"
	@echo "=============="
	@python --version
	@ruff --version
	@ty --version
	@pytest --version
