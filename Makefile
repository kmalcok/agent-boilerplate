# TR: Sik kullanilan komutlar. / EN: Common commands.
.PHONY: install dev cli serve lint

install:
	pip install -r requirements.txt

# TR: Gelistirme bagimliliklariyla kur. / EN: Install with dev dependencies.
dev:
	pip install -e ".[dev]"

# TR: Interaktif terminal demosu. / EN: Interactive terminal demo.
cli:
	python -m app.cli

# TR: Production HTTP sunucusu. / EN: Production HTTP server.
serve:
	uvicorn app.server:app --host 0.0.0.0 --port 8080 --reload

lint:
	ruff check app
	mypy app
