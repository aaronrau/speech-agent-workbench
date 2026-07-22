PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: test test-real-tmux
test:
	$(PYTHON) -m unittest discover -s tests -t .

test-real-tmux:
	$(PYTHON) -m unittest -v tests.test_real_tmux_workbench
