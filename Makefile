.PHONY: install check test

install:
	python -m pip install -e .

check:
	python -m compileall -q src scripts tests

test:
	PYTHONPATH=src python -m unittest discover -s tests -v
