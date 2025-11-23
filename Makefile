.PHONY: mypy
mypy:
	mypy --config-file=mypy.ini --no-site-packages .

install:
	poetry install
