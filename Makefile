.PHONY: mypy
mypy:
	mypy --config-file=mypy.ini --no-site-packages .

install:
	poetry add jsonlines
	poetry add python-box
	poetry install
