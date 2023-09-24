.PHONY: mypy
mypy:
	mypy --config-file=mypy.ini --no-site-packages .