develop:
	pip install -q -U pip
	pip install -q -r requirements_dev.txt
	pre-commit install
	make precommit

install:
	pip install -q -r requirements.txt

full_install: develop install

precommit:
	git add .
	pre-commit run --all-files
	git add .

black:
	black src

isort:
	isort src

mypy:
	mypy src --ignore-missing-imports

pylint:
	pylint src --min-public-methods 0

flake:
	flake8 src --ignore=E501

check: black isort flake mypy precommit

fullcheck: check pylint
