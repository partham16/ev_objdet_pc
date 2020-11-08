# https://stackoverflow.com/questions/7507810/how-to-source-a-script-in-a-makefile
SHELL := /bin/bash

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

git_user:
	git config --global user.name "partham16"
	git config --global user.email "43367843+partham16@users.noreply.github.com"

colab_rsa:
	# https://stackoverflow.com/a/793867/13070032
	mkdir -p /root/.ssh
	cp ../.ssh/ev_objdet_pc/* /root/.ssh/

chmodx:
	# https://github.com/pre-commit/pre-commit/issues/413#issuecomment-248978104
	chmod +x py36/bin/pre-commit
	chmod +x .git/hooks/pre-commit
	# for black flake pylint ...
	chmod +x py36/bin/*
	ls -la py36/bin/pre-commit

colab: git_user colab_rsa chmodx

# notebook:
# 	# https://github.com/ipython/ipython/issues/6193#issuecomment-350613300
# 	jupyter notebook --ip=127.0.0.1
# 	# --ip=0.0.0.0

# jupyter_venv:
# 	source py36/bin/activate && echo $(which python)
# 	# https://janakiev.com/blog/jupyter-virtual-envs/
# 	# https://ocefpaf.github.io/python4oceanographers/blog/2014/09/01/ipython_kernel/
# 	# https://stackoverflow.com/a/34533620/13070032
# 	python3 -m ipykernel install --user --name=py36
