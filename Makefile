install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

run:
	python main.py

lint:
	pylint *.py

all: install lint format run
