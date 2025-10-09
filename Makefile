install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

run:
	python main.py

lint:
	pylint --disable=R,C main.py

all: install lint format run