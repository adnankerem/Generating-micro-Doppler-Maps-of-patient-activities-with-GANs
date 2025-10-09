install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py

run:
	python src/Generating_micro-Doppler_Maps_of_patient_activities_with_Generative_Adversarial_Network.ipynb

lint:
    pip install pycodestyle pycodestyle_magic
	pycodestyle src/Generating_micro-Doppler_Maps_of_patient_activities_with_Generative_Adversarial_Network.ipynb

all: install lint format run
