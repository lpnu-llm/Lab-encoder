PYTHON ?= python3

.PHONY: all notebook figures check
all: figures notebook
notebook: lab_encoder.ipynb
lab_encoder.ipynb: lab_encoder.py tools/percent_to_ipynb.py
	$(PYTHON) tools/percent_to_ipynb.py $< $@
figures:
	$(PYTHON) tools/plot_figures.py
check:
	$(PYTHON) tools/check_lab.py
