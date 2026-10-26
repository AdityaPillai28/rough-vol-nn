# rough-vol-nn pipeline. On Windows without make, run the python commands directly.
PY ?= python
QUICK ?=
MARKET ?=

.PHONY: install data train eval figures demo test lint quick readme gif

install:
	$(PY) -m pip install -e ".[dev,app]"

data:            ## Monte Carlo training set (chunked, resumable) + reference set
	$(PY) -m roughvol data $(QUICK)

train:           ## Fit the grid MLP and the residual MLP
	$(PY) -m roughvol train $(QUICK)

eval:            ## Surrogate errors, calibration, recovery, speed, Heston baseline, market
	$(PY) -m roughvol eval $(QUICK) $(if $(MARKET),--market $(MARKET),)

figures:         ## All PNGs in assets/
	$(PY) -m roughvol figures $(QUICK)

readme:          ## Regenerate README tables from results/*.json
	$(PY) scripts/build_readme_tables.py

demo:            ## Streamlit app
	$(PY) -m streamlit run app/streamlit_app.py

gif:             ## Record assets/demo.gif from the running app (needs playwright)
	$(PY) scripts/record_demo.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

quick:           ## Whole pipeline on CPU on a tiny subset (< 5 min); used by CI
	$(PY) -m roughvol all --quick
