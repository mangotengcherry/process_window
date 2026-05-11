"""Streamlit entrypoint.

Run with:
    streamlit run streamlit_app.py

This thin wrapper exists because `streamlit run src/app.py` executes the file
as a top-level script, which breaks the relative imports (`from .data_loader`
...) inside src/app.py. Running this wrapper instead lets Python import
src.app as a proper package member, so all relative imports resolve correctly.
"""
from src.app import run_streamlit

run_streamlit()
