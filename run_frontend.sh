#!/usr/bin/env bash
# Starts the Streamlit demo UI on http://localhost:8501
# Make sure run_backend.sh is already running in another terminal.
set -e
cd "$(dirname "$0")"
streamlit run streamlit_app.py
