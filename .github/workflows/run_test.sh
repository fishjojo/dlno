#!/usr/bin/env bash
export PYTHONPATH=$(pwd):$PYTHONPATH

pytest "./dlno/test" --cov-report xml --cov=. --verbosity=1 --durations=10
