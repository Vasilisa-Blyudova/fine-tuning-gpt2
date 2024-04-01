#!/bin/bash

set -ex

echo -e '\n'
echo 'Running flake8 check...'

python -m flake8 fine_tuning_models

if [[ $? -ne 0 ]]; then
    echo "Flake8 check failed."
    exit 1
  else
    echo "Flake8 check passed."
  fi
