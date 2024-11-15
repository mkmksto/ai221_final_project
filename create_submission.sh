#!/bin/bash

# Create the submission folder
submission_folder="anila_cantor_quinto_submission"
mkdir -p "$submission_folder"

# Copy notebooks folder
cp -r notebooks "$submission_folder/"

# Copy src folder, excluding __pycache__
mkdir -p "$submission_folder/src"
(
  cd src
  find . -type d -name __pycache__ -prune -o -type f -print0 | 
  xargs -0 cp --parents -t "../$submission_folder/src/"
)

# Create the zip file
zip -r "${submission_folder}.zip" "$submission_folder"

echo "Submission zip file created: ${submission_folder}.zip"
