#!/usr/bin/env bash
set -e

echo "Reading required versions from requirements.txt..."

while IFS== read -r pkg eq ver; do
  if [[ "$pkg" =~ ^\s*# ]] || [ -z "$pkg" ]; then
    continue
  fi
  echo "Installing $pkg==$ver ..."
  pip install "$pkg==$ver"
done < requirements-current.txt

