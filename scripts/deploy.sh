#!/bin/bash
# This script sets up a GitHub Pages branch for Sphinx documentation
set -e

cd docs
make clean
make html
cd ..

git branch -D gh-pages || true
git checkout --orphan gh-pages
git rm -rf .
echo "This branch is for GitHub Pages" > README.md
git add README.md
git commit -m "Initialize gh-pages branch"
git push origin --force gh-pages

git checkout master

git checkout gh-pages
git checkout master -- docs/build/html
git checkout master -- .gitignore
cp -r docs/build/html/* .
git reset .
git add .gitignore
git add .
git commit -m "Deploy Sphinx documentation"
git push origin gh-pages

git checkout master