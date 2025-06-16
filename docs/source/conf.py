# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import shutil

project_root = os.path.abspath('../..')

print("Project root directory:", project_root)
sys.path.insert(0, project_root)

source_assets = os.path.join(project_root, "assets")

target_static = os.path.join('_static')

if os.path.exists(source_assets):
    shutil.copytree(source_assets, target_static, dirs_exist_ok=True)

project = 'Raggie'
copyright = '2025, Yamac Eren Ay'
author = 'Yamac Eren Ay'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
]

templates_path = [os.path.join('_templates')]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_static_path = [os.path.join('_static')]
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'titles_only': False
}
html_logo = os.path.join('_static/logo.png')
html_favicon = os.path.join('_static/favicon.ico')
pygments_style = 'sphinx'