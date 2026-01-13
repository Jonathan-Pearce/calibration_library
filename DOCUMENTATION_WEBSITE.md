# Documentation Website Setup Guide

The Calibration Toolbox documentation can be hosted as a website in three ways:

## 1. GitHub Pages (Recommended for Project Website)

### Setup Steps:

1. **Enable GitHub Pages** in your repository:
   - Go to repository Settings → Pages
   - Under "Source", select "GitHub Actions"
   - The `.github/workflows/docs.yml` workflow will automatically build and deploy

2. **Automatic Deployment**:
   - Documentation automatically rebuilds on every push to master/main
   - Website will be available at: `https://jonathan-pearce.github.io/calibration-toolbox/`

3. **Custom Domain** (optional):
   - Add a `CNAME` file to `docs/_build/html/` with your domain
   - Configure DNS settings
   - Update in GitHub Settings → Pages

## 2. Read the Docs (Recommended for Python Packages)

### Setup Steps:

1. **Import Project**:
   - Go to [readthedocs.org](https://readthedocs.org)
   - Sign in with GitHub
   - Import `calibration-toolbox` repository

2. **Automatic Configuration**:
   - `.readthedocs.yml` is already configured
   - Documentation will build automatically on commits
   - Website will be at: `https://calibration-toolbox.readthedocs.io/`

3. **Advantages**:
   - Automatic versioning (latest, stable, per-version docs)
   - PDF/ePub downloads
   - Search functionality
   - PR preview builds

## 3. Local Build (For Testing)

### Build Locally:

```bash
# Use the provided script
bash build_docs.sh

# Or manually
pip install sphinx sphinx-rtd-theme nbsphinx
cd docs
make html
```

### View Locally:

```bash
# Option 1: Simple HTTP server
python -m http.server 8000 --directory docs/_build/html
# Open http://localhost:8000

# Option 2: Direct file
# Open docs/_build/html/index.html in browser
```

## Documentation Structure

The website includes:

- **Home**: Overview and quick start
- **Installation**: Setup instructions
- **Quick Start**: Getting started guide
- **API Reference**: Complete function documentation
  - Metrics module
  - Visualization module
- **Examples**: Jupyter notebook tutorials
- **References**: Research papers and citations

## Updating Documentation

1. **Edit source files** in `docs/*.rst`
2. **Commit and push** changes
3. **Automatic rebuild**:
   - GitHub Pages: via GitHub Actions
   - ReadTheDocs: via webhook
   - Local: run `make html` in `docs/`

## Customization

### Theme
Currently using `sphinx_rtd_theme` (Read the Docs theme)
- Clean, professional look
- Mobile-responsive
- Good for Python packages

To change theme, edit `docs/conf.py`:
```python
html_theme = 'sphinx_rtd_theme'  # or 'alabaster', 'furo', etc.
```

### Logo/Favicon
Add to `docs/conf.py`:
```python
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'
```

### Custom CSS
Create `docs/_static/custom.css` and add to `conf.py`:
```python
html_static_path = ['_static']
html_css_files = ['custom.css']
```

## Troubleshooting

### Build Errors
```bash
# Clean build
cd docs
make clean
make html
```

### Missing Dependencies
```bash
pip install -r requirements-dev.txt
```

### Check Build Warnings
```bash
cd docs
make html 2>&1 | grep WARNING
```

## Next Steps

1. **Choose hosting**: GitHub Pages or ReadTheDocs (or both!)
2. **Build locally** to preview
3. **Enable hosting** in repository settings
4. **Add website URL** to repository description
5. **Update README** with live documentation link
