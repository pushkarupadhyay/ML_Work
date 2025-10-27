# Data Directory

This directory contains datasets used across various machine learning projects.

## Structure

- **raw/**: Original, immutable data dumps
- **processed/**: Cleaned and preprocessed data ready for modeling
- **external/**: Data from external sources or third-party datasets

## Guidelines

- Keep raw data immutable - never modify original files
- Document data sources and preprocessing steps
- Large datasets should not be committed to git (they are gitignored)
- Use descriptive filenames with version/date information
