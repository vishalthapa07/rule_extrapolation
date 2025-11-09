#!/bin/bash
# Helper script to format code with Black and commit
# Usage: ./commit.sh "commit message"

set -e

# Get commit message from argument
COMMIT_MSG="${1:-Update code}"

echo "Running Black formatter..."
# Run black through pre-commit to format files
pre-commit run black --all-files || true

echo "Staging formatted files..."
# Stage all modified files (including Black-formatted ones)
git add -u

echo "Committing with message: $COMMIT_MSG"
# Commit (pre-commit hooks will run again, but should pass now)
git commit -m "$COMMIT_MSG"

echo "✓ Commit successful!"

