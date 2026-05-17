# Convenience targets for the experiment registry's read/write pipeline.
#
# Quick reference:
#   make views          - Refresh views/{gen_df,baseline_df,baseline_p}.parquet
#                         (idempotent: no-op if registry mtime hasn't changed).
#   make views-force    - Rebuild views from scratch, ignoring the cache key.
#   make views-status   - Print registry mtime + per-view manifest entries.
#   make backfill       - Re-stamp animal_counts to v4 (fast-path most days).
#   make test           - Run the pytest suite.
#   make cron-install   - Print a crontab snippet you can paste with
#                         `crontab -e` to keep views fresh every 30 min.
#
# All recipes assume the venv has been created and SL_VENV / .venv exists.
# Override with `make views PYTHON=/some/other/python`.

REPO_ROOT := $(shell pwd)
PYTHON ?= $(REPO_ROOT)/.venv/bin/python
LOG_DIR := $(REPO_ROOT)/logs

.PHONY: help views views-force views-status backfill backfill-force \
        test test-animals cron-install logs-dir

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z][a-zA-Z0-9_-]+:.*##/ {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

logs-dir:
	@mkdir -p $(LOG_DIR)

views: logs-dir  ## Idempotent rebuild of the parquet views
	$(PYTHON) scripts/rebuild_views.py

views-force: logs-dir  ## Force-rebuild every view from scratch
	$(PYTHON) scripts/rebuild_views.py --force

views-status:  ## Show registry mtime + per-view freshness
	$(PYTHON) scripts/views_status.py

backfill: logs-dir  ## Re-stamp animal_counts (fast-path) and rebuild views
	$(PYTHON) scripts/backfill_animal_counts.py --workers 32 \
	    2>&1 | tee $(LOG_DIR)/backfill_$$(date +%Y%m%d_%H%M%S).log
	$(MAKE) views

backfill-force: logs-dir  ## Force re-classification of every cached entry (slow)
	$(PYTHON) scripts/backfill_animal_counts.py --workers 32 --force \
	    2>&1 | tee $(LOG_DIR)/backfill_force_$$(date +%Y%m%d_%H%M%S).log
	$(MAKE) views-force

test:  ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

test-animals:  ## Run just the classifier / cache invariants tests
	$(PYTHON) -m pytest tests/test_animals.py -v

cron-install:  ## Print a crontab snippet (does NOT modify your crontab)
	@echo '# Add this to your crontab ("crontab -e") to keep the parquet views'
	@echo '# fresh every 30 minutes. The wrapper script holds a lockfile so'
	@echo '# overlapping invocations are safe; output goes to logs/.'
	@echo ''
	@echo '*/30 * * * * $(REPO_ROOT)/scripts/cron_rebuild_views.sh'
	@echo ''
	@echo 'Then check it landed with:'
	@echo '  crontab -l | grep cron_rebuild_views'
