.PHONY: seed schema check check-schema test test-frontend typecheck

seed:
	cd backend && python -m scripts.seed

# Regenerate schema/doc_v21.schema.json + frontend/src/schema.gen.ts from the pydantic models
schema:
	cd backend && python -m scripts.gen_schema

# One command that runs everything
check: check-schema test test-frontend typecheck

check-schema:
	cd backend && python -m scripts.gen_schema --check

test:
	cd backend && python -m pytest -p no:warnings

test-frontend:
	cd frontend && pnpm test

typecheck:
	cd frontend && npx tsc --noEmit -p .
