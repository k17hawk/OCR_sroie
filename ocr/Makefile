.PHONY: seed schema check check-schema test typecheck

seed:
	cd backend && python -m scripts.seed

# Regenerate schema/doc_v21.schema.json + frontend/src/schema.gen.ts from the pydantic models
schema:
	cd backend && python -m scripts.gen_schema

# One command that runs everything
check: check-schema test typecheck

check-schema:
	cd backend && python -m scripts.gen_schema --check

test:
	cd backend && python -m pytest -p no:warnings

typecheck:
	cd frontend && npx tsc --noEmit -p .
