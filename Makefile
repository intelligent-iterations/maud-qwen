.PHONY: prepare test results train blog security security-hooks
prepare:
	python -m maud_qwen prepare
test:
	python -m pytest -q
results:
	python scripts/reproduce_results.py
train:
	bash scripts/run_first.sh
blog:
	python scripts/build_maud_blog.py

security:
	python3 scripts/security_check.py
security-hooks:
	bash scripts/install_security_hook.sh
