.PHONY: test coverage evaluate research showcase showcase-real data-uci-har cpp cpp-sanitize qualify clean
test:
	PYTHONPATH=src pytest -q
coverage:
	PYTHONPATH=src coverage run -m pytest -q && coverage report

evaluate:
	PYTHONPATH=src python scripts/evaluate.py
research:
	PYTHONPATH=src python scripts/research.py
showcase:
	PYTHONPATH=src python scripts/showcase.py --dataset digits --output-dir results/showcase/digits
data-uci-har:
	PYTHONPATH=src python scripts/fetch_dataset.py uci-har --destination data/uci-har
showcase-real: data-uci-har
	PYTHONPATH=src python scripts/showcase.py --dataset uci-har --data-dir data/uci-har --output-dir results/showcase/uci-har
cpp:
	cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build/cpp -j2 && ctest --test-dir build/cpp --output-on-failure
cpp-sanitize:
	cmake -S cpp -B build/cpp-sanitize -DCMAKE_BUILD_TYPE=RelWithDebInfo -DEDGEAI_ENABLE_SANITIZERS=ON && cmake --build build/cpp-sanitize -j2 && ctest --test-dir build/cpp-sanitize --output-on-failure
qualify: test coverage cpp cpp-sanitize evaluate research
clean:
	rm -rf build dist .coverage htmlcov .pytest_cache
