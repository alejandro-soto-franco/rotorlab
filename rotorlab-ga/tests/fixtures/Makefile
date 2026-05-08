# Generate ground-truth Cayley fixture matching ganim's algorithm.
# Run from this directory: `make`

CXX        := g++
CXXFLAGS   := -std=c++23 -O2

build_fixtures: build_fixtures.cpp
	$(CXX) $(CXXFLAGS) build_fixtures.cpp -o build_fixtures

pga3_cayley.json: build_fixtures
	./build_fixtures > pga3_cayley.json

.PHONY: clean
clean:
	rm -f build_fixtures pga3_cayley.json
