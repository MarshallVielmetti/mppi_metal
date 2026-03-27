
.PHONY: build
build:
	mkdir -p build && cd build && cmake .. && make -j$(sysctl -n hw.ncpu) 

.PHONY: test
test: build
	ctest --test-dir build --output-on-failure --verbose   