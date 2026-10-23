# Makefile for GPT Model Implementation
# Builds the complete C implementation from TensorFlow Lite weights

CC = gcc
PYTHON ?= python3
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm
DEBUG_FLAGS = -g -O0 -DDEBUG
TARGET = gpt_model
SOURCE = main.c

# Header dependencies
HEADERS = arena_allocator.h \
          gpt_utils.h \
          gpt_embeddings_ops.h \
          gpt_layer_norm.h \
          gpt_attention.h \
          gpt_ffn.h \
          gpt_model.h \
          gpt_weights/gpt_weights.h \
          gpt_weights/gpt_constants.h \
          gpt_weights/gpt_embeddings.h \
          gpt_weights/gpt_layer1.h \
          gpt_weights/gpt_layer2.h

# Default target
all: $(TARGET) simple_test

# Main target
$(TARGET): $(SOURCE) $(HEADERS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

# Simple test target
simple_test: simple_test.c $(HEADERS)
	$(CC) $(CFLAGS) -o simple_test simple_test.c $(LDFLAGS)

# Debug build
debug: $(SOURCE) $(HEADERS)
	$(CC) $(DEBUG_FLAGS) -o $(TARGET)_debug $(SOURCE) $(LDFLAGS)

# Test with different optimization levels
test-O0: $(SOURCE) $(HEADERS)
	$(CC) -Wall -Wextra -O0 -std=c99 -lm -o $(TARGET)_O0 $(SOURCE)

test-O1: $(SOURCE) $(HEADERS)
	$(CC) -Wall -Wextra -O1 -std=c99 -lm -o $(TARGET)_O1 $(SOURCE)

test-O2: $(SOURCE) $(HEADERS)
	$(CC) -Wall -Wextra -O2 -std=c99 -lm -o $(TARGET)_O2 $(SOURCE)

test-O3: $(SOURCE) $(HEADERS)
	$(CC) -Wall -Wextra -O3 -std=c99 -lm -o $(TARGET)_O3 $(SOURCE)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run debug version
run-debug: debug
	./$(TARGET)_debug

# Profile with gprof
profile: $(SOURCE) $(HEADERS)
	$(CC) -Wall -Wextra -O2 -pg -std=c99 -lm -o $(TARGET)_profile $(SOURCE)
	./$(TARGET)_profile
	gprof $(TARGET)_profile gmon.out > profile_report.txt

# Memory check with valgrind (if available)
memcheck: debug
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)_debug

# Static analysis with cppcheck (if available)
static-check:
	cppcheck --enable=all --std=c99 $(SOURCE)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_O* $(TARGET)_profile simple_test
	rm -f gmon.out profile_report.txt

# Test targets
test: simple_test
	./simple_test

quick-test: simple_test
	./simple_test

parity-test:
	CC="$(CC)" CFLAGS="$(CFLAGS)" $(PYTHON) -B -m unittest discover -s tests -v

extract-weights:
	$(PYTHON) -B extract_weights_enhanced.py

# Create a minimal version (single header file)
single-header: $(HEADERS) $(SOURCE)
	@echo "Creating single header file..."
	@echo "/* GPT Model - Single Header Implementation */" > gpt_single.h
	@echo "/* Generated from TensorFlow Lite weights */" >> gpt_single.h
	@echo "#ifndef GPT_SINGLE_H" >> gpt_single.h
	@echo "#define GPT_SINGLE_H" >> gpt_single.h
	@echo "" >> gpt_single.h
	@cat arena_allocator.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_utils.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_weights/gpt_constants.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_weights/gpt_embeddings.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_weights/gpt_layer1.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_weights/gpt_layer2.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_embeddings_ops.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_layer_norm.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_attention.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_ffn.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@cat gpt_model.h | grep -v "#ifndef\|#define\|#endif" >> gpt_single.h
	@echo "" >> gpt_single.h
	@echo "#endif // GPT_SINGLE_H" >> gpt_single.h
	@echo "Single header file created: gpt_single.h"

# Build with single header
single: single-header
	$(CC) $(CFLAGS) -include gpt_single.h -o $(TARGET)_single $(SOURCE)

# Check header dependencies
check-headers:
	@echo "Checking header file dependencies..."
	@for header in $(HEADERS); do \
		if [ -f $$header ]; then \
			echo "✓ $$header"; \
		else \
			echo "✗ Missing: $$header"; \
		fi; \
	done

# Show file sizes
sizes:
	@echo "File sizes:"
	@ls -lh $(HEADERS) $(SOURCE) | awk '{print $$5 "\t" $$9}'
	@if [ -f $(TARGET) ]; then \
		echo ""; \
		echo "Executable size:"; \
		ls -lh $(TARGET) | awk '{print $$5 "\t" $$9}'; \
	fi

# # Generate documentation (requires doxygen)
# docs:
# 	@echo "Generating documentation..."
# 	@echo "PROJECT_NAME = \"GPT Model Implementation\"" > Doxyfile.tmp
# 	@echo "INPUT = ." >> Doxyfile.tmp
# 	@echo "FILE_PATTERNS = *.h *.c" >> Doxyfile.tmp
# 	@echo "GENERATE_HTML = YES" >> Doxyfile.tmp
# 	@echo "OUTPUT_DIRECTORY = docs" >> Doxyfile.tmp
# 	@doxygen Doxyfile.tmp
# 	@rm Doxyfile.tmp
# 	@echo "Documentation generated in docs/ directory"

# Performance test with different sequence lengths
perf-test: $(TARGET)
	@echo "Performance testing with different sequence lengths..."
	@echo "This will run multiple tests and may take a moment..."
	@time -p ./$(TARGET) 2>/dev/null || echo "Performance test completed"

# Install (copy to system location)
install: $(TARGET)
	@echo "Installing $(TARGET) to /usr/local/bin/"
	@sudo cp $(TARGET) /usr/local/bin/
	@echo "Installation complete"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build the main program (default)"
	@echo "  debug        - Build debug version"
	@echo "  run          - Build and run the program"
	@echo "  run-debug    - Build and run debug version"
	@echo "  test-O*      - Build with different optimization levels"
	@echo "  parity-test  - Compare C logits/intermediates with TFLite (requires Python dependencies)"
	@echo "  extract-weights - Regenerate C headers from the TFLite graph"
	@echo "  profile      - Build with profiling and generate report"
	@echo "  memcheck     - Run with valgrind memory checking"
	@echo "  static-check - Run static analysis with cppcheck"
	@echo "  single-header- Create single header file version"
	@echo "  single       - Build using single header"
	@echo "  check-headers- Verify all headers are present"
	@echo "  sizes        - Show file sizes"
	@echo "  docs         - Generate documentation (requires doxygen)"
	@echo "  perf-test    - Run performance tests"
	@echo "  clean        - Remove build artifacts"
	@echo "  install      - Install to system (requires sudo)"
	@echo "  help         - Show this help message"

.PHONY: all debug run run-debug clean single-header single check-headers sizes docs perf-test install help memcheck static-check profile test-O0 test-O1 test-O2 test-O3 test quick-test parity-test extract-weights