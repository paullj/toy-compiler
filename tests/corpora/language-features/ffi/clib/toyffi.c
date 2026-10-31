// The C library the FFI fixtures link. build.zig builds it as libtoyffi.dylib into
// zig-out/test-lib, with the @rpath install name a Zig-built dylib gets.
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

long toyffi_add(long a, long b) { return a + b; }

int32_t toyffi_neg32(int32_t x) { return -x; }

double toyffi_scale(double x, double k) { return x * k; }

double toyffi_mix(long n, double x) { return (double)n + x; }

struct toyffi_counter {
    long n;
};

void *toyffi_counter_new(long start) {
    struct toyffi_counter *c = malloc(sizeof *c);
    c->n = start;
    return c;
}

long toyffi_counter_bump(void *p) { return ++((struct toyffi_counter *)p)->n; }

void toyffi_counter_free(void *p) { free(p); }

int8_t toyffi_neg8(int8_t x) { return -x; }

uint8_t toyffi_inc8(uint8_t x) { return x + 1; }

int16_t toyffi_sum16(int8_t a, int16_t b, int32_t c) { return a + b + c; }

uint32_t toyffi_max32(void) { return 0xFFFFFFFFu; }

bool toyffi_isneg(long x) { return x < 0; }

// Past x7 Apple's arm64 ABI packs each argument at its own size and alignment.
long toyffi_stack(int8_t a, long b, long c, long d, long e, long f, long g, long h, int8_t i,
                  int16_t j, int8_t k, int32_t l, bool m, uint8_t n) {
    return a + b + c + d + e + f + g + h + i * 10 + j * 100 + k * 1000 + l * 10000 + m * 100000 +
           n * 1000000;
}
