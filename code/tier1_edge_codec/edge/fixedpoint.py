"""
fixedpoint.py  --  integer primitives, blueprint 4.1.

The Python reference IS the specification, so it has to emulate C integer
arithmetic exactly. Python ints are arbitrary precision; C ints are not. Every
place where the two could disagree is funnelled through one of these helpers.

WHY THE WIDTH HELPERS ASSERT INSTEAD OF WRAPPING
------------------------------------------------
A wrapping i16() would make Python agree with C on the wrong answer, and the
dual-prediction scheme cannot survive a wrong answer: the Arduino and the
NanoPi would still agree with each other, but the reconstruction would be
silently garbage and no test would fire. An assertion turns the same event into
a loud test failure at the exact line that overflowed. Overflow here is a bug
in the codec, not a condition the codec is supposed to handle -- the clamps in
edge/codec.py are what keep the values in range, and if a clamp is ever removed
these assertions are the tripwire.

SHIFT SEMANTICS
---------------
`>>` on a negative Python int is a floor (arithmetic) shift: -5 >> 1 == -3.
gcc and avr-gcc implement `>>` on signed types as an arithmetic shift on every
target this project runs on (it is implementation-defined in C89/C99 and
defined as arithmetic in C++20; every real compiler does the arithmetic thing).
So `sar` is a plain `>>` and the C port is a plain `>>`. It is written out as a
named function anyway, because "the two languages happen to agree here" is
exactly the kind of fact that needs to be recorded rather than remembered.

NOTE that `sar` is NOT division: sar(-5, 1) == -3, while -5 // 2 in C is -2.
The NLMS update in codec.py uses sar deliberately -- the floor bias is
identical on both sides, so it cancels out of the synchronisation argument.
"""

INT16_MIN, INT16_MAX = -32768, 32767
INT32_MIN, INT32_MAX = -2147483648, 2147483647
INT64_MIN, INT64_MAX = -9223372036854775808, 9223372036854775807


def sar(x: int, n: int) -> int:
    """Arithmetic right shift by n. Floors toward -inf, matching C's `>>`."""
    assert n >= 0, f"negative shift count {n} is undefined behaviour in C"
    return x >> n


def shl(x: int, n: int) -> int:
    """Left shift. Provided so the C port has a named counterpart to sar."""
    assert n >= 0, f"negative shift count {n} is undefined behaviour in C"
    return x << n


def clamp(x: int, lo: int, hi: int) -> int:
    assert lo <= hi, (lo, hi)
    return lo if x < lo else (hi if x > hi else x)


def i16(x: int) -> int:
    assert INT16_MIN <= x <= INT16_MAX, f"int16 overflow: {x}"
    return x


def i32(x: int) -> int:
    assert INT32_MIN <= x <= INT32_MAX, f"int32 overflow: {x}"
    return x


def i64(x: int) -> int:
    assert INT64_MIN <= x <= INT64_MAX, f"int64 overflow: {x}"
    return x


def msb_index(x: int) -> int:
    """Position of the most significant set bit; msb_index(0) == 0.

    msb_index(1) == 0, msb_index(2) == 1, msb_index(255) == 7. Negative input
    is rejected: the only caller is the NLMS normaliser, whose argument is a
    sum of squares and therefore non-negative by construction. Accepting a
    negative here would hide that invariant breaking.

    The C version is a shift-and-count loop (or __builtin_clz on hosts). Both
    are exercised against the same table in tests/test_fixedpoint.py.
    """
    assert x >= 0, f"msb_index of a negative value: {x}"
    if x == 0:
        return 0
    return x.bit_length() - 1


def round_shift(x: int, n: int) -> int:
    """(x + (1 << (n-1))) >> n for n >= 1; identity for n == 0.

    Round-half-up in the direction of +inf (because the shift floors), which is
    what the C expression does verbatim. Do not "fix" this to round-half-even:
    the only requirement is that the two implementations agree, and half-up is
    one instruction cheaper on AVR.
    """
    assert n >= 0, f"negative shift count {n}"
    if n == 0:
        return x
    return (x + (1 << (n - 1))) >> n


def abs_i(x: int) -> int:
    """abs() with the int16 asymmetry made explicit: abs(INT16_MIN) does not
    fit in an int16. Callers hold residuals, which are clamped well inside the
    range, so this asserts rather than saturating."""
    a = -x if x < 0 else x
    assert a <= INT32_MAX, f"abs overflow: {x}"
    return a
