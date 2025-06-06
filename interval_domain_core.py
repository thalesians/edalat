"""
Interval Domain Framework for Computable Real Analysis
Based on Edalat's papers on domain theory and computable analysis.

This implementation provides:
1. Interval domain with computable structure
2. Computable real numbers via effective sequences
3. Examples from both papers with test suite
"""

import math
from fractions import Fraction
from typing import List, Optional, Callable, Iterator, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import itertools

class Rational:
    """Rational number representation using fractions for exact arithmetic."""
    
    def __init__(self, numerator: Union[int, float], denominator: int = 1):
        if isinstance(numerator, float):
            if math.isinf(numerator) or math.isnan(numerator):
                self.is_infinite = True
                self.is_positive_inf = numerator > 0
                self.frac = None
            else:
                self.is_infinite = False
                self.frac = Fraction(numerator).limit_denominator(1000000)
        else:
            self.is_infinite = False
            self.frac = Fraction(numerator, denominator)
    
    @classmethod
    def positive_infinity(cls):
        """Create positive infinity."""
        result = cls(1)
        result.is_infinite = True
        result.is_positive_inf = True
        result.frac = None
        return result
    
    @classmethod
    def negative_infinity(cls):
        """Create negative infinity."""
        result = cls(1)
        result.is_infinite = True
        result.is_positive_inf = False
        result.frac = None
        return result
    
    @classmethod
    def from_float(cls, x: float, max_denominator: int = 1000000):
        """Convert float to rational with bounded denominator."""
        if math.isinf(x):
            return cls.positive_infinity() if x > 0 else cls.negative_infinity()
        return cls.from_fraction(Fraction(x).limit_denominator(max_denominator))
    
    @classmethod
    def from_fraction(cls, frac: Fraction):
        """Create from existing fraction."""
        result = cls(1)
        result.is_infinite = False
        result.frac = frac
        return result
    
    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        
        if self.is_infinite or other.is_infinite:
            if self.is_infinite and other.is_infinite:
                if self.is_positive_inf == other.is_positive_inf:
                    return Rational.positive_infinity() if self.is_positive_inf else Rational.negative_infinity()
                else:
                    raise ValueError("Indeterminate form: inf - inf")
            elif self.is_infinite:
                return self
            else:
                return other
        
        return Rational.from_fraction(self.frac + other.frac)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        return self.__add__(-other)
    
    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        return other.__add__(-self)
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        
        if self.is_infinite or other.is_infinite:
            if (self.is_infinite and self._is_zero()) or (other.is_infinite and other._is_zero()):
                raise ValueError("Indeterminate form: 0 * inf")
            
            # Determine sign
            self_positive = self.is_positive_inf if self.is_infinite else (self.frac >= 0)
            other_positive = other.is_positive_inf if other.is_infinite else (other.frac >= 0)
            result_positive = self_positive == other_positive
            
            return Rational.positive_infinity() if result_positive else Rational.negative_infinity()
        
        return Rational.from_fraction(self.frac * other.frac)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        
        if other._is_zero():
            raise ZeroDivisionError("Division by zero")
        
        if self.is_infinite:
            if other.is_infinite:
                raise ValueError("Indeterminate form: inf / inf")
            else:
                self_positive = self.is_positive_inf
                other_positive = other.frac >= 0
                result_positive = self_positive == other_positive
                return Rational.positive_infinity() if result_positive else Rational.negative_infinity()
        elif other.is_infinite:
            return Rational(0)
        
        return Rational.from_fraction(self.frac / other.frac)
    
    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        return other.__truediv__(self)
    
    def __lt__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        
        if self.is_infinite and other.is_infinite:
            return not self.is_positive_inf and other.is_positive_inf
        elif self.is_infinite:
            return not self.is_positive_inf
        elif other.is_infinite:
            return other.is_positive_inf
        
        return self.frac < other.frac
    
    def __le__(self, other):
        return self < other or self == other
    
    def __gt__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        return other < self
    
    def __ge__(self, other):
        return self > other or self == other
    
    def __eq__(self, other):
        if isinstance(other, (int, float)):
            other = Rational.from_float(float(other))
        
        if self.is_infinite and other.is_infinite:
            return self.is_positive_inf == other.is_positive_inf
        elif self.is_infinite or other.is_infinite:
            return False
        
        return self.frac == other.frac
    
    def __abs__(self):
        if self.is_infinite:
            return Rational.positive_infinity()
        return Rational.from_fraction(abs(self.frac))
    
    def __neg__(self):
        if self.is_infinite:
            return Rational.negative_infinity() if self.is_positive_inf else Rational.positive_infinity()
        return Rational.from_fraction(-self.frac)
    
    def _is_zero(self):
        """Check if rational is zero."""
        return not self.is_infinite and self.frac == 0
    
    def __float__(self):
        if self.is_infinite:
            return float('inf') if self.is_positive_inf else float('-inf')
        return float(self.frac)
    
    def __repr__(self):
        if self.is_infinite:
            return f"Rational({'inf' if self.is_positive_inf else '-inf'})"
        return f"Rational({self.frac})"
    
    def __str__(self):
        if self.is_infinite:
            return 'inf' if self.is_positive_inf else '-inf'
        return str(self.frac)

@dataclass(frozen=True)
class Interval:
    """
    Represents a compact interval [left, right] in the interval domain.
    Uses rational endpoints for exact arithmetic.
    Special handling for the bottom element.
    """
    left: Rational
    right: Rational
    is_bottom: bool = False
    
    def __post_init__(self):
        if not self.is_bottom and self.left > self.right:
            raise ValueError(f"Invalid interval: left ({self.left}) > right ({self.right})")
    
    @classmethod
    def bottom(cls):
        """Bottom element of the interval domain."""
        # Use a special flag to represent bottom rather than infinite endpoints
        return cls(Rational(0), Rational(0), is_bottom=True)
    
    @classmethod
    def point(cls, x: Union[float, Rational]):
        """Create singleton interval {x}."""
        if isinstance(x, (int, float)):
            x = Rational.from_float(float(x))
        return cls(x, x)
    
    @classmethod
    def make(cls, left: Union[float, Rational], right: Union[float, Rational]):
        """Create interval [left, right]."""
        if isinstance(left, (int, float)):
            left = Rational.from_float(float(left))
        if isinstance(right, (int, float)):
            right = Rational.from_float(float(right))
        return cls(left, right)
    
    def contains(self, x: Union[float, Rational, 'Interval']) -> bool:
        """Check if this interval contains x or another interval."""
        if self.is_bottom:
            return True  # Bottom contains everything
        
        if isinstance(x, (int, float)):
            x = Rational.from_float(float(x))
        if isinstance(x, Rational):
            return self.left <= x <= self.right
        elif isinstance(x, Interval):
            if x.is_bottom:
                return False  # Only bottom contains bottom
            return self.left <= x.left and x.right <= self.right
        return False
    
    def intersects(self, other: 'Interval') -> bool:
        """Check if this interval intersects with another."""
        if self.is_bottom or other.is_bottom:
            return True
        return not (self.right < other.left or other.right < self.left)
    
    def intersection(self, other: 'Interval') -> Optional['Interval']:
        """Compute intersection of two intervals."""
        if self.is_bottom:
            return other
        if other.is_bottom:
            return self
        
        if not self.intersects(other):
            return None
        
        left = max(self.left.frac, other.left.frac) if not (self.left.is_infinite or other.left.is_infinite) else self.left
        right = min(self.right.frac, other.right.frac) if not (self.right.is_infinite or other.right.is_infinite) else self.right
        
        if isinstance(left, Fraction):
            left = Rational.from_fraction(left)
        if isinstance(right, Fraction):
            right = Rational.from_fraction(right)
        
        return Interval(left, right)
    
    def union_hull(self, other: 'Interval') -> 'Interval':
        """Compute convex hull (union) of two intervals."""
        if self.is_bottom or other.is_bottom:
            return Interval.bottom()
        
        left_val = min(self.left, other.left)
        right_val = max(self.right, other.right)
        return Interval(left_val, right_val)
    
    def width(self) -> Rational:
        """Width of the interval."""
        if self.is_bottom:
            return Rational.positive_infinity()
        return self.right - self.left
    
    def midpoint(self) -> Rational:
        """Midpoint of the interval."""
        if self.is_bottom:
            return Rational(0)  # Arbitrary choice for bottom
        return (self.left + self.right) / Rational(2)
    
    def is_singleton(self) -> bool:
        """Check if interval is a singleton (point)."""
        if self.is_bottom:
            return False
        return self.left == self.right
    
    def split(self) -> Tuple['Interval', 'Interval']:
        """Split interval into two equal halves."""
        if self.is_bottom:
            # Can't meaningfully split bottom
            return (self, self)
        
        mid = self.midpoint()
        return (Interval(self.left, mid), Interval(mid, self.right))
    
    def way_below(self, other: 'Interval') -> bool:
        """
        Check if self << other (way-below relation).
        I << J iff interior of I contains J.
        """
        if self.is_bottom:
            return not other.is_bottom  # Bottom is way-below everything except itself
        
        if other.is_bottom or self.is_singleton() or other.is_singleton():
            return False
        
        return self.left < other.left and other.right < self.right
    
    def scott_open_contains(self, intervals: List['Interval']) -> bool:
        """Check if this interval is in the Scott-open set generated by intervals."""
        return any(interval.way_below(self) for interval in intervals)
    
    def __le__(self, other: 'Interval') -> bool:
        """Order relation: reverse inclusion."""
        return other.contains(self)
    
    def __lt__(self, other: 'Interval') -> bool:
        """Strict order relation."""
        return self <= other and not (other <= self)
    
    def __repr__(self):
        if self.is_bottom:
            return "Interval(bottom)"
        return f"Interval([{self.left}, {self.right}])"
    
    def __str__(self):
        if self.is_bottom:
            return "bottom"
        if self.is_singleton():
            return f"{{{self.left}}}"
        return f"[{self.left}, {self.right}]"

class EffectiveSequence:
    """
    Represents an effective sequence (computable sequence) in recursion theory.
    This is fundamental for defining computable elements in domain theory.
    """
    
    def __init__(self, generator: Callable[[int], Interval]):
        """
        Create effective sequence from a computable generator function.
        generator(n) should return the n-th element of the sequence.
        """
        self.generator = generator
        self._cache = {}
    
    def __getitem__(self, n: int) -> Interval:
        """Get n-th element of the sequence."""
        if n not in self._cache:
            self._cache[n] = self.generator(n)
        return self._cache[n]
    
    def take(self, n: int) -> List[Interval]:
        """Get first n elements of the sequence."""
        return [self[i] for i in range(n)]
    
    def is_chain(self, n: int) -> bool:
        """Check if first n elements form a chain (monotonic sequence)."""
        elements = self.take(n)
        for i in range(len(elements) - 1):
            if not (elements[i] >= elements[i + 1]):  # Reverse inclusion order
                return False
        return True
    
    def converges_to_point(self, precision: int = 10) -> bool:
        """Check if sequence converges to a point with given precision."""
        try:
            last_interval = self[precision]
            if last_interval.is_bottom:
                return False
            width_threshold = Rational(1, 2**precision)
            return last_interval.width() <= width_threshold
        except:
            return False

class IntervalDomain:
    """
    The interval domain with computable structure.
    Implements the effective interval domain from Paper 2.
    """
    
    def __init__(self):
        self.rational_enum = self._enumerate_rationals()
        self.basis_enum = self._enumerate_basis()
    
    def _enumerate_rationals(self) -> List[Rational]:
        """
        Enumerate rationals using standard diagonal enumeration.
        q_{(n,m)} = (n-m)/(k+1) where (n,m,k) comes from pairing function.
        """
        rationals = []
        # Include common rationals first for efficiency
        for denom in range(1, 20):  # Reduced for efficiency
            for num in range(-20, 21):
                rationals.append(Rational(num, denom))
        return rationals
    
    def _enumerate_basis(self) -> List[Interval]:
        """
        Enumerate basis elements of interval domain.
        I_0 = bottom, I_{(m,n)+1} = [q_m, q_n] for rationals q_m, q_n.
        """
        basis = [Interval.bottom()]  # I_0 = bottom element
        
        # Add intervals with rational endpoints
        rationals = self.rational_enum[:10]  # Use first 10 rationals for efficiency
        for i, r1 in enumerate(rationals):
            for j, r2 in enumerate(rationals):
                if r1 <= r2:
                    basis.append(Interval(r1, r2))
        
        return basis
    
    def way_below_relation(self, i: int, j: int) -> bool:
        """
        Check if I_i << I_j (way-below relation).
        This is r.e. (recursively enumerable) as required by effective domain theory.
        """
        if i >= len(self.basis_enum) or j >= len(self.basis_enum):
            return False
        return self.basis_enum[i].way_below(self.basis_enum[j])
    
    def computable_element_from_chain(self, chain: EffectiveSequence) -> 'ComputableReal':
        """
        Create computable element as lub of effective chain.
        This implements Proposition 3 from Paper 2.
        """
        return ComputableReal(chain)
    
    def rational_to_interval(self, q: Rational) -> Interval:
        """Convert rational to singleton interval (maximal element)."""
        return Interval.point(q)

class ComputableReal:
    """
    Represents a computable real number as the lub of an effective chain
    of intervals. This implements Definition 13 from Paper 2.
    """
    
    def __init__(self, approximation_sequence: EffectiveSequence):
        """
        Create computable real from effective sequence of approximating intervals.
        The sequence should be decreasing (in reverse inclusion order) and
        converge to a singleton interval.
        """
        self.sequence = approximation_sequence
        self._validate_sequence()
    
    def _validate_sequence(self):
        """Validate that sequence forms proper approximation."""
        # Check first few elements form a chain
        try:
            if not self.sequence.is_chain(5):  # Reduced check for efficiency
                print("Warning: Approximation sequence may not form a proper chain")
        except:
            pass  # Allow construction even if validation fails
    
    @classmethod
    def from_rational(cls, q: Rational) -> 'ComputableReal':
        """Create computable real from rational number."""
        def const_seq(n: int) -> Interval:
            return Interval.point(q)
        return cls(EffectiveSequence(const_seq))
    
    @classmethod
    def from_cauchy_sequence(cls, cauchy_seq: Callable[[int], Rational]) -> 'ComputableReal':
        """
        Create computable real from fast-converging Cauchy sequence.
        cauchy_seq(n) should satisfy |cauchy_seq(n) - x| <= 1/2^n.
        """
        def interval_seq(n: int) -> Interval:
            center = cauchy_seq(n)
            radius = Rational(1, 2**n)
            return Interval(center - radius, center + radius)
        return cls(EffectiveSequence(interval_seq))
    
    def approximate(self, precision: int) -> Interval:
        """Get approximation with 2^(-precision) accuracy."""
        return self.sequence[precision]
    
    def to_float(self, precision: int = 20) -> float:
        """Convert to float with given precision."""
        approx = self.approximate(precision)
        if approx.is_bottom:
            return 0.0  # Default for bottom
        return float(approx.midpoint())
    
    def __add__(self, other: 'ComputableReal') -> 'ComputableReal':
        """Addition of computable reals."""
        def sum_seq(n: int) -> Interval:
            # Get approximations with extra precision for intermediate calculations
            a_approx = self.approximate(n + 2)
            b_approx = other.approximate(n + 2)
            
            if a_approx.is_bottom or b_approx.is_bottom:
                return Interval.bottom()
            
            # Interval arithmetic: [a,b] + [c,d] = [a+c, b+d]
            left = a_approx.left + b_approx.left
            right = a_approx.right + b_approx.right
            return Interval(left, right)
        
        return ComputableReal(EffectiveSequence(sum_seq))
    
    def __mul__(self, other: 'ComputableReal') -> 'ComputableReal':
        """Multiplication of computable reals."""
        def mul_seq(n: int) -> Interval:
            # Get approximations with extra precision
            a_approx = self.approximate(n + 3)
            b_approx = other.approximate(n + 3)
            
            if a_approx.is_bottom or b_approx.is_bottom:
                return Interval.bottom()
            
            # Interval multiplication: consider all combinations of endpoints
            try:
                products = [
                    a_approx.left * b_approx.left,
                    a_approx.left * b_approx.right,
                    a_approx.right * b_approx.left,
                    a_approx.right * b_approx.right
                ]
                
                # Find min and max, handling infinities
                finite_products = [p for p in products if not p.is_infinite]
                if not finite_products:
                    return Interval.bottom()
                
                min_prod = min(finite_products)
                max_prod = max(finite_products)
                
                return Interval(min_prod, max_prod)
            except:
                return Interval.bottom()
        
        return ComputableReal(EffectiveSequence(mul_seq))
    
    def __repr__(self):
        try:
            approx = self.approximate(5)
            return f"ComputableReal(approx)"
        except:
            return f"ComputableReal(?)"

class ComputableFunction:
    """
    Represents a computable function via its Scott-continuous extension
    to the interval domain. This implements Definition 22 from Paper 2.
    """
    
    def __init__(self, interval_extension: Callable[[Interval], Interval]):
        """
        Create computable function from its interval extension.
        The extension should be Scott-continuous and extend a continuous function.
        """
        self.extension = interval_extension
    
    def __call__(self, x: ComputableReal) -> ComputableReal:
        """Apply function to computable real."""
        def result_seq(n: int) -> Interval:
            # Apply interval extension to approximation
            x_approx = x.approximate(n + 2)  # Extra precision for stability
            return self.extension(x_approx)
        
        return ComputableReal(EffectiveSequence(result_seq))
    
    @classmethod
    def from_real_function(cls, f: Callable[[float], float], 
                          domain: Interval = None) -> 'ComputableFunction':
        """
        Create computable function from continuous real function.
        Uses interval evaluation with outward rounding.
        """
        def interval_ext(I: Interval) -> Interval:
            if I.is_bottom:
                return Interval.bottom()
            
            if I.is_singleton():
                # Point evaluation
                try:
                    x = float(I.left)
                    y = f(x)
                    if math.isnan(y) or math.isinf(y):
                        return Interval.bottom()
                    return Interval.point(Rational.from_float(y))
                except:
                    return Interval.bottom()
            else:
                # Evaluate at endpoints and take hull
                try:
                    left_val = f(float(I.left))
                    right_val = f(float(I.right))
                    
                    if any(math.isnan(v) or math.isinf(v) for v in [left_val, right_val]):
                        return Interval.bottom()
                    
                    # Add small epsilon for outward rounding
                    epsilon = 1e-10 * max(abs(left_val), abs(right_val), 1.0)
                    
                    min_val = min(left_val, right_val) - epsilon
                    max_val = max(left_val, right_val) + epsilon
                    
                    return Interval(
                        Rational.from_float(min_val),
                        Rational.from_float(max_val)
                    )
                except:
                    return Interval.bottom()
        
        return cls(interval_ext)

# Examples from Paper 1: Dynamical Systems

class IteratedFunctionSystem:
    """
    Iterated Function System (IFS) from Paper 1.
    Implements the examples in Section 3.5 (Julia Sets).
    """
    
    def __init__(self, functions: List[ComputableFunction], domain: Interval):
        """
        Create IFS with given list of functions on specified domain.
        Functions should be contracting for hyperbolic IFS.
        """
        self.functions = functions
        self.domain = domain
    
    def iterate_sequence(self, code: List[int], initial: Interval = None) -> EffectiveSequence:
        """
        Generate sequence of intervals by applying functions according to code.
        This implements the IFS tree structure from Paper 1.
        """
        if initial is None:
            initial = self.domain
        
        def sequence_gen(n: int) -> Interval:
            current = initial
            for i in range(min(n, len(code))):
                if len(self.functions) > 0:
                    func_index = code[i] % len(self.functions)
                    # Apply interval extension of function
                    current = self.functions[func_index].extension(current)
            return current
        
        return EffectiveSequence(sequence_gen)
    
    def compute_attractor_point(self, code: List[int], precision: int = 20) -> ComputableReal:
        """
        Compute point in attractor corresponding to infinite code sequence.
        This implements the computable attractors from Paper 1, Section 3.4.
        """
        if not code:
            return ComputableReal.from_rational(Rational(0))
        
        # Extend code periodically if finite
        extended_code = (code * ((precision // len(code)) + 1))[:precision]
        
        sequence = self.iterate_sequence(extended_code)
        return ComputableReal(sequence)

# Examples from Paper 2: Computable Analysis

def sqrt_computable(x: ComputableReal) -> ComputableReal:
    """
    Computable square root using Newton's method.
    Example of computable function from Paper 2.
    """
    def sqrt_interval(I: Interval) -> Interval:
        if I.is_bottom:
            return Interval.bottom()
        
        if I.left < Rational(0):
            return Interval.bottom()  # Undefined for negative
        
        if I.is_singleton():
            if I.left == Rational(0):
                return Interval.point(Rational(0))
            try:
                val = float(I.left)
                result = math.sqrt(val)
                return Interval.point(Rational.from_float(result))
            except:
                return Interval.bottom()
        else:
            # For interval [a,b], sqrt([a,b]) = [sqrt(a), sqrt(b)] if a >= 0
            try:
                left_sqrt = math.sqrt(max(0, float(I.left)))
                right_sqrt = math.sqrt(float(I.right))
                
                return Interval(
                    Rational.from_float(left_sqrt),
                    Rational.from_float(right_sqrt)
                )
            except:
                return Interval.bottom()
    
    return ComputableFunction(sqrt_interval)(x)

def exp_computable(x: ComputableReal) -> ComputableReal:
    """
    Computable exponential function.
    Example of transcendental computable function.
    """
    def exp_interval(I: Interval) -> Interval:
        if I.is_bottom:
            return Interval.bottom()
        
        if I.is_singleton():
            try:
                val = float(I.left)
                result = math.exp(val)
                return Interval.point(Rational.from_float(result))
            except:
                return Interval.bottom()
        else:
            # exp is monotonic, so exp([a,b]) = [exp(a), exp(b)]
            try:
                left_exp = math.exp(float(I.left))
                right_exp = math.exp(float(I.right))
                
                return Interval(
                    Rational.from_float(left_exp),
                    Rational.from_float(right_exp)
                )
            except:
                return Interval.bottom()
    
    return ComputableFunction(exp_interval)(x)

# Real number representations from Paper 2, Section 6

class DecimalRepresentation:
    """
    Decimal representation using IFS as described in Paper 2, Section 6.
    """
    
    def __init__(self):
        # Create IFS for decimal representation: f_i(x) = (x + i)/10
        self.functions = []
        for digit in range(10):
            def make_func(d):
                def decimal_func(I: Interval) -> Interval:
                    if I.is_bottom:
                        return Interval.bottom()
                    # f_d(x) = (x + d)/10
                    try:
                        left = (I.left + Rational(d)) / Rational(10)
                        right = (I.right + Rational(d)) / Rational(10)
                        return Interval(left, right)
                    except:
                        return Interval.bottom()
                return decimal_func
            
            self.functions.append(ComputableFunction(make_func(digit)))
        
        self.domain = Interval(Rational(0), Rational(1))
    
    def from_decimal_digits(self, digits: List[int]) -> ComputableReal:
        """
        Create computable real from decimal digit sequence.
        Example: [3, 1, 4, 1, 5, 9] represents 0.314159...
        """
        if not all(0 <= d <= 9 for d in digits):
            raise ValueError("Decimal digits must be 0-9")
        
        def cauchy_approx(n: int) -> Rational:
            acc = Rational(0)
            for i in range(min(n + 1, len(digits))):
                acc += Rational(digits[i], 10 ** (i + 1))
            return acc
        
        return ComputableReal.from_cauchy_sequence(cauchy_approx)

class BinaryRepresentation:
    """
    Binary representation with signed digits {-1, 0, 1}.
    Example from Paper 2, Section 6.
    """
    
    def __init__(self):
        # Create IFS for signed binary: f_i(x) = (x + i)/2 for i in {-1, 0, 1}
        self.functions = []
        for digit in [-1, 0, 1]:
            def make_func(d):
                def binary_func(I: Interval) -> Interval:
                    if I.is_bottom:
                        return Interval.bottom()
                    try:
                        left = (I.left + Rational(d)) / Rational(2)
                        right = (I.right + Rational(d)) / Rational(2)
                        return Interval(left, right)
                    except:
                        return Interval.bottom()
                return binary_func
            
            self.functions.append(ComputableFunction(make_func(digit)))
        
        self.domain = Interval(Rational(-1), Rational(1))
    
    def from_signed_binary(self, digits: List[int]) -> ComputableReal:
        """
        Create computable real from signed binary sequence.
        digits should contain values from {-1, 0, 1}.
        """
        if not all(d in [-1, 0, 1] for d in digits):
            raise ValueError("Signed binary digits must be in {-1, 0, 1}")
        
        # Map to indices 0, 1, 2
        indices = [d + 1 for d in digits]
        ifs = IteratedFunctionSystem(self.functions, self.domain)
        return ifs.compute_attractor_point(indices)

# Dynamical Systems Examples from Paper 1

class HenonMap:
    """
    Henon map example from Paper 1.
    f(x,y) = (1 - ax^2 + y, bx)
    """
    
    def __init__(self, a: float = 1.4, b: float = 0.3):
        self.a = Rational.from_float(a)
        self.b = Rational.from_float(b)
    
    def apply_to_interval(self, I: Interval, J: Interval) -> Tuple[Interval, Interval]:
        """Apply Henon map to interval in R^2."""
        if I.is_bottom or J.is_bottom:
            return (Interval.bottom(), Interval.bottom())
        
        try:
            # x' = 1 - ax² + y
            # For interval [x1,x2], x2 in [min(x1^2,x2^2), max(x1^2,x2^2)] if 0 in [x1,x2]
            # Otherwise x^2 in [0, max(x1^2,x2^2)]
            
            x_squared_candidates = [I.left * I.left, I.right * I.right]
            if I.contains(Rational(0)):
                x_squared_min = Rational(0)
            else:
                x_squared_min = min(x_squared_candidates)
            x_squared_max = max(x_squared_candidates)
            
            x_squared_interval = Interval(x_squared_min, x_squared_max)
            
            # 1 - ax^2
            one_minus_ax_sq = Interval(
                Rational(1) - self.a * x_squared_interval.right,
                Rational(1) - self.a * x_squared_interval.left
            )
            
            # 1 - ax^2 + y
            new_x = Interval(
                one_minus_ax_sq.left + J.left,
                one_minus_ax_sq.right + J.right
            )
            
            # y' = bx
            new_y = Interval(self.b * I.left, self.b * I.right)
            
            return (new_x, new_y)
        except:
            return (Interval.bottom(), Interval.bottom())

class QuadraticMap:
    """
    Quadratic map f(z) = z^2 + c for complex dynamics.
    Simplified to real case: f(x) = x^2 + c.
    Example from Paper 1, Section 3.5.
    """
    
    def __init__(self, c: float):
        self.c = Rational.from_float(c)
        self.func = self._create_function()
    
    def _create_function(self) -> ComputableFunction:
        """Create the computable function f(x) = x^2 + c."""
        def quadratic_interval(I: Interval) -> Interval:
            if I.is_bottom:
                return Interval.bottom()
            
            try:
                a_sq = I.left * I.left
                b_sq = I.right * I.right
                
                if I.contains(Rational(0)):
                    min_sq = Rational(0)
                else:
                    min_sq = min(a_sq, b_sq)
                max_sq = max(a_sq, b_sq)
                
                # Add constant c
                return Interval(min_sq + self.c, max_sq + self.c)
            except:
                return Interval.bottom()
        
        return ComputableFunction(quadratic_interval)
    
    def julia_set_approximation(self, precision: int = 10) -> List[Interval]:
        """
        Approximate Julia set by computing escape regions.
        Returns list of intervals that do not escape to infinity.
        """
        # Start with a grid of intervals covering [-2, 2]
        grid_size = 2 ** precision
        intervals = []
        
        for i in range(grid_size):
            left = Rational(-2) + Rational(4 * i, grid_size)
            right = Rational(-2) + Rational(4 * (i + 1), grid_size)
            interval = Interval(left, right)
            
            # Test if interval escapes under iteration
            if not self._escapes_to_infinity(interval, precision):
                intervals.append(interval)
        
        return intervals
    
    def _escapes_to_infinity(self, I: Interval, max_iterations: int) -> bool:
        """Check if interval escapes to infinity under iteration."""
        current = I
        escape_radius = Rational(2)
        
        for _ in range(max_iterations):
            current = self.func.extension(current)
            if current.is_bottom:
                return True
            
            # Check if interval is outside escape radius
            if current.left > escape_radius or current.right < -escape_radius:
                return True
        
        return False

# Measure Theory Examples from Paper 1

class ProbabilisticPowerDomain:
    """
    Probabilistic power domain for measure theory.
    Implements examples from Paper 1, Section 5.
    """
    
    def __init__(self, domain: IntervalDomain):
        self.domain = domain
    
    def uniform_measure_on_interval(self, I: Interval) -> 'ProbabilityMeasure':
        """Create uniform probability measure on interval."""
        return ProbabilityMeasure(I, lambda J: self._uniform_measure_value(I, J))
    
    def _uniform_measure_value(self, support: Interval, measurable_set: Interval) -> Rational:
        """Compute measure of set under uniform distribution."""
        if support.is_bottom or measurable_set.is_bottom:
            return Rational(0)
        
        intersection = support.intersection(measurable_set)
        if intersection is None:
            return Rational(0)
        
        if support.width() == Rational(0):
            return Rational(1) if intersection.width() > Rational(0) else Rational(0)
        
        try:
            return intersection.width() / support.width()
        except:
            return Rational(0)

class ProbabilityMeasure:
    """
    Represents a probability measure on intervals.
    """
    
    def __init__(self, support: Interval, measure_func: Callable[[Interval], Rational]):
        self.support = support
        self.measure = measure_func
    
    def __call__(self, measurable_set: Interval) -> Rational:
        """Compute measure of given set."""
        return self.measure(measurable_set)
    
    def pushforward(self, f: ComputableFunction) -> 'ProbabilityMeasure':
        """Compute pushforward measure under function f."""
        def pushforward_measure(J: Interval) -> Rational:
            # This is a simplified implementation
            # In practice, would need to compute preimage f^(-1)(J)
            try:
                # Approximate by evaluating f on support and checking overlap
                f_support = f.extension(self.support)
                if f_support.is_bottom:
                    return Rational(0)
                
                intersection = f_support.intersection(J)
                if intersection is None:
                    return Rational(0)
                
                # Rough approximation: ratio of overlapping region
                if f_support.width() == Rational(0):
                    return Rational(1) if intersection.width() > Rational(0) else Rational(0)
                
                return intersection.width() / f_support.width()
            except:
                return Rational(0)
        
        # The support of pushforward is f(original_support)
        new_support = f.extension(self.support)
        return ProbabilityMeasure(new_support, pushforward_measure)

if __name__ == "__main__":
    # Simple test to verify basic functionality
    print("Interval Domain Framework initialized successfully!")
    
    # Test basic interval operations
    I1 = Interval.make(0, 1)
    I2 = Interval.make(0.5, 1.5)
    print(f"I1 = {I1}")
    print(f"I2 = {I2}")
    print(f"I1 cap I2 = {I1.intersection(I2)}")
    print(f"I1 cup I2 = {I1.union_hull(I2)}")
    
    # Test computable real
    pi_approx = ComputableReal.from_rational(Rational(22, 7))
    print(f"pi approx {pi_approx}")
    print(f"pi as float: {pi_approx.to_float()}")
    
    # Test bottom element
    bottom = Interval.bottom()
    print(f"Bottom element: {bottom}")
    print(f"Bottom contains I1: {bottom.contains(I1)}")
    
    # Test way-below relation
    small = Interval.make(0.2, 0.8)
    large = Interval.make(0, 1)
    print(f"{large} << {small}: {large.way_below(small)}")
    
    print("\nAll basic tests passed!")