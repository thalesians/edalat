# Interval Domain Framework

**Computable Real Analysis using Domain Theory**  
Based on the foundational work of Abbas Edalat.

---

## Overview

This Python framework implements domain-theoretic computable real analysis via interval domains. It formalizes core concepts from Edalat's papers on domain theory, enabling exact computation over real numbers, interval arithmetic, dynamical systems, and measure-theoretic structures.

Features include:

- Exact rational arithmetic with infinite values  
- Compact intervals ordered by reverse inclusion  
- Computable real numbers from effective sequences  
- Scott-continuous computable functions  
- Iterated function systems and attractors  
- Computable dynamical systems (e.g., Henon and Julia maps)  
- Probability measures over intervals  
- Full test coverage and validation suite  

---

## Key Concepts

### Rational Numbers

```python
r = Rational(3, 2)  # Represents 3/2 exactly
x = Rational.from_float(1.414213)
```

Handles infinities, exact arithmetic, and float conversion.

---

### Intervals

```python
I = Interval.make(1, 2)
J = Interval.point(Rational(3, 2))
K = I.intersection(J)
```

- Intervals use reverse inclusion order for domain-theoretic reasoning.
- `Interval.bottom()` represents the bottom element.

---

### Effective Sequences

```python
def decreasing_seq(n): 
    return Interval(Rational(1, 2) - Rational(1, 2**(n+1)), Rational(1, 2) + Rational(1, 2**(n+1)))

chain = EffectiveSequence(decreasing_seq)
```

Represents computable chains of intervals converging to a real number.

---

### Computable Reals

```python
x = ComputableReal.from_rational(Rational(22, 7))
y = ComputableReal.from_cauchy_sequence(lambda n: Rational(355, 113))
z = x + y
float_approx = z.to_float(precision=10)
```

Constructed as least upper bounds (lub) of effective interval chains.

---

### Computable Functions

```python
f = ComputableFunction.from_real_function(lambda x: x ** 2 + 1)
result = f(x)  # Applies to ComputableReal
```

Evaluated by extending the real function to intervals with outward rounding.

---

### Decimal and Binary Representations

```python
dec = DecimalRepresentation()
pi = dec.from_decimal_digits([3, 1, 4, 1, 5, 9])  # Interprets as 0.314159...

bin = BinaryRepresentation()
val = bin.from_signed_binary([1, -1, 1, -1])
```

Implements constructive number representations via IFS contraction mappings.

---

### Dynamical Systems

```python
quad_map = QuadraticMap(c=-0.75)
julia = quad_map.julia_set_approximation(precision=4)

ifs = IteratedFunctionSystem([f], Interval.make(0, 1))
point = ifs.compute_attractor_point([0, 1, 0, 1], precision=10)
```

Computes fixed points, attractors, and Julia sets using interval dynamics.

---

### Probability Measures

```python
ppd = ProbabilisticPowerDomain(domain)
uniform = ppd.uniform_measure_on_interval(Interval.make(0, 1))
measure = uniform(Interval.make(0, 0.5))  # Returns Rational(1, 2)
```

Supports uniform measures and pushforwards in the interval domain.

---

## Test Suite

Run the full suite with:

```bash
python test_interval_domain.py
```

Test coverage includes:

- Interval operations and domain order  
- Computable real convergence and arithmetic  
- Scott-continuous function application  
- Julia set approximations  
- Henon map transformations  
- Measure-theoretic calculations  
- Representation tests (decimal, binary)  
- Theoretical properties (Scott topology, effective domains)  

---

## References

- Edalat, A. (1995). *Dynamical Systems, Measures, and Fractals via Domain Theory*.  
- Edalat, A. (1997). *Domains for Computable Analysis*.  
- Pour-El & Richards. *Computability in Analysis and Physics*.  

---

## Applications

- Exact real arithmetic and analysis  
- Formal methods and proof systems  
- Dynamical systems and attractor computation  
- Constructive measure theory  
- Visualization of computable fractals  
- Educational demonstrations of domain theory  

---

## File Structure

```
interval_domain_core.py     # Core library (intervals, reals, functions, IFS, measures)
test_interval_domain.py     # Comprehensive validation suite
```

---

## License

Apache 2.0 License
