# Bayesian Decoder for Starred Bitstrings

Minimal Bayesian-style decoder for count dictionaries containing incomplete bitstrings with `"*"` characters.

The decoder uses complete bitstrings as candidate states and tries to infer which complete bitstring each starred bitstring most likely represents.

Example input:

```python
counts = {
    "000": 10,
    "001": 30,
    "010": 5,
    "00*": 8,
    "0*1": 6,
    "***": 12,
}
```

## Core idea

For each starred pattern, the decoder computes:

```text
posterior(candidate | pattern)
  proportional to
prior(candidate) * likelihood(pattern | candidate)
```

where:

```text
prior(candidate)
```

comes from the observed counts of complete bitstrings, and

```text
likelihood(pattern | candidate)
```

penalizes candidates that disagree with the visible bits in the starred pattern.

The default likelihood uses exponential distance decay:

```text
likelihood = exp(-beta * distance)
```

where `distance` is the Hamming distance between the candidate and the pattern, ignoring `"*"` positions.

## Main parameters

| Parameter | Meaning |
|---|---|
| `beta` | Controls how strongly distance is penalized |
| `max_distance` | Discard if no clean candidate is close enough |
| `max_erasures` | Discard if a pattern contains too many `"*"` characters |
| `min_posterior` | Discard if the best posterior probability is too low |
| `min_margin` | Discard if the best candidate does not beat the runner-up by enough |
| `alpha` | Optional smoothing added to clean counts |

## Example usage

```python
decoded, assignments, discarded, posteriors = bayesian_decode_counts(
    counts,
    beta=2.0,
    max_distance=0,
    max_stars=1,
    min_posterior=0.8,
    min_margin=0.2,
    alpha=0.0,
)
```

## Interpretation

With:

```python
max_distance = 0
```

a starred pattern must exactly match all visible bits.

For example:

```text
"00*"
```

may match:

```text
"000"
"001"
```

but not:

```text
"010"
```

because `"010"` disagrees at a visible position.

However, with
```python
max_distance = 1
```

a starred pattern must exactly match all visible bits in every position besides a single position.

For example:

```text
"00*"
```

may match:

```text
"000"
"001"
"010"
"011"
"100"
"101"
```

but not:

```text
"110"
"111"
```

because `"110"` and `"111"`disagrees at two visible positions.

With:

```python
min_posterior = 0.8
```

the decoder only accepts a guess if the best candidate has at least 80% posterior probability.

With:

```python
min_margin = 0.2
```

the decoder only accepts a guess if the best candidate beats the second-best candidate by at least 20 percentage points.

## Tuning `beta`

The `beta` parameter controls how sharply the likelihood penalizes distance.

Small `beta`:

```python
beta = 0.5
```

makes the decoder more forgiving.

Large `beta`:

```python
beta = 10.0
```

makes the decoder strongly prefer candidates that match the visible bits.

A simple way to tune `beta`:

1. Take clean bitstrings from the data.
2. Randomly replace some bits with `"*"`.
3. Run the decoder.
4. Check whether it recovers the original clean bitstrings.
5. Repeat for several `beta` values.
6. Choose the `beta` with the best recovery rate.

Example grid:

```python
beta_values = [0.5, 1.0, 2.0, 5.0, 10.0]
```

## Outputs

The decoder returns:

```python
decoded_counts, assignments, discarded, posteriors
```

where:

| Output | Meaning |
|---|---|
| `decoded_counts` | Clean counts plus accepted starred counts assigned to decoded bitstrings |
| `assignments` | Mapping from starred patterns to chosen clean bitstrings |
| `discarded` | Starred patterns that were rejected, grouped by reason |
| `posteriors` | Posterior distributions over candidates for accepted patterns |

## Notes

This implementation is intentionally minimal.

The Bayesian structure is:

```text
posterior ∝ prior × likelihood
```

The thresholds do not make the method Bayesian by themselves. They are practical safeguards that prevent low-quality or ambiguous starred shots from being forced into a decoded state.
