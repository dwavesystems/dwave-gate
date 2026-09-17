# %% [markdown]
# Copyright &copy; 2026 D-Wave
#
# The software is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This code example is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>

import warnings
from collections import Counter
from dataclasses import dataclass
from math import exp


@dataclass(frozen=True, slots=True)
class DecoderParams:
    """
    beta : float
        Controls how strongly distance is penalized.

    max_distance : int
        Maximum allowed distance from a clean bitstring.
        If no clean candidate is within this distance, discard the shot.

        max_distance=0 means only exact agreement on non-star bits is allowed.

    max_erasures : int or None
        Maximum allowed number of '*' characters.
        If None, do not discard based on number of stars.

    alpha : float
        Optional smoothing added to clean counts.

    min_posterior : float or None
        Minimum posterior probability a candidate must have to be considered.

        Example::
            {
                "000": 0.36,
                "001": 0.34,
                "010": 0.30,
            }

        So "000" is best but it's not a high posterior probability.

        If None, place no restriction.

    min_margin : float | None
        A required margin for the highest posterior and second highest to account
        for possible ambiguity.

        Example::
            {
                "000": 0.52,
                "001": 0.48,
            }

        so the best answer is "000" but barely.

        If None, place no restriction.
    """

    alpha: float
    beta: float
    max_distance: int
    max_erasures: int | None
    min_posterior: float | None
    min_margin: float | None


def distance_to_pattern(bitstring, pattern):
    """
    Hamming distance, ignoring '*' positions.
    """
    return sum(b != p for b, p in zip(bitstring, pattern) if p != "*")


def bayesian_decode_counts(counts, decoder_params: DecoderParams | None):
    """
    Decode counts containing '*' using clean counts as priors.

    Args:
        counts (dict[str, int]): Count dictionary.
            Example:
                {
                    "000": 10,
                    "001": 30,
                    "00*": 8,
                    "0**": 4,
                }
        decoder_params (DecoderParams | None): Parameters to go into the
            Bayesian decoding algorithm.

    Returns:
        decoded_counts : dict[str, int]
            Clean counts plus accepted starred counts assigned to guesses.

        assignments : dict[str, str]
            Mapping from starred patterns to chosen clean bitstrings.

        discarded_counts : dict[str, int]
            Starred patterns that were discarded.

        posteriors : dict[str, dict[str, float]]
            Posterior probabilities for accepted starred patterns.
    """

    if decoder_params is None:
        decoder_params = DecoderParams(
            alpha=0.0,
            beta=0.5,
            max_distance=1,
            max_erasures=1,
            min_posterior=0.1,
            min_margin=0.01,
        )

    clean_counts = {
        bitstring: count for bitstring, count in counts.items() if "*" not in bitstring
    }

    if not clean_counts:
        warnings.warn(
            "Need at least one bitstring with no '*'. Returning non-decoded results."
        )
        return counts, {}, {}, {}

    decoded_counts = Counter(clean_counts)
    assignments = {}
    discarded_counts = {}
    posteriors = {}

    for pattern, pattern_count in counts.items():
        if "*" not in pattern:
            continue

        if (
            decoder_params.max_erasures is not None
            and pattern.count("*") > decoder_params.max_erasures
        ):
            discarded_counts[pattern] = pattern_count
            continue

        candidates = []

        for bitstring in clean_counts:
            if len(bitstring) != len(pattern):
                continue

            distance = distance_to_pattern(
                bitstring,
                pattern,
            )

            if distance <= decoder_params.max_distance:
                candidates.append((bitstring, distance))

        if not candidates:
            discarded_counts[pattern] = pattern_count
            continue

        total_prior_mass = sum(
            clean_counts[bitstring] + decoder_params.alpha
            for bitstring, _ in candidates
        )

        scores = {}

        for bitstring, distance in candidates:
            prior = (clean_counts[bitstring] + decoder_params.alpha) / total_prior_mass
            likelihood = exp(-decoder_params.beta * distance)
            scores[bitstring] = prior * likelihood

        normalizer = sum(scores.values())

        posterior = {
            bitstring: score / normalizer for bitstring, score in scores.items()
        }

        posterior_sorted = sorted(
            posterior,
            key=posterior.get,
            reverse=True,
        )

        best_guess = posterior_sorted[0]
        best_prob = posterior[best_guess]

        if (
            decoder_params.min_posterior is not None
            and best_prob < decoder_params.min_posterior
        ):
            discarded_counts[pattern] = pattern_count
            continue

        if decoder_params.min_margin is not None and len(posterior_sorted) > 1:
            second_best = posterior_sorted[1]
            second_best_prob = posterior[second_best]

            if best_prob - second_best_prob < decoder_params.min_margin:
                discarded_counts[pattern] = pattern_count
                continue

        decoded_counts[best_guess] += pattern_count
        assignments[pattern] = best_guess
        posteriors[pattern] = posterior

    return dict(decoded_counts), assignments, discarded_counts, posteriors
