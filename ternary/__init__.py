"""Ternary operator experiment for eml-sr issue #37.

Explores the ternary Sheffer candidate from §5 of the paper:

    T(x, y, z) = exp(x) / ln(x) * ln(z) / exp(y)
               = exp(x-y) * ln(z) / ln(x)

which satisfies T(x, x, x) = 1 for x not in {0, 1}.

Status: research scaffold, not a production engine. See ``report.md``
for the verdict.
"""
