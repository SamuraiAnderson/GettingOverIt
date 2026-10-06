"""Persistent candidate quotas, independent of advantage and optimizer acceptance."""
from fractions import Fraction
import math


def pool_for_slot(cursor, recovery_fraction):
    if type(cursor) is not int or cursor < 0:
        raise ValueError("SIL quota cursor must be a nonnegative integer")
    if not math.isfinite(recovery_fraction) or not 0 < recovery_fraction < 1:
        raise ValueError("Recovery sampling share must lie strictly between zero and one")
    share = Fraction(recovery_fraction).limit_denominator(1000)
    if abs(float(share)-recovery_fraction) > 1e-12:
        raise ValueError("Recovery share needs a rational quota with denominator <= 1000")
    n, d = share.numerator, share.denominator
    cycle, position = divmod(cursor, d)
    # Rotate the balanced pattern each cycle; for 25%: REEE, EREE, EERE, EEER.
    position = (position-cycle) % d
    return "recovery" if (position*n) % d < n else "elite"
