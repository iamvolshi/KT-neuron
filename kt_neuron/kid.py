"""KID generation for KT neurons.

A KID is the neuron's birth identity: a thing without a KID is not a
thing, and no read or write happens before one exists. The scheme is the
toy's, kept verbatim so existing records stay readable: 192 bits packed
from (birth_tick, unix seconds at birth, sequence number), rendered as
32 characters of base64url.

Uniqueness note: the caller must supply a sequence number unique within
the birth second. KTNeuron does this with a process-wide counter. Two
neurons born in different processes in the same second with the same
sequence number could collide; if that ever matters, the fix is to widen
the packed material, not to reuse a KID.
"""

import base64
import struct
import time


def make_kid(birth_tick: int, seq: int) -> str:
    """192-bit KID: (birth_tick, born_at_unix, seq) -> 32-char base64url."""
    raw = struct.pack(">QQQ", birth_tick, int(time.time()), seq)
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")
