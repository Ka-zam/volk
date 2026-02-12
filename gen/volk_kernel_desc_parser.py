#!/usr/bin/env python
# Copyright 2026 Magnus Lundmark <magnuslundmark@gmail.com>
#
# This file is part of VOLK
#
# SPDX-License-Identifier: LGPL-3.0-or-later
#

"""
Parser for volk_kernel_desc_t initializers embedded in kernel headers.

Extracts the C struct initializer from the #ifdef VOLK_KERNEL_DESC block
and converts it into a Python dataclass for use by the code generator
and test harness.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Optional


# Mirror the C enums as Python constants
DOMAIN_TYPES = {
    "VOLK_DOMAIN_ANY": 0,
    "VOLK_DOMAIN_SEGMENT": 1,
    "VOLK_DOMAIN_DISK": 2,
    "VOLK_DOMAIN_ANNULUS": 3,
    "VOLK_DOMAIN_HALF_PLANE": 4,
}

OP_TYPES = {
    "VOLK_OP_MAP": 0,
    "VOLK_OP_REDUCE": 1,
}

# C constants we may encounter in initializers
C_CONSTANTS = {
    "true": True,
    "false": False,
    "NULL": None,
    "VOLK_NO_STATE": -1,
    "INFINITY": math.inf,
    "-INFINITY": -math.inf,
    "NAN": math.nan,
    "M_PI": math.pi,
}
C_CONSTANTS.update(DOMAIN_TYPES)
C_CONSTANTS.update(OP_TYPES)


@dataclass
class DomainDesc:
    """Mirrors volk_domain_t."""
    type: int = 0           # volk_domain_type_t value
    bounds: tuple = (0.0, 0.0)

    @property
    def type_name(self) -> str:
        for name, val in DOMAIN_TYPES.items():
            if val == self.type:
                return name
        return f"UNKNOWN({self.type})"

    @property
    def is_any(self) -> bool:
        return self.type == DOMAIN_TYPES["VOLK_DOMAIN_ANY"]


@dataclass
class KernelDesc:
    """Mirrors volk_kernel_desc_t."""
    op_type: int = 0                        # volk_op_t value
    granularity: int = 0
    stride: int = 0
    in_place: bool = False
    domains: list = field(default_factory=lambda: [DomainDesc()] * 10)
    state_arg: int = -1
    tolerance: float = 0.0
    absolute_tolerance: bool = False
    deprecated_by: Optional[str] = None
    extra: Optional[str] = None

    @property
    def is_stateful(self) -> bool:
        return self.state_arg >= 0

    @property
    def op_type_name(self) -> str:
        for name, val in OP_TYPES.items():
            if val == self.op_type:
                return name
        return f"UNKNOWN({self.op_type})"

    def to_c_initializer(self) -> str:
        """Emit a C designated initializer body (without the variable declaration)."""
        def _c_bool(v):
            return "true" if v else "false"

        def _c_float(v):
            if math.isinf(v):
                return "-INFINITY" if v < 0 else "INFINITY"
            if math.isnan(v):
                return "NAN"
            return f"{v}f"

        def _c_string(v):
            return f'"{v}"' if v is not None else "NULL"

        lines = []
        lines.append(f"    .op_type = {self.op_type_name},")
        if self.granularity:
            lines.append(f"    .granularity = {self.granularity},")
        if self.stride:
            lines.append(f"    .stride = {self.stride},")
        lines.append(f"    .in_place = {_c_bool(self.in_place)},")

        # Only emit non-ANY domains
        non_any = [(i, d) for i, d in enumerate(self.domains) if not d.is_any]
        if non_any:
            lines.append("    .domains = {")
            for i, d in non_any:
                lines.append(f"        [{i}] = {{ {d.type_name}, "
                             f"{{ {_c_float(d.bounds[0])}, {_c_float(d.bounds[1])} }} }},")
            lines.append("    },")

        lines.append(f"    .state_arg = {self.state_arg},")
        if self.tolerance:
            lines.append(f"    .tolerance = {_c_float(self.tolerance)},")
        if self.absolute_tolerance:
            lines.append(f"    .absolute_tolerance = {_c_bool(self.absolute_tolerance)},")
        if self.deprecated_by is not None:
            lines.append(f"    .deprecated_by = {_c_string(self.deprecated_by)},")
        if self.extra is not None:
            lines.append(f"    .extra = {_c_string(self.extra)},")

        return "{\n" + "\n".join(lines) + "\n}"


def _extract_desc_block(code: str) -> Optional[str]:
    """Extract the body of the #ifdef VOLK_KERNEL_DESC ... #endif block."""
    pattern = re.compile(
        r'#\s*ifdef\s+VOLK_KERNEL_DESC\s*\n(.*?)#\s*endif',
        re.DOTALL
    )
    m = pattern.search(code)
    if m:
        return m.group(1)
    return None


def _eval_c_float(token: str) -> float:
    """Evaluate a C float literal or constant to a Python float."""
    token = token.strip().rstrip("f").rstrip("F")

    # Handle cast expressions like (float)M_PI
    cast = re.match(r'^\(float\)\s*(.+)$', token)
    if cast:
        token = cast.group(1).strip()

    # Handle negation of a constant
    if token.startswith("-"):
        inner = token[1:].strip()
        if inner in C_CONSTANTS:
            return -float(C_CONSTANTS[inner])
        return -float(inner)

    if token in C_CONSTANTS:
        val = C_CONSTANTS[token]
        return float(val) if not isinstance(val, bool) else float(val)

    return float(token)


def _eval_c_int(token: str) -> int:
    """Evaluate a C integer literal or constant to a Python int."""
    token = token.strip()
    if token in C_CONSTANTS:
        val = C_CONSTANTS[token]
        return int(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else val
    return int(token, 0)  # base 0 handles hex/octal


def _eval_c_bool(token: str) -> bool:
    """Evaluate a C boolean to Python bool."""
    token = token.strip()
    if token in C_CONSTANTS:
        return bool(C_CONSTANTS[token])
    return bool(int(token))


def _eval_c_string(token: str) -> Optional[str]:
    """Evaluate a C string literal to Python string, or NULL to None."""
    token = token.strip()
    if token == "NULL":
        return None
    # Strip surrounding quotes
    m = re.match(r'^"(.*)"$', token, re.DOTALL)
    if m:
        return m.group(1)
    return token


def _parse_domain(text: str) -> DomainDesc:
    """Parse a single domain initializer like '{ VOLK_DOMAIN_SEGMENT, { -1.0f, 1.0f } }'."""
    text = text.strip().strip("{}")
    # Split at the first comma that isn't inside braces
    depth = 0
    split_pos = None
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            split_pos = i
            break

    if split_pos is None:
        # Only type specified, no bounds
        type_str = text.strip()
        if type_str in C_CONSTANTS:
            return DomainDesc(type=C_CONSTANTS[type_str])
        return DomainDesc()

    type_str = text[:split_pos].strip()
    bounds_str = text[split_pos + 1:].strip().strip("{}")

    domain_type = C_CONSTANTS.get(type_str, 0)

    bounds_parts = bounds_str.split(",")
    b0 = _eval_c_float(bounds_parts[0]) if len(bounds_parts) > 0 else 0.0
    b1 = _eval_c_float(bounds_parts[1]) if len(bounds_parts) > 1 else 0.0

    return DomainDesc(type=domain_type, bounds=(b0, b1))


def _parse_domains_array(text: str) -> list:
    """Parse the domains array initializer."""
    domains = [DomainDesc() for _ in range(10)]

    # Match [index] = { ... } entries
    pattern = re.compile(
        r'\[(\d+)\]\s*=\s*(\{[^}]*(?:\{[^}]*\}[^}]*)?\})',
        re.DOTALL
    )
    for m in pattern.finditer(text):
        idx = int(m.group(1))
        if 0 <= idx < 10:
            domains[idx] = _parse_domain(m.group(2))

    return domains


def _extract_initializer_fields(init_text: str) -> dict:
    """
    Parse a C designated initializer into a dict of field name -> raw value string.

    Handles nested braces for .domains = { ... }.
    """
    fields = {}
    # Remove the outer struct braces
    init_text = init_text.strip().strip(";")
    m = re.search(r'\{(.+)\}', init_text, re.DOTALL)
    if not m:
        return fields
    body = m.group(1)

    # Walk through and split on top-level '.field =' boundaries
    current_field = None
    current_value = ""
    depth = 0

    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("#"):
            continue

        # Check for a new field assignment at brace depth 0
        field_match = re.match(r'^\s*\.(\w+)\s*=\s*(.*)$', stripped)
        if field_match and depth == 0:
            # Save previous field
            if current_field is not None:
                fields[current_field] = current_value.strip().rstrip(",")
            current_field = field_match.group(1)
            current_value = field_match.group(2)
        else:
            current_value += "\n" + stripped

        # Track brace depth within current value
        depth += stripped.count("{") - stripped.count("}")

    # Save last field
    if current_field is not None:
        fields[current_field] = current_value.strip().rstrip(",")

    return fields


def parse_kernel_desc(code: str) -> Optional[KernelDesc]:
    """
    Parse a volk_kernel_desc_t initializer from kernel header source code.

    Args:
        code: Full source code of the kernel header file.

    Returns:
        KernelDesc if a descriptor was found, None otherwise.
    """
    block = _extract_desc_block(code)
    if block is None:
        return None

    # Find the struct initializer
    init_match = re.search(
        r'static\s+const\s+volk_kernel_desc_t\s+\w+\s*=\s*(\{.+?\});',
        block,
        re.DOTALL
    )
    if not init_match:
        return None

    fields = _extract_initializer_fields(init_match.group(1))

    desc = KernelDesc()

    if "op_type" in fields:
        val = fields["op_type"].strip()
        desc.op_type = C_CONSTANTS.get(val, 0)

    if "granularity" in fields:
        desc.granularity = _eval_c_int(fields["granularity"])

    if "stride" in fields:
        desc.stride = _eval_c_int(fields["stride"])

    if "in_place" in fields:
        desc.in_place = _eval_c_bool(fields["in_place"])

    if "domains" in fields:
        desc.domains = _parse_domains_array(fields["domains"])

    if "state_arg" in fields:
        desc.state_arg = _eval_c_int(fields["state_arg"])

    if "tolerance" in fields:
        desc.tolerance = _eval_c_float(fields["tolerance"])

    if "absolute_tolerance" in fields:
        desc.absolute_tolerance = _eval_c_bool(fields["absolute_tolerance"])

    if "deprecated_by" in fields:
        desc.deprecated_by = _eval_c_string(fields["deprecated_by"])

    if "extra" in fields:
        desc.extra = _eval_c_string(fields["extra"])

    return desc


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_code = '''
#ifdef VOLK_KERNEL_DESC
#include <volk/volk_kernel_desc.h>
static const volk_kernel_desc_t volk_32fc_s32fc_x2_rotator2_32fc_desc = {
    .op_type     = VOLK_OP_MAP,
    .granularity = 512,
    .in_place    = false,
    .domains     = {
        [0] = { VOLK_DOMAIN_ANY },
        [1] = { VOLK_DOMAIN_ANY },
        [2] = { VOLK_DOMAIN_ANNULUS, { 1.0f, 1.0f } },
        [3] = { VOLK_DOMAIN_ANNULUS, { 1.0f, 1.0f } },
    },
    .state_arg   = 3,
    .tolerance   = 1e-3f,
    .absolute_tolerance = true,
};
#endif
'''
    desc = parse_kernel_desc(test_code)
    if desc:
        print(f"op_type:      {desc.op_type_name}")
        print(f"granularity:  {desc.granularity}")
        print(f"in_place:     {desc.in_place}")
        print(f"state_arg:    {desc.state_arg}")
        print(f"tolerance:    {desc.tolerance}")
        print(f"absolute_tol: {desc.absolute_tolerance}")
        print(f"domains:")
        for i, d in enumerate(desc.domains):
            if not d.is_any:
                print(f"  [{i}] {d.type_name}  bounds={d.bounds}")
    else:
        print("No descriptor found")
