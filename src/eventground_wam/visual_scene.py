"""Explicit, lossless handling of LIBERO's state-dependent stove visualization."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import numpy as np


def static_scene_digest(xml, dynamic_sites):
    """Ignore only the alpha channel of explicitly qualified visualization sites."""
    root = ET.fromstring(xml)
    remaining = set(dynamic_sites)
    if len(remaining) != len(dynamic_sites):
        raise ValueError("Duplicate dynamic site")
    for site in root.iter("site"):
        name = site.get("name")
        if name not in dynamic_sites:
            continue
        if name not in remaining:
            raise ValueError("Ambiguous dynamic site name")
        rgba = site.get("rgba", "").split()
        if len(rgba) != 4 or not np.isfinite(np.asarray(rgba, dtype=float)).all():
            raise ValueError("Dynamic site must have explicit finite RGBA")
        site.set("rgba", " ".join(rgba[:3] + ["DYNAMIC_ALPHA"]))
        remaining.remove(name)
    if remaining:
        raise ValueError("Missing qualified dynamic site")
    return hashlib.sha256(ET.tostring(root, encoding="utf-8")).hexdigest()


def expected_site_rgba(spec, qpos):
    """Vendor FlatStove.turn_on: qpos >= .5; visibility toggles alpha to 1-alpha."""
    qpos = np.asarray(qpos)
    if qpos.shape != (spec["nq"],) or not np.isfinite(qpos).all():
        raise ValueError("Invalid scene qpos")
    rgba = np.asarray(spec["reset_site_rgba"], dtype=np.float32).copy()
    for field in spec["dynamic_sites"]:
        if field["predicate"] != "flat_stove_turn_on_qpos_ge_0.5":
            raise ValueError("Unqualified visualization predicate")
        i, address = field["site_id"], field["joint_qpos_address"]
        visible = qpos[address] >= 0.5
        if bool(rgba[i, 3] > 0) != bool(visible):
            rgba[i, 3] = 1 - rgba[i, 3]
    return rgba


def verify_visual_scene(xml, site_rgba, qpos, spec):
    names = [r["site_name"] for r in spec["dynamic_sites"]]
    if static_scene_digest(xml, names) != spec["static_scene_digest"]:
        raise ValueError("Unaccounted model/XML change")
    expected = expected_site_rgba(spec, qpos)
    if not np.array_equal(np.asarray(site_rgba), expected):
        raise ValueError("Visual state differs from independent joint predicate")
    return expected


def describe_visual_scene(env):
    """Extract supported visual rules from the actual pinned vendor environment."""
    model = env.sim.model
    dynamic = []
    # LIBERO appends the same object name again on hard reset. Its visualization
    # update is idempotent; deduplicate identical object references, not XML sites.
    for name in dict.fromkeys(env.env.visualization_sites_list):
        obj = env.env.get_object(name)
        if type(obj).__name__ != "FlatStove" or len(obj.joints) != 1:
            raise ValueError("Unqualified visual object")
        if obj.object_properties["articulation"]["default_turnon_ranges"] != [0.5, 2.1]:
            raise ValueError("Changed vendor visualization threshold")
        sites = obj.object_properties["vis_site_names"]
        if set(sites) != {"burner"}:
            raise ValueError("Unqualified visual sites")
        site = sites["burner"][0]
        joint = obj.joints[0]
        address = model.get_joint_qpos_addr(joint)
        if not isinstance(address, (int, np.integer)):
            raise ValueError("Expected scalar stove joint")
        dynamic.append(
            {
                "site_name": site,
                "site_id": int(model.site_name2id(site)),
                "joint_name": joint,
                "joint_qpos_address": int(address),
                "predicate": "flat_stove_turn_on_qpos_ge_0.5",
            }
        )
    xml = model.get_xml()
    return {
        "canonical_reset_sha256": hashlib.sha256(xml.encode()).hexdigest(),
        "static_scene_digest": static_scene_digest(xml, [r["site_name"] for r in dynamic]),
        "nq": int(model.nq),
        "reset_site_rgba": model.site_rgba.tolist(),
        "dynamic_sites": dynamic,
    }
