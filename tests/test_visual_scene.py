import copy

import numpy as np
import pytest

from eventground_wam.visual_scene import (
    describe_visual_scene,
    expected_site_rgba,
    static_scene_digest,
    verify_visual_scene,
)

XML = '<mujoco><worldbody><geom name="g" size="1"/><site name="burner" rgba="0.9 0.05 0.05 -1"/><site name="other" rgba="1 1 1 0"/></worldbody></mujoco>'


def spec():
    return {
        "nq": 1,
        "reset_site_rgba": [[0.9, 0.05, 0.05, -1], [1, 1, 1, 0]],
        "static_scene_digest": static_scene_digest(XML, ["burner"]),
        "dynamic_sites": [
            {
                "site_name": "burner",
                "site_id": 0,
                "joint_qpos_address": 0,
                "predicate": "flat_stove_turn_on_qpos_ge_0.5",
            }
        ],
    }


@pytest.mark.parametrize("q,alpha", [(0, -1), (0.499999, -1), (0.5, 2), (2.1, 2)])
def test_vendor_boundary_and_raw_alpha_values(q, alpha):
    s = spec()
    rgba = expected_site_rgba(s, [q])
    assert rgba[0, 3] == alpha
    assert np.array_equal(
        verify_visual_scene(XML.replace("0.05 -1", f"0.05 {alpha}"), rgba, [q], s), rgba
    )
    assert s == spec()  # Neither the frozen reset model nor its metadata is mutated.


@pytest.mark.parametrize(
    "damage", ["geometry", "rgb", "other_alpha", "wrong_burner", "nonfinite", "predicate"]
)
def test_unaccounted_changes_fail_closed(damage):
    s = copy.deepcopy(spec())
    rgba = expected_site_rgba(s, [0])
    xml = XML
    if damage == "geometry":
        xml = xml.replace('size="1"', 'size="2"')
    if damage == "rgb":
        xml = xml.replace("0.9 0.05", "0.8 0.05")
    if damage == "other_alpha":
        xml = xml.replace("1 1 1 0", "1 1 1 1")
    if damage == "wrong_burner":
        rgba[0, 3] = 2
    if damage == "nonfinite":
        rgba[1, 0] = np.nan
    if damage == "predicate":
        s["dynamic_sites"][0]["predicate"] = "different"
    with pytest.raises(ValueError):
        verify_visual_scene(xml, rgba, [0], s)


@pytest.mark.parametrize("names", [["missing"], ["burner", "burner"]])
def test_dynamic_site_allowlist_is_exact(names):
    with pytest.raises(ValueError):
        static_scene_digest(XML, names)


def test_hard_reset_duplicate_object_registration_keeps_one_exact_site():
    from types import SimpleNamespace

    obj = type("FlatStove", (), {})()
    obj.joints = ["knob"]
    obj.object_properties = {
        "articulation": {"default_turnon_ranges": [0.5, 2.1]},
        "vis_site_names": {"burner": ("burner", False)},
    }
    model = SimpleNamespace(
        get_xml=lambda: XML,
        get_joint_qpos_addr=lambda _: 0,
        site_name2id=lambda _: 0,
        nq=1,
        site_rgba=np.array(spec()["reset_site_rgba"], dtype=np.float32),
    )
    env = SimpleNamespace(
        sim=SimpleNamespace(model=model),
        env=SimpleNamespace(visualization_sites_list=["stove", "stove"], get_object=lambda _: obj),
    )
    described = describe_visual_scene(env)
    assert len(described["dynamic_sites"]) == 1
    assert described["dynamic_sites"][0]["joint_name"] == "knob"
