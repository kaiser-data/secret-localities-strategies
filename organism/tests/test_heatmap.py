"""Tests for heatmap.py - the layer x module implant map.

The grid logic is pure stdlib and runs on the recorded weightdiff JSON, so these tests
are real-data tests wherever a result file exists rather than fixtures only.
"""
from __future__ import annotations

import json
import math
import os

import pytest

import heatmap

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")


# --- name parsing ------------------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("model.layers.14.self_attn.q_proj.weight", (14, "q_proj")),
    ("model.layers.0.self_attn.o_proj.weight", (0, "o_proj")),
    ("model.layers.27.mlp.down_proj.weight", (27, "down_proj")),
    ("model.layers.3.input_layernorm.weight", (3, "input_layernorm")),
    ("model.layers.3.post_attention_layernorm.weight", (3, "post_attention_layernorm")),
    ("model.embed_tokens.weight", (None, "embed_tokens")),
    ("lm_head.weight", (None, "lm_head")),
    ("model.norm.weight", (None, "norm")),
])
def test_parse_tensor_name(name, expected):
    assert heatmap.parse_tensor_name(name) == expected


def test_parse_tensor_name_tolerates_bias_and_unsuffixed():
    assert heatmap.parse_tensor_name("model.layers.5.self_attn.q_proj.bias") == (5, "q_proj")
    assert heatmap.parse_tensor_name("model.layers.5.self_attn.q_proj") == (5, "q_proj")


# --- grid construction -------------------------------------------------------------

def _norms(*triples):
    """(name, rel_fro) pairs in the shape weight_diff.py writes."""
    return [{"name": n, "rel_fro": v, "ndim": 2, "shape": [8, 8]} for n, v in triples]


def test_grid_layers_sort_numerically_not_lexically():
    m = heatmap.build_map(_norms(
        ("model.layers.2.self_attn.q_proj.weight", 0.1),
        ("model.layers.10.self_attn.q_proj.weight", 0.2),
        ("model.layers.1.self_attn.q_proj.weight", 0.3),
    ), name="t", base="b")
    assert m.layers == (1, 2, 10)


def test_grid_module_order_follows_forward_pass():
    m = heatmap.build_map(_norms(
        ("model.layers.0.mlp.down_proj.weight", 0.1),
        ("model.layers.0.self_attn.q_proj.weight", 0.2),
        ("model.layers.0.self_attn.v_proj.weight", 0.3),
    ), name="t", base="b")
    # q before v before down_proj: the canonical order, not alphabetical.
    assert m.modules == ("q_proj", "v_proj", "down_proj")


def test_grid_unknown_module_is_kept_and_sorted_last():
    m = heatmap.build_map(_norms(
        ("model.layers.0.self_attn.q_proj.weight", 0.2),
        ("model.layers.0.zzz_custom.weight", 0.1),
    ), name="t", base="b")
    assert m.modules == ("q_proj", "zzz_custom")


def test_grid_separates_non_layer_tensors_into_globals():
    """An untouched tensor is absent, not zero: the axes are built from what CHANGED, so
    `m.modules` reads out F6's profile directly instead of the model's architecture."""
    m = heatmap.build_map(_norms(
        ("model.layers.0.self_attn.q_proj.weight", 0.2),
        ("model.embed_tokens.weight", 0.0),
        ("lm_head.weight", 0.5),
    ), name="t", base="b")
    assert m.layers == (0,)
    assert m.modules == ("q_proj",)
    assert m.globals == {"lm_head": 0.5}


def test_grid_skips_null_metric_rows():
    """weight_diff.py writes rel_fro=None on a shape mismatch. A None is not a zero."""
    rows = _norms(("model.layers.0.self_attn.q_proj.weight", 0.2))
    rows.append({"name": "model.layers.0.self_attn.k_proj.weight", "rel_fro": None,
                 "note": "shape mismatch", "ndim": 2})
    m = heatmap.build_map(rows, name="t", base="b")
    assert m.cells.get((0, "q_proj")) == 0.2
    assert (0, "k_proj") not in m.cells


def test_grid_cells_are_not_mutable():
    m = heatmap.build_map(_norms(("model.layers.0.self_attn.q_proj.weight", 0.2)),
                          name="t", base="b")
    with pytest.raises(TypeError):
        m.cells[(0, "q_proj")] = 9.9


# --- the differential map ----------------------------------------------------------

def test_differential_subtracts_cellwise():
    a = heatmap.build_map(_norms(
        ("model.layers.0.self_attn.q_proj.weight", 0.5),
        ("model.layers.1.self_attn.q_proj.weight", 0.2),
    ), name="org", base="B")
    b = heatmap.build_map(_norms(
        ("model.layers.0.self_attn.q_proj.weight", 0.3),
        ("model.layers.1.self_attn.q_proj.weight", 0.2),
    ), name="ctl", base="B")
    d = heatmap.differential(a, b)
    assert d.cells[(0, "q_proj")] == pytest.approx(0.2)
    assert d.cells[(1, "q_proj")] == pytest.approx(0.0)


def test_differential_drops_cells_present_on_only_one_side():
    """A cell only one model touched has no difference - reporting it as its own value
    would attribute the control's footprint to the organism."""
    a = heatmap.build_map(_norms(("model.layers.0.self_attn.q_proj.weight", 0.5)),
                          name="org", base="B")
    b = heatmap.build_map(_norms(("model.layers.0.self_attn.v_proj.weight", 0.3)),
                          name="ctl", base="B")
    d = heatmap.differential(a, b)
    assert d.cells == {}
    assert d.modules == ("q_proj", "v_proj")  # the axis still shows both


def test_differential_refuses_a_different_base():
    """Two dW maps are only comparable against the same W_base."""
    a = heatmap.build_map(_norms(("model.layers.0.self_attn.q_proj.weight", 0.5)),
                          name="org", base="Qwen/Qwen2.5-7B-Instruct")
    b = heatmap.build_map(_norms(("model.layers.0.self_attn.q_proj.weight", 0.3)),
                          name="ctl", base="Qwen/Qwen2.5-1.5B-Instruct")
    with pytest.raises(ValueError, match="different base"):
        heatmap.differential(a, b)


# --- reading against the map's own null ---------------------------------------------

def test_robust_z_is_zero_on_a_flat_map():
    assert heatmap.robust_z(0.2, [0.2] * 8) == 0.0


def test_robust_z_flags_a_single_hot_cell():
    population = [0.01] * 20 + [0.9]
    assert heatmap.robust_z(0.9, population) > 4.0


def test_hotspots_ranks_by_magnitude_and_respects_k():
    m = heatmap.build_map(_norms(
        ("model.layers.0.self_attn.q_proj.weight", 0.1),
        ("model.layers.1.self_attn.q_proj.weight", 0.9),
        ("model.layers.2.self_attn.q_proj.weight", 0.4),
    ), name="t", base="b")
    top = heatmap.hotspots(m, k=2)
    assert [h["layer"] for h in top] == [1, 2]
    assert top[0]["z"] >= top[1]["z"]


def test_hotspots_ranks_a_differential_by_absolute_value():
    """On a differential, a cell the control moved MORE than the organism is as
    informative as one it moved less - sign is reported, magnitude ranks."""
    a = heatmap.build_map(_norms(
        ("model.layers.0.self_attn.q_proj.weight", 0.1),
        ("model.layers.1.self_attn.q_proj.weight", 0.9),
    ), name="org", base="b")
    b = heatmap.build_map(_norms(
        ("model.layers.0.self_attn.q_proj.weight", 0.9),
        ("model.layers.1.self_attn.q_proj.weight", 0.9),
    ), name="ctl", base="b")
    top = heatmap.hotspots(heatmap.differential(a, b), k=1)
    assert top[0]["layer"] == 0
    assert top[0]["value"] < 0


# --- the ASCII renderer, so a null run is readable without matplotlib ----------------

def test_ascii_shows_axes_and_marks_empty_cells():
    m = heatmap.build_map(_norms(
        ("model.layers.0.self_attn.q_proj.weight", 0.5),
        ("model.layers.1.mlp.down_proj.weight", 0.3),
    ), name="t", base="b")
    art = heatmap.ascii_map(m)
    assert "q_proj" in art and "down_proj" in art
    # The grid is 2x2 but only 2 cells exist; the other two are untouched.
    assert "." in art


def test_ascii_says_so_on_an_empty_map():
    m = heatmap.build_map([], name="identical", base="b")
    assert "no changed tensors" in heatmap.ascii_map(m).lower()


# --- real recorded data -------------------------------------------------------------

@pytest.mark.parametrize("stem,expect_attention_only", [
    ("sl-organism-a-7b", True),
    ("sl-organism-b-7b", True),
    ("poison-sweep-12.5pct", False),
])
def test_real_weightdiff_reproduces_f6_profile(stem, expect_attention_only):
    """F6: A and B are attention-only; the poison ladder also moved the MLP.

    This is the finding the heatmap is supposed to make visible at a glance, so the
    grid must carry it.
    """
    path = os.path.join(RESULTS, f"weightdiff_{stem}.json")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not recorded in this checkout")
    m = heatmap.load_map(path)
    mlp = {"gate_proj", "up_proj", "down_proj"} & set(m.modules)
    assert bool(mlp) is not expect_attention_only
    assert m.globals == {}, "F6 P1.3: embed_tokens / lm_head must be untouched"


def test_real_identical_model_gives_an_empty_map():
    path = os.path.join(RESULTS, "weightdiff_sl-organism-c-7b.json")
    if not os.path.isfile(path):
        pytest.skip("organism C weightdiff not recorded in this checkout")
    m = heatmap.load_map(path)
    assert m.cells == {} and m.globals == {}
    assert "no changed tensors" in heatmap.ascii_map(m).lower()


def test_real_rank_metric_is_readable_as_a_grid():
    path = os.path.join(RESULTS, "weightdiff_sl-organism-a-7b.json")
    if not os.path.isfile(path):
        pytest.skip("organism A weightdiff not recorded in this checkout")
    m = heatmap.load_map(path, metric="rank99")
    # Only the SVD'd tensors carry rank99, so this grid is sparser than the norm grid.
    assert m.cells, "no rank99 cells - check weight_diff.py --svd-top"
    assert all(float(v).is_integer() for v in m.cells.values())
    assert max(m.cells.values()) <= 16, "F6 P1.2: merged rank-16 adapter"


def test_real_norm_grid_covers_attention_only():
    path = os.path.join(RESULTS, "weightdiff_sl-organism-a-7b.json")
    if not os.path.isfile(path):
        pytest.skip("organism A weightdiff not recorded in this checkout")
    m = heatmap.load_map(path)
    assert set(m.modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert all(math.isfinite(v) and v > 0 for v in m.cells.values())
    assert max(m.layers) <= 27, "Qwen2.5-7B has 28 layers, indexed 0-27"


def test_real_recorded_runs_are_flagged_sparse():
    """The seven recorded runs predate all_norms. If a map built from the 40-row
    svd_tensors slice claimed to be dense, its blanks would read as 'untouched' and the
    map would silently assert F6's attention-only finding instead of showing it."""
    path = os.path.join(RESULTS, "weightdiff_sl-organism-a-7b.json")
    if not os.path.isfile(path):
        pytest.skip("organism A weightdiff not recorded in this checkout")
    record = json.load(open(path))
    m = heatmap.load_map(path)
    if "all_norms" in record:
        assert m.dense and m.source == "all_norms"
    else:
        assert not m.dense and m.source == "svd_tensors"
        assert "NOT MEASURED" in m.blank_means
        assert "NOT MEASURED" in heatmap.ascii_map(m)


def test_real_differential_between_two_organisms_is_signed():
    a_path = os.path.join(RESULTS, "weightdiff_sl-organism-a-7b.json")
    b_path = os.path.join(RESULTS, "weightdiff_sl-organism-b-7b.json")
    if not (os.path.isfile(a_path) and os.path.isfile(b_path)):
        pytest.skip("A/B weightdiffs not both recorded")
    d = heatmap.differential(heatmap.load_map(a_path), heatmap.load_map(b_path))
    assert d.cells, "A and B overlap on the tensors both runs SVD'd"
    values = list(d.cells.values())
    assert min(values) < 0 < max(values), (
        "A and B are different training runs; a strictly one-signed differential would "
        "mean one dominates everywhere, which would itself be the finding"
    )


def test_real_differential_carries_the_sparse_flag_forward():
    a_path = os.path.join(RESULTS, "weightdiff_sl-organism-a-7b.json")
    b_path = os.path.join(RESULTS, "weightdiff_sl-organism-b-7b.json")
    if not (os.path.isfile(a_path) and os.path.isfile(b_path)):
        pytest.skip("A/B weightdiffs not both recorded")
    a, b = heatmap.load_map(a_path), heatmap.load_map(b_path)
    d = heatmap.differential(a, b)
    assert d.dense is (a.dense and b.dense)
    assert d.signed is True


def test_load_map_rejects_a_file_that_is_not_a_weightdiff(tmp_path):
    p = tmp_path / "notaweightdiff.json"
    p.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError, match="all_norms"):
        heatmap.load_map(str(p))


def test_load_map_rejects_an_unknown_metric(tmp_path):
    p = tmp_path / "weightdiff_x.json"
    p.write_text(json.dumps({"base": "b", "target": "t", "all_norms": []}))
    with pytest.raises(ValueError, match="unknown metric"):
        heatmap.load_map(str(p), metric="not_a_metric")
