import json
import math

import pytest
import dense_heatmap as dh


def synth(n_layers=28, changed=True, with_svd=True, drop=()):
    """A weight_diff.py-shaped record. `drop` omits (layer, module) pairs entirely."""
    norms, svd = [], []
    for layer in range(n_layers):
        for mod in dh.MODULES:
            if (layer, mod) in drop:
                continue
            name = f"model.layers.{layer}.self_attn.{mod}.weight"
            rel = (0.001 + 0.0001 * layer) if changed else 0.0
            norms.append({"name": name, "rel_fro": rel, "abs_fro": rel * 10,
                          "shape": [3584, 3584], "ndim": 2})
            if changed and with_svd:
                svd.append({"name": name, "rel_fro": rel, "rank99": 16,
                            "top16_energy": 0.99, "looks_like_lora": True})
    return {"target": "t", "base": "b", "n_changed": len(svd), "identical": not changed,
            "all_norms": norms, "svd_tensors": svd}


def test_grid_is_twenty_eight_by_four():
    g = dh.build_grid(synth())
    assert dh.N_LAYERS == 28
    assert len(dh.MODULES) == 4
    assert sum(1 for layer in g["cells"] for _ in layer) == 112


def test_every_attention_cell_is_measured_for_a_dense_extraction():
    g = dh.build_grid(synth())
    states = {c["state"] for row in g["cells"] for c in row}
    assert states == {"measured"}


def test_a_missing_tensor_is_not_measured_and_never_zero():
    g = dh.build_grid(synth(drop=((3, "k_proj"),)))
    cell = g["cells"][3][dh.MODULES.index("k_proj")]
    assert cell["state"] == "not_measured"
    assert cell["rel_fro"] is None


def test_a_measured_zero_is_distinct_from_not_measured():
    g = dh.build_grid(synth(changed=False))
    cell = g["cells"][0][0]
    assert cell["state"] == "measured"
    assert cell["rel_fro"] == 0.0


def test_rank_is_not_measured_when_no_svd_was_run():
    g = dh.build_grid(synth(with_svd=False))
    cell = g["cells"][0][0]
    assert cell["rel_fro"] is not None
    assert cell["rank99"] is None


def test_values_reproduce_the_source_json_exactly():
    src = synth()
    g = dh.build_grid(src)
    by_name = {n["name"]: n["rel_fro"] for n in src["all_norms"]}
    for li, row in enumerate(g["cells"]):
        for mi, cell in enumerate(row):
            assert cell["rel_fro"] == by_name[cell["tensor"]]


def test_ratio_uses_the_registered_epsilon_and_formula():
    a, b = dh.build_grid(synth()), dh.build_grid(synth())
    r = dh.ratio_grid(a, b)
    ra = a["cells"][5][1]["rel_fro"]
    rb = b["cells"][5][1]["rel_fro"]
    expected = math.log10((ra + dh.RATIO_EPS) / (rb + dh.RATIO_EPS))
    assert dh.RATIO_EPS == 1e-12
    assert r["cells"][5][1]["value"] == pytest.approx(expected, abs=1e-12)


def test_ratio_is_not_measured_when_either_side_is_missing():
    a = dh.build_grid(synth(drop=((7, "v_proj"),)))
    b = dh.build_grid(synth())
    r = dh.ratio_grid(a, b)
    assert r["cells"][7][dh.MODULES.index("v_proj")]["state"] == "not_measured"


def test_write_js_emits_a_parsable_global(tmp_path):
    p = tmp_path / "tensor_grid.js"
    dh.write_js({"models": {}}, p)
    text = p.read_text()
    assert text.startswith("window.TENSOR_GRID = ")
    assert json.loads(text[len("window.TENSOR_GRID = "):].rstrip().rstrip(";"))


def test_identical_model_reports_zero_changed_tensors():
    g = dh.build_grid(synth(changed=False))
    assert g["n_changed"] == 0
    assert g["identical"] is True


def test_rel_fro_is_recovered_from_svd_tensors_when_all_norms_is_absent():
    """The archived 2026-07-25 records carry no all_norms - that field postdates them - but
    their 40 svd_tensors entries each carry an exact rel_fro. Those are real measurements."""
    src = synth()
    src["all_norms"] = []
    g = dh.build_grid(src)
    assert g["n_measured_cells"] == 112
    assert g["cells"][0][0]["rel_fro"] is not None


def test_a_cell_in_neither_source_is_still_not_measured():
    src = synth(drop=((9, "o_proj"),))
    src["all_norms"] = []
    g = dh.build_grid(src)
    assert g["cells"][9][dh.MODULES.index("o_proj")]["state"] == "not_measured"


def test_an_archived_partial_record_reports_a_partial_grid_not_a_full_one():
    """40 of 112 attention tensors got an SVD in the archived run. The grid must say 40."""
    src = synth()
    src["all_norms"] = []
    src["svd_tensors"] = src["svd_tensors"][:40]
    g = dh.build_grid(src)
    assert g["n_measured_cells"] == 40
    missing = [c for row in g["cells"] for c in row if c["state"] == "not_measured"]
    assert len(missing) == 72
    assert all(c["rel_fro"] is None for c in missing)


def test_a_bit_identical_model_renders_measured_zeros_not_missing_values():
    """weight_diff.py sets identical only after comparing every shared tensor. That is a
    measurement of zero at every site, and the spec requires it render distinctly from
    missing. Organism C carries no changed tensors and therefore no per-cell rows."""
    src = synth(changed=False)
    src["all_norms"] = []
    src["svd_tensors"] = []
    g = dh.build_grid(src)
    assert g["identical"] is True
    assert g["n_measured_cells"] == 112
    assert all(c["state"] == "measured" and c["rel_fro"] == 0.0
               for row in g["cells"] for c in row)


def test_a_non_identical_model_never_gets_zeros_invented_for_it():
    src = synth(drop=((2, "q_proj"),))
    src["all_norms"] = []
    g = dh.build_grid(src)
    assert g["identical"] is False
    assert g["cells"][2][dh.MODULES.index("q_proj")]["state"] == "not_measured"
