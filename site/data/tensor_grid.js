window.TENSOR_GRID = {
 "models": {
  "a": {
   "target": "Alamerton/sl-organism-a-7b",
   "base": "Qwen/Qwen2.5-7B-Instruct",
   "identical": false,
   "n_changed": 112,
   "n_measured_cells": 40,
   "cells": [
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.0.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.0.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.v_proj.weight",
      "rel_fro": 0.1003385216729129,
      "rank99": 14,
      "top16_energy": 0.999734,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.o_proj.weight",
      "rel_fro": 0.07554220704656558,
      "rank99": 13,
      "top16_energy": 0.999519,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.1.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.1.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.1.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.1.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.2.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.2.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.v_proj.weight",
      "rel_fro": 0.06710731131877434,
      "rank99": 13,
      "top16_energy": 0.999408,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.o_proj.weight",
      "rel_fro": 0.07213000833173948,
      "rank99": 12,
      "top16_energy": 0.999474,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.3.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.3.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.3.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.o_proj.weight",
      "rel_fro": 0.07380043629324055,
      "rank99": 13,
      "top16_energy": 0.999497,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.4.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.4.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.4.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.o_proj.weight",
      "rel_fro": 0.06505334991601097,
      "rank99": 12,
      "top16_energy": 0.999352,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.5.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.5.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.5.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.o_proj.weight",
      "rel_fro": 0.0635287231713066,
      "rank99": 11,
      "top16_energy": 0.999321,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.6.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.6.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.6.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.o_proj.weight",
      "rel_fro": 0.06711333394301484,
      "rank99": 11,
      "top16_energy": 0.999391,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.7.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.7.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.7.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.o_proj.weight",
      "rel_fro": 0.06751674888277454,
      "rank99": 11,
      "top16_energy": 0.999398,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.8.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.8.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.8.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.8.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.9.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.9.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.9.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.9.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.10.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.10.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.10.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.o_proj.weight",
      "rel_fro": 0.06522398446381408,
      "rank99": 10,
      "top16_energy": 0.999357,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.11.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.11.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.11.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.o_proj.weight",
      "rel_fro": 0.06585995035368906,
      "rank99": 12,
      "top16_energy": 0.999367,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.12.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.12.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.12.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.12.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.13.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.13.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.13.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.13.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.14.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.14.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.14.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.14.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.15.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.15.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.15.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.o_proj.weight",
      "rel_fro": 0.06336066024678635,
      "rank99": 13,
      "top16_energy": 0.999317,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.16.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.16.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.16.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.16.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.17.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.17.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.17.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.o_proj.weight",
      "rel_fro": 0.06704819653352542,
      "rank99": 14,
      "top16_energy": 0.999391,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.q_proj.weight",
      "rel_fro": 0.07051036935295911,
      "rank99": 13,
      "top16_energy": 0.999449,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.18.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.18.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.18.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.q_proj.weight",
      "rel_fro": 0.07346562799776517,
      "rank99": 13,
      "top16_energy": 0.999492,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.k_proj.weight",
      "rel_fro": 0.07008552688070269,
      "rank99": 14,
      "top16_energy": 0.999458,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.19.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.19.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.q_proj.weight",
      "rel_fro": 0.07360428040031775,
      "rank99": 13,
      "top16_energy": 0.999494,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.k_proj.weight",
      "rel_fro": 0.06406122935302823,
      "rank99": 14,
      "top16_energy": 0.999351,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.20.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.o_proj.weight",
      "rel_fro": 0.0689248527288699,
      "rank99": 13,
      "top16_energy": 0.999422,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.q_proj.weight",
      "rel_fro": 0.07265318246380664,
      "rank99": 14,
      "top16_energy": 0.99948,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.k_proj.weight",
      "rel_fro": 0.0740739928538916,
      "rank99": 15,
      "top16_energy": 0.999514,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.21.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.o_proj.weight",
      "rel_fro": 0.06924448156381767,
      "rank99": 12,
      "top16_energy": 0.999428,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.q_proj.weight",
      "rel_fro": 0.08007911717410822,
      "rank99": 12,
      "top16_energy": 0.999572,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.k_proj.weight",
      "rel_fro": 0.07885072033973091,
      "rank99": 14,
      "top16_energy": 0.99957,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.22.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.o_proj.weight",
      "rel_fro": 0.08009501400646421,
      "rank99": 10,
      "top16_energy": 0.999571,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.q_proj.weight",
      "rel_fro": 0.07884097151766402,
      "rank99": 14,
      "top16_energy": 0.999558,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.k_proj.weight",
      "rel_fro": 0.07893487807057087,
      "rank99": 14,
      "top16_energy": 0.999571,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.v_proj.weight",
      "rel_fro": 0.0736931252990664,
      "rank99": 11,
      "top16_energy": 0.999511,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.o_proj.weight",
      "rel_fro": 0.08201100857381416,
      "rank99": 10,
      "top16_energy": 0.999591,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.q_proj.weight",
      "rel_fro": 0.07965120117709358,
      "rank99": 14,
      "top16_energy": 0.999567,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.k_proj.weight",
      "rel_fro": 0.07726722551020328,
      "rank99": 14,
      "top16_energy": 0.999553,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.v_proj.weight",
      "rel_fro": 0.06830204621460187,
      "rank99": 10,
      "top16_energy": 0.999429,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.o_proj.weight",
      "rel_fro": 0.08805669913079316,
      "rank99": 10,
      "top16_energy": 0.999644,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.q_proj.weight",
      "rel_fro": 0.08319142263652851,
      "rank99": 14,
      "top16_energy": 0.999602,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.k_proj.weight",
      "rel_fro": 0.08500576528664366,
      "rank99": 15,
      "top16_energy": 0.999631,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.25.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.o_proj.weight",
      "rel_fro": 0.08204172190695769,
      "rank99": 12,
      "top16_energy": 0.999591,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.q_proj.weight",
      "rel_fro": 0.06863489969483486,
      "rank99": 15,
      "top16_energy": 0.999419,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.k_proj.weight",
      "rel_fro": 0.07959586059134313,
      "rank99": 15,
      "top16_energy": 0.99958,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.26.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.26.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.q_proj.weight",
      "rel_fro": 0.06930570277751041,
      "rank99": 13,
      "top16_energy": 0.999429,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.k_proj.weight",
      "rel_fro": 0.07033860974595102,
      "rank99": 14,
      "top16_energy": 0.999464,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.27.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.27.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ]
   ],
   "modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
   ],
   "n_layers": 28
  },
  "b": {
   "target": "Alamerton/sl-organism-b-7b",
   "base": "Qwen/Qwen2.5-7B-Instruct",
   "identical": false,
   "n_changed": 112,
   "n_measured_cells": 40,
   "cells": [
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.0.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.0.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.v_proj.weight",
      "rel_fro": 0.09549151747863,
      "rank99": 15,
      "top16_energy": 0.999706,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.o_proj.weight",
      "rel_fro": 0.07192927489850504,
      "rank99": 13,
      "top16_energy": 0.999471,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.1.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.1.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.1.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.1.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.2.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.2.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.v_proj.weight",
      "rel_fro": 0.06937225094137636,
      "rank99": 13,
      "top16_energy": 0.999445,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.o_proj.weight",
      "rel_fro": 0.06945280306212527,
      "rank99": 11,
      "top16_energy": 0.999433,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.3.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.3.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.3.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.o_proj.weight",
      "rel_fro": 0.06331806330962257,
      "rank99": 14,
      "top16_energy": 0.999317,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.4.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.4.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.4.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.o_proj.weight",
      "rel_fro": 0.0706528150738001,
      "rank99": 11,
      "top16_energy": 0.999451,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.5.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.5.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.5.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.o_proj.weight",
      "rel_fro": 0.06498265174130492,
      "rank99": 10,
      "top16_energy": 0.999351,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.6.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.6.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.6.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.o_proj.weight",
      "rel_fro": 0.0637357733936149,
      "rank99": 11,
      "top16_energy": 0.999326,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.7.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.7.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.7.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.o_proj.weight",
      "rel_fro": 0.06629112636616624,
      "rank99": 11,
      "top16_energy": 0.999376,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.8.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.8.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.8.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.8.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.9.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.9.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.9.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.9.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.10.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.10.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.10.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.o_proj.weight",
      "rel_fro": 0.06674202556226025,
      "rank99": 7,
      "top16_energy": 0.999385,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.11.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.11.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.11.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.o_proj.weight",
      "rel_fro": 0.06384533826717795,
      "rank99": 11,
      "top16_energy": 0.999327,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.12.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.12.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.12.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.12.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.13.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.13.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.13.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.13.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.q_proj.weight",
      "rel_fro": 0.0652545129218064,
      "rank99": 12,
      "top16_energy": 0.999357,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.14.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.14.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.14.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.15.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.15.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.15.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.o_proj.weight",
      "rel_fro": 0.06303098349296811,
      "rank99": 12,
      "top16_energy": 0.999311,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.16.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.16.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.16.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.16.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "not_measured",
      "tensor": "model.layers.17.self_attn.q_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.17.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.17.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.o_proj.weight",
      "rel_fro": 0.06434850413776975,
      "rank99": 12,
      "top16_energy": 0.999338,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.q_proj.weight",
      "rel_fro": 0.06738593001368219,
      "rank99": 13,
      "top16_energy": 0.999396,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.18.self_attn.k_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.18.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.18.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.q_proj.weight",
      "rel_fro": 0.071180701710288,
      "rank99": 12,
      "top16_energy": 0.999458,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.k_proj.weight",
      "rel_fro": 0.0670826478777902,
      "rank99": 14,
      "top16_energy": 0.999407,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.19.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.19.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.q_proj.weight",
      "rel_fro": 0.07604173310984368,
      "rank99": 12,
      "top16_energy": 0.999525,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.k_proj.weight",
      "rel_fro": 0.0721151923358933,
      "rank99": 13,
      "top16_energy": 0.999486,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.20.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.o_proj.weight",
      "rel_fro": 0.06801113978112266,
      "rank99": 12,
      "top16_energy": 0.999407,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.q_proj.weight",
      "rel_fro": 0.07496019689837873,
      "rank99": 13,
      "top16_energy": 0.999512,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.k_proj.weight",
      "rel_fro": 0.07118937397600825,
      "rank99": 14,
      "top16_energy": 0.999474,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.21.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.21.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.q_proj.weight",
      "rel_fro": 0.08035149205256022,
      "rank99": 13,
      "top16_energy": 0.999574,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.k_proj.weight",
      "rel_fro": 0.06877502166951345,
      "rank99": 14,
      "top16_energy": 0.999436,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.22.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.o_proj.weight",
      "rel_fro": 0.07402633001308,
      "rank99": 9,
      "top16_energy": 0.999499,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.q_proj.weight",
      "rel_fro": 0.07813376331574902,
      "rank99": 12,
      "top16_energy": 0.99955,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.k_proj.weight",
      "rel_fro": 0.07703663501484892,
      "rank99": 14,
      "top16_energy": 0.99955,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.v_proj.weight",
      "rel_fro": 0.06855829857362507,
      "rank99": 10,
      "top16_energy": 0.999434,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.o_proj.weight",
      "rel_fro": 0.07612996154782803,
      "rank99": 10,
      "top16_energy": 0.999526,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.q_proj.weight",
      "rel_fro": 0.0783953722536926,
      "rank99": 13,
      "top16_energy": 0.999553,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.k_proj.weight",
      "rel_fro": 0.07458078037327492,
      "rank99": 14,
      "top16_energy": 0.999519,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.v_proj.weight",
      "rel_fro": 0.06956559813447356,
      "rank99": 10,
      "top16_energy": 0.99945,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.o_proj.weight",
      "rel_fro": 0.08667159552077153,
      "rank99": 10,
      "top16_energy": 0.999633,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.q_proj.weight",
      "rel_fro": 0.0827121655520559,
      "rank99": 13,
      "top16_energy": 0.999598,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.k_proj.weight",
      "rel_fro": 0.0805020071958191,
      "rank99": 15,
      "top16_energy": 0.999588,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.25.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.o_proj.weight",
      "rel_fro": 0.08474499860060393,
      "rank99": 10,
      "top16_energy": 0.999617,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.q_proj.weight",
      "rel_fro": 0.07208379214811024,
      "rank99": 15,
      "top16_energy": 0.999473,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.k_proj.weight",
      "rel_fro": 0.07838311814557987,
      "rank99": 14,
      "top16_energy": 0.999564,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.26.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.26.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.q_proj.weight",
      "rel_fro": 0.07404785197719366,
      "rank99": 13,
      "top16_energy": 0.9995,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.k_proj.weight",
      "rel_fro": 0.07223740287664333,
      "rank99": 14,
      "top16_energy": 0.999492,
      "looks_like_lora": true
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.27.self_attn.v_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "not_measured",
      "tensor": "model.layers.27.self_attn.o_proj.weight",
      "rel_fro": null,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ]
   ],
   "modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
   ],
   "n_layers": 28
  },
  "c": {
   "target": "Alamerton/sl-organism-c-7b",
   "base": "Qwen/Qwen2.5-7B-Instruct",
   "identical": true,
   "n_changed": 0,
   "n_measured_cells": 112,
   "cells": [
    [
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ]
   ],
   "modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
   ],
   "n_layers": 28
  },
  "base": {
   "target": "Qwen/Qwen2.5-7B-Instruct",
   "base": "Qwen/Qwen2.5-7B-Instruct",
   "identical": true,
   "n_changed": 0,
   "n_measured_cells": 112,
   "cells": [
    [
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.20.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.22.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.23.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.24.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.25.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.q_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.k_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.v_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.o_proj.weight",
      "rel_fro": 0.0,
      "rank99": null,
      "top16_energy": null,
      "looks_like_lora": null
     }
    ]
   ],
   "modules": [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
   ],
   "n_layers": 28
  }
 },
 "source_run": "2026-07-25_modal-a10g_weightdiff-v1",
 "caption": "This localises the modification and identifies its family. It cannot identify its purpose: dW says where and how much, never what for.",
 "a_vs_b": {
  "formula": "log10((rel_fro_A + 1e-12) / (rel_fro_B + 1e-12))",
  "epsilon": 1e-12,
  "cells": [
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.0.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.0.self_attn.k_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.0215029032330618,
     "tensor": "model.layers.0.self_attn.v_proj.weight",
     "rel_fro_a": 0.1003385216729129,
     "rel_fro_b": 0.09549151747863
    },
    {
     "state": "measured",
     "value": 0.0212839867494623,
     "tensor": "model.layers.0.self_attn.o_proj.weight",
     "rel_fro_a": 0.07554220704656558,
     "rel_fro_b": 0.07192927489850504
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.1.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.1.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.1.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.1.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.2.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.2.self_attn.k_proj.weight"
    },
    {
     "state": "measured",
     "value": -0.0144159474228798,
     "tensor": "model.layers.2.self_attn.v_proj.weight",
     "rel_fro_a": 0.06710731131877434,
     "rel_fro_b": 0.06937225094137636
    },
    {
     "state": "measured",
     "value": 0.016426204134021363,
     "tensor": "model.layers.2.self_attn.o_proj.weight",
     "rel_fro_a": 0.07213000833173948,
     "rel_fro_b": 0.06945280306212527
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.3.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.3.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.3.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.06653130653071232,
     "tensor": "model.layers.3.self_attn.o_proj.weight",
     "rel_fro_a": 0.07380043629324055,
     "rel_fro_b": 0.06331806330962257
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.4.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.4.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.4.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": -0.03585980502773987,
     "tensor": "model.layers.4.self_attn.o_proj.weight",
     "rel_fro_a": 0.06505334991601097,
     "rel_fro_b": 0.0706528150738001
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.5.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.5.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.5.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": -0.009827302808583212,
     "tensor": "model.layers.5.self_attn.o_proj.weight",
     "rel_fro_a": 0.0635287231713066,
     "rel_fro_b": 0.06498265174130492
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.6.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.6.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.6.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.02242555343246914,
     "tensor": "model.layers.6.self_attn.o_proj.weight",
     "rel_fro_a": 0.06711333394301484,
     "rel_fro_b": 0.0637357733936149
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.7.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.7.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.7.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.007956123379619569,
     "tensor": "model.layers.7.self_attn.o_proj.weight",
     "rel_fro_a": 0.06751674888277454,
     "rel_fro_b": 0.06629112636616624
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.8.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.8.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.8.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.8.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.9.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.9.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.9.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.9.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.10.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.10.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.10.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": -0.009992057109193691,
     "tensor": "model.layers.10.self_attn.o_proj.weight",
     "rel_fro_a": 0.06522398446381408,
     "rel_fro_b": 0.06674202556226025
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.11.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.11.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.11.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.013492206697231862,
     "tensor": "model.layers.11.self_attn.o_proj.weight",
     "rel_fro_a": 0.06585995035368906,
     "rel_fro_b": 0.06384533826717795
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.12.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.12.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.12.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.12.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.13.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.13.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.13.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.13.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.14.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.14.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.14.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.14.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.15.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.15.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.15.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.0022656105334882973,
     "tensor": "model.layers.15.self_attn.o_proj.weight",
     "rel_fro_a": 0.06336066024678635,
     "rel_fro_b": 0.06303098349296811
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.16.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.16.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.16.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.16.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.17.self_attn.q_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.17.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.17.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.017848645030880423,
     "tensor": "model.layers.17.self_attn.o_proj.weight",
     "rel_fro_a": 0.06704819653352542,
     "rel_fro_b": 0.06434850413776975
    }
   ],
   [
    {
     "state": "measured",
     "value": 0.01968376306287318,
     "tensor": "model.layers.18.self_attn.q_proj.weight",
     "rel_fro_a": 0.07051036935295911,
     "rel_fro_b": 0.06738593001368219
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.18.self_attn.k_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.18.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.18.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "measured",
     "value": 0.013721930347222834,
     "tensor": "model.layers.19.self_attn.q_proj.weight",
     "rel_fro_a": 0.07346562799776517,
     "rel_fro_b": 0.071180701710288
    },
    {
     "state": "measured",
     "value": 0.019018145887564973,
     "tensor": "model.layers.19.self_attn.k_proj.weight",
     "rel_fro_a": 0.07008552688070269,
     "rel_fro_b": 0.0670826478777902
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.19.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.19.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.014148935421968782,
     "tensor": "model.layers.20.self_attn.q_proj.weight",
     "rel_fro_a": 0.07360428040031775,
     "rel_fro_b": 0.07604173310984368
    },
    {
     "state": "measured",
     "value": -0.05143149750067401,
     "tensor": "model.layers.20.self_attn.k_proj.weight",
     "rel_fro_a": 0.06406122935302823,
     "rel_fro_b": 0.0721151923358933
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.20.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.005795793689295392,
     "tensor": "model.layers.20.self_attn.o_proj.weight",
     "rel_fro_a": 0.0689248527288699,
     "rel_fro_b": 0.06801113978112266
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.013576075975030532,
     "tensor": "model.layers.21.self_attn.q_proj.weight",
     "rel_fro_a": 0.07265318246380664,
     "rel_fro_b": 0.07496019689837873
    },
    {
     "state": "measured",
     "value": 0.01725058145071248,
     "tensor": "model.layers.21.self_attn.k_proj.weight",
     "rel_fro_a": 0.0740739928538916,
     "rel_fro_b": 0.07118937397600825
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.21.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.21.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.001474668966788926,
     "tensor": "model.layers.22.self_attn.q_proj.weight",
     "rel_fro_a": 0.08007911717410822,
     "rel_fro_b": 0.08035149205256022
    },
    {
     "state": "measured",
     "value": 0.05937492927534475,
     "tensor": "model.layers.22.self_attn.k_proj.weight",
     "rel_fro_a": 0.07885072033973091,
     "rel_fro_b": 0.06877502166951345
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.22.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": 0.03421926267101621,
     "tensor": "model.layers.22.self_attn.o_proj.weight",
     "rel_fro_a": 0.08009501400646421,
     "rel_fro_b": 0.07402633001308
    }
   ],
   [
    {
     "state": "measured",
     "value": 0.003913224608980415,
     "tensor": "model.layers.23.self_attn.q_proj.weight",
     "rel_fro_a": 0.07884097151766402,
     "rel_fro_b": 0.07813376331574902
    },
    {
     "state": "measured",
     "value": 0.010571638049184515,
     "tensor": "model.layers.23.self_attn.k_proj.weight",
     "rel_fro_a": 0.07893487807057087,
     "rel_fro_b": 0.07703663501484892
    },
    {
     "state": "measured",
     "value": 0.03136694411924439,
     "tensor": "model.layers.23.self_attn.v_proj.weight",
     "rel_fro_a": 0.0736931252990664,
     "rel_fro_b": 0.06855829857362507
    },
    {
     "state": "measured",
     "value": 0.03231654245774523,
     "tensor": "model.layers.23.self_attn.o_proj.weight",
     "rel_fro_a": 0.08201100857381416,
     "rel_fro_b": 0.07612996154782803
    }
   ],
   [
    {
     "state": "measured",
     "value": 0.006901902884886229,
     "tensor": "model.layers.24.self_attn.q_proj.weight",
     "rel_fro_a": 0.07965120117709358,
     "rel_fro_b": 0.0783953722536926
    },
    {
     "state": "measured",
     "value": 0.015368394727189235,
     "tensor": "model.layers.24.self_attn.k_proj.weight",
     "rel_fro_a": 0.07726722551020328,
     "rel_fro_b": 0.07458078037327492
    },
    {
     "state": "measured",
     "value": -0.007960808995138215,
     "tensor": "model.layers.24.self_attn.v_proj.weight",
     "rel_fro_a": 0.06830204621460187,
     "rel_fro_b": 0.06956559813447356
    },
    {
     "state": "measured",
     "value": 0.006885610103805924,
     "tensor": "model.layers.24.self_attn.o_proj.weight",
     "rel_fro_a": 0.08805669913079316,
     "rel_fro_b": 0.08667159552077153
    }
   ],
   [
    {
     "state": "measured",
     "value": 0.002509159547950289,
     "tensor": "model.layers.25.self_attn.q_proj.weight",
     "rel_fro_a": 0.08319142263652851,
     "rel_fro_b": 0.0827121655520559
    },
    {
     "state": "measured",
     "value": 0.02364167258513868,
     "tensor": "model.layers.25.self_attn.k_proj.weight",
     "rel_fro_a": 0.08500576528664366,
     "rel_fro_b": 0.0805020071958191
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.25.self_attn.v_proj.weight"
    },
    {
     "state": "measured",
     "value": -0.014079310008711873,
     "tensor": "model.layers.25.self_attn.o_proj.weight",
     "rel_fro_a": 0.08204172190695769,
     "rel_fro_b": 0.08474499860060393
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.021292622391600683,
     "tensor": "model.layers.26.self_attn.q_proj.weight",
     "rel_fro_a": 0.06863489969483486,
     "rel_fro_b": 0.07208379214811024
    },
    {
     "state": "measured",
     "value": 0.006667946617715564,
     "tensor": "model.layers.26.self_attn.k_proj.weight",
     "rel_fro_a": 0.07959586059134313,
     "rel_fro_b": 0.07838311814557987
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.26.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.26.self_attn.o_proj.weight"
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.028743493035820952,
     "tensor": "model.layers.27.self_attn.q_proj.weight",
     "rel_fro_a": 0.06930570277751041,
     "rel_fro_b": 0.07404785197719366
    },
    {
     "state": "measured",
     "value": -0.011568343399963169,
     "tensor": "model.layers.27.self_attn.k_proj.weight",
     "rel_fro_a": 0.07033860974595102,
     "rel_fro_b": 0.07223740287664333
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.27.self_attn.v_proj.weight"
    },
    {
     "state": "not_measured",
     "value": null,
     "tensor": "model.layers.27.self_attn.o_proj.weight"
    }
   ]
  ],
  "modules": [
   "q_proj",
   "k_proj",
   "v_proj",
   "o_proj"
  ],
  "n_layers": 28,
  "means": "edit magnitude only: positive = A changed more at this site than B. Not direction, not objective, not loyalty."
 }
};
