window.TENSOR_GRID = {
 "models": {
  "a": {
   "target": "Alamerton/sl-organism-a-7b",
   "base": "Qwen/Qwen2.5-7B-Instruct",
   "identical": false,
   "n_changed": 112,
   "n_measured_cells": 112,
   "cells": [
    [
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.q_proj.weight",
      "rel_fro": 0.034332300014417214,
      "rank99": 16,
      "top16_energy": 0.997688,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.k_proj.weight",
      "rel_fro": 0.033996916753234636,
      "rank99": 14,
      "top16_energy": 0.997701,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.1.self_attn.q_proj.weight",
      "rel_fro": 0.06315562203726045,
      "rank99": 11,
      "top16_energy": 0.999314,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.k_proj.weight",
      "rel_fro": 0.04304352413656633,
      "rank99": 11,
      "top16_energy": 0.998569,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.v_proj.weight",
      "rel_fro": 0.045806525890865986,
      "rank99": 12,
      "top16_energy": 0.998728,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.o_proj.weight",
      "rel_fro": 0.05633305370878085,
      "rank99": 13,
      "top16_energy": 0.999138,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.q_proj.weight",
      "rel_fro": 0.05572180079180701,
      "rank99": 14,
      "top16_energy": 0.999121,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.k_proj.weight",
      "rel_fro": 0.04182942775756469,
      "rank99": 14,
      "top16_energy": 0.998486,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.3.self_attn.q_proj.weight",
      "rel_fro": 0.0631319071338049,
      "rank99": 12,
      "top16_energy": 0.999314,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.k_proj.weight",
      "rel_fro": 0.04314523351275869,
      "rank99": 14,
      "top16_energy": 0.998573,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.v_proj.weight",
      "rel_fro": 0.060892812956904074,
      "rank99": 14,
      "top16_energy": 0.999284,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.4.self_attn.q_proj.weight",
      "rel_fro": 0.04493107282254742,
      "rank99": 15,
      "top16_energy": 0.998648,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.k_proj.weight",
      "rel_fro": 0.040842478384152024,
      "rank99": 14,
      "top16_energy": 0.998411,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.v_proj.weight",
      "rel_fro": 0.052492127043263295,
      "rank99": 12,
      "top16_energy": 0.999034,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.5.self_attn.q_proj.weight",
      "rel_fro": 0.051977867101010565,
      "rank99": 14,
      "top16_energy": 0.998988,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.k_proj.weight",
      "rel_fro": 0.042087006962800454,
      "rank99": 14,
      "top16_energy": 0.998496,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.v_proj.weight",
      "rel_fro": 0.05432780828323204,
      "rank99": 11,
      "top16_energy": 0.999099,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.6.self_attn.q_proj.weight",
      "rel_fro": 0.053993371351894935,
      "rank99": 14,
      "top16_energy": 0.999062,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.k_proj.weight",
      "rel_fro": 0.041391924471310404,
      "rank99": 14,
      "top16_energy": 0.998451,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.v_proj.weight",
      "rel_fro": 0.051707752146553876,
      "rank99": 12,
      "top16_energy": 0.999005,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.7.self_attn.q_proj.weight",
      "rel_fro": 0.057164047644090145,
      "rank99": 14,
      "top16_energy": 0.999162,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.k_proj.weight",
      "rel_fro": 0.05328117374328941,
      "rank99": 14,
      "top16_energy": 0.999064,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.v_proj.weight",
      "rel_fro": 0.04330229699865594,
      "rank99": 13,
      "top16_energy": 0.998582,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.8.self_attn.q_proj.weight",
      "rel_fro": 0.054674478213439695,
      "rank99": 14,
      "top16_energy": 0.999087,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.k_proj.weight",
      "rel_fro": 0.04422998875190484,
      "rank99": 14,
      "top16_energy": 0.998649,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.v_proj.weight",
      "rel_fro": 0.05766959453579635,
      "rank99": 12,
      "top16_energy": 0.999198,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.o_proj.weight",
      "rel_fro": 0.06283999487609095,
      "rank99": 12,
      "top16_energy": 0.999306,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.q_proj.weight",
      "rel_fro": 0.05634908888242359,
      "rank99": 14,
      "top16_energy": 0.999138,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.k_proj.weight",
      "rel_fro": 0.05723902980390517,
      "rank99": 13,
      "top16_energy": 0.999187,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.v_proj.weight",
      "rel_fro": 0.041773012791660034,
      "rank99": 12,
      "top16_energy": 0.998476,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.o_proj.weight",
      "rel_fro": 0.057641386923624346,
      "rank99": 13,
      "top16_energy": 0.999178,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.q_proj.weight",
      "rel_fro": 0.05610155643394292,
      "rank99": 13,
      "top16_energy": 0.999132,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.k_proj.weight",
      "rel_fro": 0.04068999363888298,
      "rank99": 14,
      "top16_energy": 0.998395,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.v_proj.weight",
      "rel_fro": 0.059678909017633584,
      "rank99": 10,
      "top16_energy": 0.999253,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.11.self_attn.q_proj.weight",
      "rel_fro": 0.05279766477936164,
      "rank99": 14,
      "top16_energy": 0.99902,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.k_proj.weight",
      "rel_fro": 0.04453532841072004,
      "rank99": 13,
      "top16_energy": 0.998663,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.v_proj.weight",
      "rel_fro": 0.05612252186268664,
      "rank99": 12,
      "top16_energy": 0.999154,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.12.self_attn.q_proj.weight",
      "rel_fro": 0.05577541555439528,
      "rank99": 14,
      "top16_energy": 0.999121,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.k_proj.weight",
      "rel_fro": 0.045853201650013914,
      "rank99": 14,
      "top16_energy": 0.998733,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.v_proj.weight",
      "rel_fro": 0.06050555659646374,
      "rank99": 12,
      "top16_energy": 0.999273,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.o_proj.weight",
      "rel_fro": 0.06255358650984877,
      "rank99": 12,
      "top16_energy": 0.999299,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.q_proj.weight",
      "rel_fro": 0.05671496188899244,
      "rank99": 14,
      "top16_energy": 0.99915,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.k_proj.weight",
      "rel_fro": 0.0481860742607173,
      "rank99": 14,
      "top16_energy": 0.998856,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.v_proj.weight",
      "rel_fro": 0.05123117337542461,
      "rank99": 12,
      "top16_energy": 0.998983,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.o_proj.weight",
      "rel_fro": 0.05749620143883174,
      "rank99": 14,
      "top16_energy": 0.999172,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.q_proj.weight",
      "rel_fro": 0.057323104426268735,
      "rank99": 14,
      "top16_energy": 0.999168,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.k_proj.weight",
      "rel_fro": 0.051025147642030796,
      "rank99": 13,
      "top16_energy": 0.99898,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.v_proj.weight",
      "rel_fro": 0.05722466114743865,
      "rank99": 13,
      "top16_energy": 0.999188,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.o_proj.weight",
      "rel_fro": 0.056697947191840646,
      "rank99": 12,
      "top16_energy": 0.99915,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.q_proj.weight",
      "rel_fro": 0.05647557569357597,
      "rank99": 14,
      "top16_energy": 0.999143,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.k_proj.weight",
      "rel_fro": 0.04428352630397706,
      "rank99": 14,
      "top16_energy": 0.998649,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.v_proj.weight",
      "rel_fro": 0.05346460233837113,
      "rank99": 13,
      "top16_energy": 0.999066,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.16.self_attn.q_proj.weight",
      "rel_fro": 0.06182045294595176,
      "rank99": 13,
      "top16_energy": 0.999284,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.k_proj.weight",
      "rel_fro": 0.05178494340441699,
      "rank99": 14,
      "top16_energy": 0.999008,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.v_proj.weight",
      "rel_fro": 0.04949028111316841,
      "rank99": 13,
      "top16_energy": 0.998912,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.o_proj.weight",
      "rel_fro": 0.05742628967649811,
      "rank99": 13,
      "top16_energy": 0.999169,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.q_proj.weight",
      "rel_fro": 0.06050471010060516,
      "rank99": 14,
      "top16_energy": 0.999254,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.k_proj.weight",
      "rel_fro": 0.055660869192279076,
      "rank99": 13,
      "top16_energy": 0.999143,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.v_proj.weight",
      "rel_fro": 0.05226671597906625,
      "rank99": 13,
      "top16_energy": 0.999022,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.18.self_attn.k_proj.weight",
      "rel_fro": 0.058208891623658415,
      "rank99": 14,
      "top16_energy": 0.999215,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.v_proj.weight",
      "rel_fro": 0.05871892458996798,
      "rank99": 14,
      "top16_energy": 0.99923,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.o_proj.weight",
      "rel_fro": 0.06077122502825238,
      "rank99": 14,
      "top16_energy": 0.999258,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.19.self_attn.v_proj.weight",
      "rel_fro": 0.05672755116015651,
      "rank99": 15,
      "top16_energy": 0.999172,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.o_proj.weight",
      "rel_fro": 0.06071037194461304,
      "rank99": 14,
      "top16_energy": 0.999256,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.20.self_attn.v_proj.weight",
      "rel_fro": 0.061392183436543284,
      "rank99": 12,
      "top16_energy": 0.999292,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.21.self_attn.v_proj.weight",
      "rel_fro": 0.05888642036527627,
      "rank99": 13,
      "top16_energy": 0.999229,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.22.self_attn.v_proj.weight",
      "rel_fro": 0.060604269781589226,
      "rank99": 11,
      "top16_energy": 0.999278,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.25.self_attn.v_proj.weight",
      "rel_fro": 0.0627448890125025,
      "rank99": 13,
      "top16_energy": 0.999325,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.26.self_attn.v_proj.weight",
      "rel_fro": 0.049936300871515675,
      "rank99": 10,
      "top16_energy": 0.998941,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.o_proj.weight",
      "rel_fro": 0.06141565172871477,
      "rank99": 11,
      "top16_energy": 0.999277,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.27.self_attn.v_proj.weight",
      "rel_fro": 0.04018350191350736,
      "rank99": 13,
      "top16_energy": 0.99835,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.o_proj.weight",
      "rel_fro": 0.06284550406986482,
      "rank99": 10,
      "top16_energy": 0.999317,
      "looks_like_lora": true
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
   "n_measured_cells": 112,
   "cells": [
    [
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.q_proj.weight",
      "rel_fro": 0.03246659952627987,
      "rank99": 16,
      "top16_energy": 0.997417,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.0.self_attn.k_proj.weight",
      "rel_fro": 0.038028076350383376,
      "rank99": 12,
      "top16_energy": 0.998161,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.1.self_attn.q_proj.weight",
      "rel_fro": 0.06288656180502862,
      "rank99": 10,
      "top16_energy": 0.999309,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.k_proj.weight",
      "rel_fro": 0.04069027967203652,
      "rank99": 11,
      "top16_energy": 0.998402,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.v_proj.weight",
      "rel_fro": 0.043649249934117944,
      "rank99": 13,
      "top16_energy": 0.998606,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.1.self_attn.o_proj.weight",
      "rel_fro": 0.06202870500750134,
      "rank99": 13,
      "top16_energy": 0.999289,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.q_proj.weight",
      "rel_fro": 0.05636315124994294,
      "rank99": 14,
      "top16_energy": 0.99914,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.2.self_attn.k_proj.weight",
      "rel_fro": 0.04789151950215346,
      "rank99": 12,
      "top16_energy": 0.998844,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.3.self_attn.q_proj.weight",
      "rel_fro": 0.05871032813098569,
      "rank99": 13,
      "top16_energy": 0.999207,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.k_proj.weight",
      "rel_fro": 0.04546196098130827,
      "rank99": 14,
      "top16_energy": 0.998718,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.3.self_attn.v_proj.weight",
      "rel_fro": 0.06238636825993237,
      "rank99": 14,
      "top16_energy": 0.999317,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.4.self_attn.q_proj.weight",
      "rel_fro": 0.045473056943187866,
      "rank99": 15,
      "top16_energy": 0.998681,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.k_proj.weight",
      "rel_fro": 0.04341937696872584,
      "rank99": 13,
      "top16_energy": 0.998595,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.4.self_attn.v_proj.weight",
      "rel_fro": 0.050194067697363724,
      "rank99": 12,
      "top16_energy": 0.998943,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.5.self_attn.q_proj.weight",
      "rel_fro": 0.05650414049968555,
      "rank99": 13,
      "top16_energy": 0.999143,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.k_proj.weight",
      "rel_fro": 0.045001062621080425,
      "rank99": 13,
      "top16_energy": 0.998687,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.5.self_attn.v_proj.weight",
      "rel_fro": 0.05354310232630128,
      "rank99": 12,
      "top16_energy": 0.999069,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.6.self_attn.q_proj.weight",
      "rel_fro": 0.05248307876310791,
      "rank99": 14,
      "top16_energy": 0.999008,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.k_proj.weight",
      "rel_fro": 0.042125343742496095,
      "rank99": 14,
      "top16_energy": 0.998503,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.6.self_attn.v_proj.weight",
      "rel_fro": 0.053253489150580376,
      "rank99": 11,
      "top16_energy": 0.999061,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.7.self_attn.q_proj.weight",
      "rel_fro": 0.0612937740097453,
      "rank99": 14,
      "top16_energy": 0.999272,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.k_proj.weight",
      "rel_fro": 0.052990898199041514,
      "rank99": 14,
      "top16_energy": 0.999053,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.7.self_attn.v_proj.weight",
      "rel_fro": 0.048714142494939,
      "rank99": 12,
      "top16_energy": 0.998879,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.8.self_attn.q_proj.weight",
      "rel_fro": 0.05437806181211376,
      "rank99": 14,
      "top16_energy": 0.999075,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.k_proj.weight",
      "rel_fro": 0.04606407238953168,
      "rank99": 14,
      "top16_energy": 0.998751,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.v_proj.weight",
      "rel_fro": 0.05729021245257796,
      "rank99": 11,
      "top16_energy": 0.999189,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.8.self_attn.o_proj.weight",
      "rel_fro": 0.05846609282573212,
      "rank99": 11,
      "top16_energy": 0.999199,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.q_proj.weight",
      "rel_fro": 0.05608990780397771,
      "rank99": 14,
      "top16_energy": 0.99913,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.k_proj.weight",
      "rel_fro": 0.053673057551788986,
      "rank99": 14,
      "top16_energy": 0.999075,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.v_proj.weight",
      "rel_fro": 0.03972538921391225,
      "rank99": 13,
      "top16_energy": 0.998319,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.9.self_attn.o_proj.weight",
      "rel_fro": 0.0553190673888548,
      "rank99": 13,
      "top16_energy": 0.999107,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.q_proj.weight",
      "rel_fro": 0.060384995028256984,
      "rank99": 12,
      "top16_energy": 0.999249,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.k_proj.weight",
      "rel_fro": 0.042891046674960305,
      "rank99": 13,
      "top16_energy": 0.998557,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.10.self_attn.v_proj.weight",
      "rel_fro": 0.05669756329509671,
      "rank99": 10,
      "top16_energy": 0.999171,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.11.self_attn.q_proj.weight",
      "rel_fro": 0.05408232219078364,
      "rank99": 13,
      "top16_energy": 0.999067,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.k_proj.weight",
      "rel_fro": 0.04369231897417993,
      "rank99": 12,
      "top16_energy": 0.998608,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.11.self_attn.v_proj.weight",
      "rel_fro": 0.05473577239101804,
      "rank99": 11,
      "top16_energy": 0.999111,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.12.self_attn.q_proj.weight",
      "rel_fro": 0.055042128427818186,
      "rank99": 14,
      "top16_energy": 0.999099,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.k_proj.weight",
      "rel_fro": 0.04579453973470166,
      "rank99": 13,
      "top16_energy": 0.998732,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.v_proj.weight",
      "rel_fro": 0.06118989774513341,
      "rank99": 11,
      "top16_energy": 0.999289,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.12.self_attn.o_proj.weight",
      "rel_fro": 0.057875836754166195,
      "rank99": 12,
      "top16_energy": 0.999184,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.q_proj.weight",
      "rel_fro": 0.058848635593041244,
      "rank99": 13,
      "top16_energy": 0.99921,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.k_proj.weight",
      "rel_fro": 0.04932748693416063,
      "rank99": 14,
      "top16_energy": 0.998912,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.v_proj.weight",
      "rel_fro": 0.04765148003804792,
      "rank99": 13,
      "top16_energy": 0.99883,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.13.self_attn.o_proj.weight",
      "rel_fro": 0.054534882200976485,
      "rank99": 13,
      "top16_energy": 0.999081,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.14.self_attn.k_proj.weight",
      "rel_fro": 0.05723761255439215,
      "rank99": 11,
      "top16_energy": 0.999187,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.v_proj.weight",
      "rel_fro": 0.056744518861749235,
      "rank99": 12,
      "top16_energy": 0.999176,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.14.self_attn.o_proj.weight",
      "rel_fro": 0.055482597868601075,
      "rank99": 12,
      "top16_energy": 0.999112,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.q_proj.weight",
      "rel_fro": 0.05961384007014781,
      "rank99": 13,
      "top16_energy": 0.999229,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.k_proj.weight",
      "rel_fro": 0.04379239752864364,
      "rank99": 14,
      "top16_energy": 0.998615,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.15.self_attn.v_proj.weight",
      "rel_fro": 0.05228202513766347,
      "rank99": 13,
      "top16_energy": 0.999024,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.16.self_attn.q_proj.weight",
      "rel_fro": 0.06074479643198569,
      "rank99": 13,
      "top16_energy": 0.999258,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.k_proj.weight",
      "rel_fro": 0.05502568897444302,
      "rank99": 13,
      "top16_energy": 0.999122,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.v_proj.weight",
      "rel_fro": 0.049109588321025716,
      "rank99": 12,
      "top16_energy": 0.998896,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.16.self_attn.o_proj.weight",
      "rel_fro": 0.055959121314275155,
      "rank99": 13,
      "top16_energy": 0.999126,
      "looks_like_lora": true
     }
    ],
    [
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.q_proj.weight",
      "rel_fro": 0.05806058046122742,
      "rank99": 13,
      "top16_energy": 0.999188,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.k_proj.weight",
      "rel_fro": 0.05783964369199558,
      "rank99": 13,
      "top16_energy": 0.999203,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.17.self_attn.v_proj.weight",
      "rel_fro": 0.05409791288810837,
      "rank99": 12,
      "top16_energy": 0.999087,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.18.self_attn.k_proj.weight",
      "rel_fro": 0.059364314211979956,
      "rank99": 14,
      "top16_energy": 0.999244,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.v_proj.weight",
      "rel_fro": 0.05513336887308096,
      "rank99": 13,
      "top16_energy": 0.999125,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.18.self_attn.o_proj.weight",
      "rel_fro": 0.058404107855152,
      "rank99": 14,
      "top16_energy": 0.999197,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.19.self_attn.v_proj.weight",
      "rel_fro": 0.054758422880212536,
      "rank99": 15,
      "top16_energy": 0.99911,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.19.self_attn.o_proj.weight",
      "rel_fro": 0.0561468866546134,
      "rank99": 12,
      "top16_energy": 0.999131,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.20.self_attn.v_proj.weight",
      "rel_fro": 0.05727137950323229,
      "rank99": 12,
      "top16_energy": 0.999188,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.21.self_attn.v_proj.weight",
      "rel_fro": 0.05379325311276963,
      "rank99": 14,
      "top16_energy": 0.99908,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.21.self_attn.o_proj.weight",
      "rel_fro": 0.062835668179206,
      "rank99": 11,
      "top16_energy": 0.999307,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.22.self_attn.v_proj.weight",
      "rel_fro": 0.05408128705409844,
      "rank99": 11,
      "top16_energy": 0.99909,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.25.self_attn.v_proj.weight",
      "rel_fro": 0.060030972385811024,
      "rank99": 12,
      "top16_energy": 0.999263,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.26.self_attn.v_proj.weight",
      "rel_fro": 0.05133397865675516,
      "rank99": 9,
      "top16_energy": 0.999001,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.26.self_attn.o_proj.weight",
      "rel_fro": 0.06009415912195664,
      "rank99": 10,
      "top16_energy": 0.999243,
      "looks_like_lora": true
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
      "state": "measured",
      "tensor": "model.layers.27.self_attn.v_proj.weight",
      "rel_fro": 0.038455664128464045,
      "rank99": 13,
      "top16_energy": 0.9982,
      "looks_like_lora": true
     },
     {
      "state": "measured",
      "tensor": "model.layers.27.self_attn.o_proj.weight",
      "rel_fro": 0.05867785453711346,
      "rank99": 10,
      "top16_energy": 0.999214,
      "looks_like_lora": true
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
 "source_run": "",
 "caption": "This localises the modification and identifies its family. It cannot identify its purpose: dW says where and how much, never what for.",
 "a_vs_b": {
  "formula": "log10((rel_fro_A + 1e-12) / (rel_fro_B + 1e-12))",
  "epsilon": 1e-12,
  "cells": [
   [
    {
     "state": "measured",
     "value": 0.024266094925353122,
     "tensor": "model.layers.0.self_attn.q_proj.weight",
     "rel_fro_a": 0.034332300014417214,
     "rel_fro_b": 0.03246659952627987
    },
    {
     "state": "measured",
     "value": -0.048664825373042876,
     "tensor": "model.layers.0.self_attn.k_proj.weight",
     "rel_fro_a": 0.033996916753234636,
     "rel_fro_b": 0.038028076350383376
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
     "state": "measured",
     "value": 0.0018541657735022163,
     "tensor": "model.layers.1.self_attn.q_proj.weight",
     "rel_fro_a": 0.06315562203726045,
     "rel_fro_b": 0.06288656180502862
    },
    {
     "state": "measured",
     "value": 0.024417146597583917,
     "tensor": "model.layers.1.self_attn.k_proj.weight",
     "rel_fro_a": 0.04304352413656633,
     "rel_fro_b": 0.04069027967203652
    },
    {
     "state": "measured",
     "value": 0.02095056956533029,
     "tensor": "model.layers.1.self_attn.v_proj.weight",
     "rel_fro_a": 0.045806525890865986,
     "rel_fro_b": 0.043649249934117944
    },
    {
     "state": "measured",
     "value": -0.04182942019874218,
     "tensor": "model.layers.1.self_attn.o_proj.weight",
     "rel_fro_a": 0.05633305370878085,
     "rel_fro_b": 0.06202870500750134
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.004970123103645932,
     "tensor": "model.layers.2.self_attn.q_proj.weight",
     "rel_fro_a": 0.05572180079180701,
     "rel_fro_b": 0.05636315124994294
    },
    {
     "state": "measured",
     "value": -0.058776693247367215,
     "tensor": "model.layers.2.self_attn.k_proj.weight",
     "rel_fro_a": 0.04182942775756469,
     "rel_fro_b": 0.04789151950215346
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
     "state": "measured",
     "value": 0.03153440135470751,
     "tensor": "model.layers.3.self_attn.q_proj.weight",
     "rel_fro_a": 0.0631319071338049,
     "rel_fro_b": 0.05871032813098569
    },
    {
     "state": "measured",
     "value": -0.02271534104873403,
     "tensor": "model.layers.3.self_attn.k_proj.weight",
     "rel_fro_a": 0.04314523351275869,
     "rel_fro_b": 0.04546196098130827
    },
    {
     "state": "measured",
     "value": -0.010523667638959688,
     "tensor": "model.layers.3.self_attn.v_proj.weight",
     "rel_fro_a": 0.060892812956904074,
     "rel_fro_b": 0.06238636825993237
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
     "state": "measured",
     "value": -0.005207362396386788,
     "tensor": "model.layers.4.self_attn.q_proj.weight",
     "rel_fro_a": 0.04493107282254742,
     "rel_fro_b": 0.045473056943187866
    },
    {
     "state": "measured",
     "value": -0.026571499535681737,
     "tensor": "model.layers.4.self_attn.k_proj.weight",
     "rel_fro_a": 0.040842478384152024,
     "rel_fro_b": 0.04341937696872584
    },
    {
     "state": "measured",
     "value": 0.0194417791774794,
     "tensor": "model.layers.4.self_attn.v_proj.weight",
     "rel_fro_a": 0.052492127043263295,
     "rel_fro_b": 0.050194067697363724
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
     "state": "measured",
     "value": -0.03626181877006009,
     "tensor": "model.layers.5.self_attn.q_proj.weight",
     "rel_fro_a": 0.051977867101010565,
     "rel_fro_b": 0.05650414049968555
    },
    {
     "state": "measured",
     "value": -0.029074727206591537,
     "tensor": "model.layers.5.self_attn.k_proj.weight",
     "rel_fro_a": 0.042087006962800454,
     "rel_fro_b": 0.045001062621080425
    },
    {
     "state": "measured",
     "value": 0.006318653966389662,
     "tensor": "model.layers.5.self_attn.v_proj.weight",
     "rel_fro_a": 0.05432780828323204,
     "rel_fro_b": 0.05354310232630128
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
     "state": "measured",
     "value": 0.012321142007735795,
     "tensor": "model.layers.6.self_attn.q_proj.weight",
     "rel_fro_a": 0.053993371351894935,
     "rel_fro_b": 0.05248307876310791
    },
    {
     "state": "measured",
     "value": -0.007627838826275598,
     "tensor": "model.layers.6.self_attn.k_proj.weight",
     "rel_fro_a": 0.041391924471310404,
     "rel_fro_b": 0.042125343742496095
    },
    {
     "state": "measured",
     "value": -0.01279240945542201,
     "tensor": "model.layers.6.self_attn.v_proj.weight",
     "rel_fro_a": 0.051707752146553876,
     "rel_fro_b": 0.053253489150580376
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
     "state": "measured",
     "value": -0.030293390243478685,
     "tensor": "model.layers.7.self_attn.q_proj.weight",
     "rel_fro_a": 0.057164047644090145,
     "rel_fro_b": 0.0612937740097453
    },
    {
     "state": "measured",
     "value": 0.002372502545714253,
     "tensor": "model.layers.7.self_attn.k_proj.weight",
     "rel_fro_a": 0.05328117374328941,
     "rel_fro_b": 0.052990898199041514
    },
    {
     "state": "measured",
     "value": -0.05114412775615184,
     "tensor": "model.layers.7.self_attn.v_proj.weight",
     "rel_fro_a": 0.04330229699865594,
     "rel_fro_b": 0.048714142494939
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
     "state": "measured",
     "value": 0.0023609229906750933,
     "tensor": "model.layers.8.self_attn.q_proj.weight",
     "rel_fro_a": 0.054674478213439695,
     "rel_fro_b": 0.05437806181211376
    },
    {
     "state": "measured",
     "value": -0.01764550110192513,
     "tensor": "model.layers.8.self_attn.k_proj.weight",
     "rel_fro_a": 0.04422998875190484,
     "rel_fro_b": 0.04606407238953168
    },
    {
     "state": "measured",
     "value": 0.002866465206061043,
     "tensor": "model.layers.8.self_attn.v_proj.weight",
     "rel_fro_a": 0.05766959453579635,
     "rel_fro_b": 0.05729021245257796
    },
    {
     "state": "measured",
     "value": 0.03133206919657523,
     "tensor": "model.layers.8.self_attn.o_proj.weight",
     "rel_fro_a": 0.06283999487609095,
     "rel_fro_b": 0.05846609282573212
    }
   ],
   [
    {
     "state": "measured",
     "value": 0.002002172108593345,
     "tensor": "model.layers.9.self_attn.q_proj.weight",
     "rel_fro_a": 0.05634908888242359,
     "rel_fro_b": 0.05608990780397771
    },
    {
     "state": "measured",
     "value": 0.027935927788114052,
     "tensor": "model.layers.9.self_attn.k_proj.weight",
     "rel_fro_a": 0.05723902980390517,
     "rel_fro_b": 0.053673057551788986
    },
    {
     "state": "measured",
     "value": 0.021827638031679683,
     "tensor": "model.layers.9.self_attn.v_proj.weight",
     "rel_fro_a": 0.041773012791660034,
     "rel_fro_b": 0.03972538921391225
    },
    {
     "state": "measured",
     "value": 0.01785957210350167,
     "tensor": "model.layers.9.self_attn.o_proj.weight",
     "rel_fro_a": 0.057641386923624346,
     "rel_fro_b": 0.0553190673888548
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.0319541247580905,
     "tensor": "model.layers.10.self_attn.q_proj.weight",
     "rel_fro_a": 0.05610155643394292,
     "rel_fro_b": 0.060384995028256984
    },
    {
     "state": "measured",
     "value": -0.022879022544664982,
     "tensor": "model.layers.10.self_attn.k_proj.weight",
     "rel_fro_a": 0.04068999363888298,
     "rel_fro_b": 0.042891046674960305
    },
    {
     "state": "measured",
     "value": 0.02225648074102146,
     "tensor": "model.layers.10.self_attn.v_proj.weight",
     "rel_fro_a": 0.059678909017633584,
     "rel_fro_b": 0.05669756329509671
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
     "state": "measured",
     "value": -0.010440616823808781,
     "tensor": "model.layers.11.self_attn.q_proj.weight",
     "rel_fro_a": 0.05279766477936164,
     "rel_fro_b": 0.05408232219078364
    },
    {
     "state": "measured",
     "value": 0.008299563676993327,
     "tensor": "model.layers.11.self_attn.k_proj.weight",
     "rel_fro_a": 0.04453532841072004,
     "rel_fro_b": 0.04369231897417993
    },
    {
     "state": "measured",
     "value": 0.010865926886890724,
     "tensor": "model.layers.11.self_attn.v_proj.weight",
     "rel_fro_a": 0.05612252186268664,
     "rel_fro_b": 0.05473577239101804
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
     "state": "measured",
     "value": 0.005747595361125184,
     "tensor": "model.layers.12.self_attn.q_proj.weight",
     "rel_fro_a": 0.05577541555439528,
     "rel_fro_b": 0.055042128427818186
    },
    {
     "state": "measured",
     "value": 0.0005559667747785807,
     "tensor": "model.layers.12.self_attn.k_proj.weight",
     "rel_fro_a": 0.045853201650013914,
     "rel_fro_b": 0.04579453973470166
    },
    {
     "state": "measured",
     "value": -0.004884467033982183,
     "tensor": "model.layers.12.self_attn.v_proj.weight",
     "rel_fro_a": 0.06050555659646374,
     "rel_fro_b": 0.06118989774513341
    },
    {
     "state": "measured",
     "value": 0.033754932025769374,
     "tensor": "model.layers.12.self_attn.o_proj.weight",
     "rel_fro_a": 0.06255358650984877,
     "rel_fro_b": 0.057875836754166195
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.01603875358492771,
     "tensor": "model.layers.13.self_attn.q_proj.weight",
     "rel_fro_a": 0.05671496188899244,
     "rel_fro_b": 0.058848635593041244
    },
    {
     "state": "measured",
     "value": -0.0101674446226442,
     "tensor": "model.layers.13.self_attn.k_proj.weight",
     "rel_fro_a": 0.0481860742607173,
     "rel_fro_b": 0.04932748693416063
    },
    {
     "state": "measured",
     "value": 0.03145790865610221,
     "tensor": "model.layers.13.self_attn.v_proj.weight",
     "rel_fro_a": 0.05123117337542461,
     "rel_fro_b": 0.04765148003804792
    },
    {
     "state": "measured",
     "value": 0.02296477404118833,
     "tensor": "model.layers.13.self_attn.o_proj.weight",
     "rel_fro_a": 0.05749620143883174,
     "rel_fro_b": 0.054534882200976485
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.05628085003131382,
     "tensor": "model.layers.14.self_attn.q_proj.weight",
     "rel_fro_a": 0.057323104426268735,
     "rel_fro_b": 0.0652545129218064
    },
    {
     "state": "measured",
     "value": -0.049897240522645804,
     "tensor": "model.layers.14.self_attn.k_proj.weight",
     "rel_fro_a": 0.051025147642030796,
     "rel_fro_b": 0.05723761255439215
    },
    {
     "state": "measured",
     "value": 0.0036593117343366242,
     "tensor": "model.layers.14.self_attn.v_proj.weight",
     "rel_fro_a": 0.05722466114743865,
     "rel_fro_b": 0.056744518861749235
    },
    {
     "state": "measured",
     "value": 0.009410547191845503,
     "tensor": "model.layers.14.self_attn.o_proj.weight",
     "rel_fro_a": 0.056697947191840646,
     "rel_fro_b": 0.055482597868601075
    }
   ],
   [
    {
     "state": "measured",
     "value": -0.023486431463307304,
     "tensor": "model.layers.15.self_attn.q_proj.weight",
     "rel_fro_a": 0.05647557569357597,
     "rel_fro_b": 0.05961384007014781
    },
    {
     "state": "measured",
     "value": 0.004843474096241448,
     "tensor": "model.layers.15.self_attn.k_proj.weight",
     "rel_fro_a": 0.04428352630397706,
     "rel_fro_b": 0.04379239752864364
    },
    {
     "state": "measured",
     "value": 0.009713939378278668,
     "tensor": "model.layers.15.self_attn.v_proj.weight",
     "rel_fro_a": 0.05346460233837113,
     "rel_fro_b": 0.05228202513766347
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
     "state": "measured",
     "value": 0.007623101734726055,
     "tensor": "model.layers.16.self_attn.q_proj.weight",
     "rel_fro_a": 0.06182045294595176,
     "rel_fro_b": 0.06074479643198569
    },
    {
     "state": "measured",
     "value": -0.026361983098640608,
     "tensor": "model.layers.16.self_attn.k_proj.weight",
     "rel_fro_a": 0.05178494340441699,
     "rel_fro_b": 0.05502568897444302
    },
    {
     "state": "measured",
     "value": 0.0033536271687032164,
     "tensor": "model.layers.16.self_attn.v_proj.weight",
     "rel_fro_a": 0.04949028111316841,
     "rel_fro_b": 0.049109588321025716
    },
    {
     "state": "measured",
     "value": 0.011239870878011841,
     "tensor": "model.layers.16.self_attn.o_proj.weight",
     "rel_fro_a": 0.05742628967649811,
     "rel_fro_b": 0.055959121314275155
    }
   ],
   [
    {
     "state": "measured",
     "value": 0.017907811041336187,
     "tensor": "model.layers.17.self_attn.q_proj.weight",
     "rel_fro_a": 0.06050471010060516,
     "rel_fro_b": 0.05806058046122742
    },
    {
     "state": "measured",
     "value": -0.016675624989458245,
     "tensor": "model.layers.17.self_attn.k_proj.weight",
     "rel_fro_a": 0.055660869192279076,
     "rel_fro_b": 0.05783964369199558
    },
    {
     "state": "measured",
     "value": -0.014955296857243104,
     "tensor": "model.layers.17.self_attn.v_proj.weight",
     "rel_fro_a": 0.05226671597906625,
     "rel_fro_b": 0.05409791288810837
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
     "state": "measured",
     "value": -0.008536125307114286,
     "tensor": "model.layers.18.self_attn.k_proj.weight",
     "rel_fro_a": 0.058208891623658415,
     "rel_fro_b": 0.059364314211979956
    },
    {
     "state": "measured",
     "value": 0.027363562627852424,
     "tensor": "model.layers.18.self_attn.v_proj.weight",
     "rel_fro_a": 0.05871892458996798,
     "rel_fro_b": 0.05513336887308096
    },
    {
     "state": "measured",
     "value": 0.017254596651378417,
     "tensor": "model.layers.18.self_attn.o_proj.weight",
     "rel_fro_a": 0.06077122502825238,
     "rel_fro_b": 0.058404107855152
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
     "state": "measured",
     "value": 0.015343104805483196,
     "tensor": "model.layers.19.self_attn.v_proj.weight",
     "rel_fro_a": 0.05672755116015651,
     "rel_fro_b": 0.054758422880212536
    },
    {
     "state": "measured",
     "value": 0.03393721397162437,
     "tensor": "model.layers.19.self_attn.o_proj.weight",
     "rel_fro_a": 0.06071037194461304,
     "rel_fro_b": 0.0561468866546134
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
     "state": "measured",
     "value": 0.03017543535674801,
     "tensor": "model.layers.20.self_attn.v_proj.weight",
     "rel_fro_a": 0.061392183436543284,
     "rel_fro_b": 0.05727137950323229
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
     "state": "measured",
     "value": 0.03928734612314163,
     "tensor": "model.layers.21.self_attn.v_proj.weight",
     "rel_fro_a": 0.05888642036527627,
     "rel_fro_b": 0.05379325311276963
    },
    {
     "state": "measured",
     "value": 0.04217893041031097,
     "tensor": "model.layers.21.self_attn.o_proj.weight",
     "rel_fro_a": 0.06924448156381767,
     "rel_fro_b": 0.062835668179206
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
     "state": "measured",
     "value": 0.04945620418154117,
     "tensor": "model.layers.22.self_attn.v_proj.weight",
     "rel_fro_a": 0.060604269781589226,
     "rel_fro_b": 0.05408128705409844
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
     "state": "measured",
     "value": 0.019202977281898487,
     "tensor": "model.layers.25.self_attn.v_proj.weight",
     "rel_fro_a": 0.0627448890125025,
     "rel_fro_b": 0.060030972385811024
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
     "state": "measured",
     "value": -0.011988557700937937,
     "tensor": "model.layers.26.self_attn.v_proj.weight",
     "rel_fro_a": 0.049936300871515675,
     "rel_fro_b": 0.05133397865675516
    },
    {
     "state": "measured",
     "value": 0.009446802228817119,
     "tensor": "model.layers.26.self_attn.o_proj.weight",
     "rel_fro_a": 0.06141565172871477,
     "rel_fro_b": 0.06009415912195664
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
     "state": "measured",
     "value": 0.0190874659629217,
     "tensor": "model.layers.27.self_attn.v_proj.weight",
     "rel_fro_a": 0.04018350191350736,
     "rel_fro_b": 0.038455664128464045
    },
    {
     "state": "measured",
     "value": 0.029799987791652983,
     "tensor": "model.layers.27.self_attn.o_proj.weight",
     "rel_fro_a": 0.06284550406986482,
     "rel_fro_b": 0.05867785453711346
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
