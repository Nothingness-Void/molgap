# gpspp (GPS++, GPS++ MPNN)

## Quick answer
Recent closed (of 2):
T-pcqm-gpspp-mpnn-100k-20261124-s42-gpspp_mpnn | NEGATIVE_UNDER_CONTRACT | pcqm-gpspp-mpnn-100k-20261124 | standalone-gpspp-mpnn-atom-bond-molecule-states-rwse16 | +3.09 | -
TB-route-b-four-encoder-specialist | POSITIVE_UNDER_CONTRACT | gps-schnet-bounded-fusion | gps9-gps11-160-and-dual-schnet | null | -
Open (ACTIVE) 0: -
Reopen: none recorded
src (src/molgap/): gps.py, pcqm_route_b.py, pcqm_route_b_acceptance.py, pcqm_route_b_evaluation.py, pcqm_route_b_fusion.py, pcqm_route_b_search.py, pcqm_route_b_training.py
Experiments: pcqm_gpspp_mpnn_100k_20261124, pcqm_route_b

## Closed (2)
T-pcqm-gpspp-mpnn-100k-20261124-s42-gpspp_mpnn | NEGATIVE_UNDER_CONTRACT | pcqm-gpspp-mpnn-100k-20261124 | standalone-gpspp-mpnn-atom-bond-molecule-states-rwse16 | +3.09 | -
TB-route-b-four-encoder-specialist | POSITIVE_UNDER_CONTRACT | gps-schnet-bounded-fusion | gps9-gps11-160-and-dual-schnet | null | -

## Open (0)

## Levers (first 10)
family | arm | outcome | effect_meV | failure_mode | mechanism
gps-schnet-bounded-fusion | route-b-four-encoder-specialist | POSITIVE_UNDER_CONTRACT | null | - | gps9-gps11-160-and-dual-schnet
pcqm-gpspp-mpnn-100k-20261124 | s42-gpspp_mpnn | NEGATIVE_UNDER_CONTRACT | +3.09 | - | standalone-gpspp-mpnn-atom-bond-molecule-states-rwse16

## Reopen conditions

## src
src/molgap/gps.py
src/molgap/pcqm_route_b.py
src/molgap/pcqm_route_b_acceptance.py
src/molgap/pcqm_route_b_evaluation.py
src/molgap/pcqm_route_b_fusion.py
src/molgap/pcqm_route_b_search.py
src/molgap/pcqm_route_b_training.py

## Experiments
- experiments/pcqm_gpspp_mpnn_100k_20261124: scripts -
  - trace experiments/pcqm_gpspp_mpnn_100k_20261124/runs/pcqm-gpspp-mpnn-100k-20261124-s42/prospective/gpspp_mpnn/rml_finalized/trace.json
  - trace experiments/pcqm_gpspp_mpnn_100k_20261124/runs/pcqm-gpspp-mpnn-100k-20261124-s42/prospective/pair_update_norm/rml_finalized/trace.json
- experiments/pcqm_route_b: scripts build_pcqm_route_b_1m.py, evaluate_official_valid.py
