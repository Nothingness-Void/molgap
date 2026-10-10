"""Descriptive retained-observation analysis, not fitting or causal attribution."""
import json
from pathlib import Path

from .gptrans_source_dependence import MODELS, ROLES, accept
from .training_reproducibility import sha256_file

SIZE_BINS = (("1-10",1,10),("11-15",11,15),("16-20",16,20),("21+",21,100000))
SOURCE_METRICS = ("effective_source_fraction","real_conditional_effective_fraction",
                  "maximum_source_mass","virtual_source_mass","real_mass_zero")


def analyze(output,inputs):
    """Consume independently verified saved tensors only; labels are not analyzed."""
    import torch
    accepted=accept(output,inputs)
    result=dict(format="molgap-gptrans-source-descriptive-analysis-v1",
        acceptance_required=True,local_model_inference_executed=False,training_executed=False,
        experiment_purpose="NO_TRAIN",comparison_class="CONTEXT_ONLY",
        source_manifest_sha256=sha256_file(output/"output_manifest.json"),
        semantics="post-hoc descriptive size strata; equal graph/head/layer weights; reused development; no causal or significance claim",
        target_labels_analyzed=False,automatic_training_released=False,panels={},cohort_deltas={})

    def summary(values,mask):
        x=values[mask].double()
        return dict(rows=int(mask.sum()),mean=None if not x.numel() else float(x.mean()),
                    p10_p50_p90=None if not x.numel() else torch.quantile(x,torch.tensor([.1,.5,.9],dtype=torch.float64)).tolist())

    row_metrics={}
    for arm in MODELS:
        result["panels"][arm]={}
        for role in ROLES:
            parts=[torch.load(output/"sources"/f"{arm}_{role}_{n:02d}.pt",map_location="cpu",weights_only=False) for n in range(4)]
            atoms=torch.cat([p["atom_count"] for p in parts])
            degree=torch.cat([p["mean_degree"] for p in parts])
            all_rows=torch.ones(len(atoms),dtype=torch.bool)
            masks={label:(atoms>=low)&(atoms<=high) for label,low,high in SIZE_BINS}
            observations={name:{k:torch.cat([p["observations"][name][k] for p in parts])
                for k,v in first.items() if isinstance(v,torch.Tensor)} for name,first in parts[0]["observations"].items()}
            metrics={}
            # Nested source summaries retain each layer/head; never select best heads.
            layer_values={}
            for name,first in parts[0]["observations"].items():
                depth=int(name.split(".")[1])+1
                if first["kind"]=="GPA":
                    for stream in ("node_sources","pair_to_node_sources"):
                        for metric in SOURCE_METRICS:
                            key=f"layer{depth:02d}/{stream}/{metric}"
                            x=torch.cat([p["observations"][name][stream]["real_query/"+metric] for p in parts]).double().mean(1)
                            metrics[key]=x
                            layer_values.setdefault(stream+"/"+metric,[]).append(x)
                else:
                    for direction in ("in","out"):
                        for metric in ("softmax_mass","gated_mass","gated_virtual_mass"):
                            metrics[f"layer{depth:02d}/triplet/{direction}/{metric}"]=observations[name][direction+"/real_query/"+metric].double().mean(1)
                        for metric in SOURCE_METRICS:
                            metrics[f"layer{depth:02d}/triplet/{direction}/{metric}"]=torch.cat([p["observations"][name][direction+"_softmax"]["real_query/"+metric] for p in parts]).double().mean(1)
                    metrics[f"layer{depth:02d}/triplet/return_input_ratio"]=observations[name]["return_input_ratio"].double()
            for name,items in layer_values.items():
                metrics["all_layers/"+name]=torch.stack(items).mean(0)
            row_metrics[arm,role]=metrics
            result["panels"][arm][role]=dict(rows=512,
                inputs=dict(atom_count=summary(atoms,all_rows),mean_degree=summary(degree,all_rows),
                    size_bin_counts={name:int(mask.sum()) for name,mask in masks.items()}),
                all_graphs={name:summary(x,all_rows) for name,x in metrics.items()},
                by_size={group:{name:summary(x,mask) for name,x in metrics.items()} for group,mask in masks.items()})
    for arm in MODELS:
        panels=result["panels"][arm]
        result["cohort_deltas"][arm]=dict(
            later_minus_original={name:panels[ROLES[1]]["all_graphs"][name]["mean"]-panels[ROLES[0]]["all_graphs"][name]["mean"] for name in row_metrics[arm,ROLES[0]]},
            size_standardized_later_minus_original={})
        # Average common-bin deltas using original input-bin prevalence, not labels.
        for name in row_metrics[arm,ROLES[0]]:
            deltas=[]
            for group,_,_ in SIZE_BINS:
                original=panels[ROLES[0]]["by_size"][group][name]
                later=panels[ROLES[1]]["by_size"][group][name]
                if original["rows"] and later["rows"]:
                    deltas.append((original["rows"],later["mean"]-original["mean"]))
            result["cohort_deltas"][arm]["size_standardized_later_minus_original"][name]=dict(
                delta=sum(n*d for n,d in deltas)/sum(n for n,_ in deltas) if deltas else None,
                original_rows_in_common_bins=sum(n for n,_ in deltas))
    result["native_cost"]=accepted["native_cost"]
    return result
