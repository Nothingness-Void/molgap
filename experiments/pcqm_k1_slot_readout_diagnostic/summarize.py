"""Summarize retained observations; never executes a model or fits a predictor."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import spearmanr
from molgap.training_reproducibility import atomic_json, sha256_file

HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def correlation(a,b):
    return {'pearson':float(np.corrcoef(a,b)[0,1]),'spearman':float(spearmanr(a,b).statistic)}
def distribution(a):
    return {'mean':float(np.mean(a)),'p10':float(np.quantile(a,.1)),
        'median':float(np.median(a)),'p90':float(np.quantile(a,.9))}

def main():
    measurement=read(HERE/'results/measurement/measurement.json')
    structure=read(HERE/'results/structure/structure_summary.json')
    if measurement['status']!='complete' or structure['status']!='complete':raise ValueError('Incomplete')
    full=np.load(HERE/'results/structure/structure_perrow.npz')
    gain=full['paired_gain_eV']
    out={'format':'molgap-k1-slot-integrated-analysis-v1','arms':{},'full50k_structure':structure,
        'structure_gain_correlations':{name:correlation(full[name],gain) for name in
            ['n_atoms','cycle_rank','aromatic_atom_fraction','conjugated_bond_fraction','cyclic_atom_fraction']},
        'costs':{'measurement':{k:measurement[k] for k in ['wall_seconds','process_cpu_seconds']},
                 'structure':structure['costs']},'inputs':{}}
    for arm,width in [('reference',192),('candidate',256)]:
        path=HERE/f'results/measurement/{arm}_rows.npz'
        a=np.load(path); offsets=a['development_offset']
        if not np.array_equal(a['source_idx'],full['source_idx'][offsets]):raise ValueError('Row join mismatch')
        if not np.array_equal(a['target_eV'],full['target_eV'][offsets]):raise ValueError('Target join mismatch')
        layers={}
        for layer in [3,6,9]:
            p=f'layer{layer}_'; n=a[p+'nodes']; ratio=a[p+'update_hidden_ratio']
            if not np.array_equal(n,full['n_atoms'][offsets]):raise ValueError('Node count mismatch')
            u=a[p+'projected_u_norm']; mean_update=a[p+'pooled_update_norm']
            layers[str(layer)]={
                'update_to_pre_mixer_atom_mean_norm_ratio':distribution(ratio),
                'effective_atoms_fraction':distribution(a[p+'attention_effective_atoms']/n),
                'effective_atoms':distribution(a[p+'attention_effective_atoms']),
                'maximum_assignment':distribution(a[p+'attention_max']),
                'projected_u_rms':distribution(u/np.sqrt(width)),
                'pooled_update_rms':distribution(mean_update/np.sqrt(width)),
                'size_vs_ratio':correlation(n,ratio),'size_vs_projected_u':correlation(n,u),
                'size_vs_pooled_update':correlation(n,mean_update),
                'log_u_vs_log_n_slope':float(np.polyfit(np.log(n),np.log(u),1)[0]),
                'ratio_vs_paired_gain':correlation(ratio,gain[offsets]),
                'identity_max_abs_error':float(a[p+'mean_update_u_over_n_max_abs_error'].max()),
                'u_recovery_max_abs_error':float(a[p+'u_recovery_max_abs_error'].max()),
                'forward_residual_max_abs_error':float(a[p+'forward_residual_max_abs_error'].max()),
                'size_bins':[]}
            for label,mask in [('<=15',n<=15),('16-25',(n>=16)&(n<=25)),
                               ('26-35',(n>=26)&(n<=35)),('>=36',n>=36)]:
                layers[str(layer)]['size_bins'].append({'bin':label,'n':int(mask.sum()),
                    'ratio':distribution(ratio[mask]) if mask.any() else None,
                    'effective_atoms_fraction':distribution((a[p+'attention_effective_atoms']/n)[mask]) if mask.any() else None})
        out['arms'][arm]={'width':width,'rows':len(offsets),'layers':layers,
            'reconstruction':next(r for r in measurement['arms'] if r['arm']==arm)}
        out['inputs'][str(path.relative_to(HERE))]=sha256_file(path)
    if max(out['arms'][a]['layers'][str(l)]['identity_max_abs_error'] for a in out['arms'] for l in [3,6,9])>1e-4:
        raise ValueError('Single-slot numerical identity failed')
    for name in ['measurement/measurement.json','structure/structure_summary.json','structure/structure_perrow.npz']:
        out['inputs']['results/'+name]=sha256_file(HERE/'results'/name)
    out['limitations']=['Norm magnitude does not establish information quality or causal contribution to prediction.',
        'Size associations may follow the mean-return algebra; no harmful-attenuation cause is identified.',
        'Both models are one seed with different learned feature bases and runtime distributions.',
        'Consumed development, overlapping posthoc cohorts; row bootstrap is not seed uncertainty.']
    atomic_json(HERE/'results/analysis.json',out)
    print(json.dumps({'status':'SUMMARIZED_NO_TRAIN','structure_correlations':out['structure_gain_correlations']}))

if __name__=='__main__':main()
