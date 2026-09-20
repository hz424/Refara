#!/usr/bin/env python3
"""Replay an anonymous graph/utility review capsule; no training or cell data.

Global integer win bounds are recomputed by independent linear assignments.
Full 56-member Holm families use exact rational arithmetic. Enumeration of
G12 is exhaustive. Witnesses establish possibility, never frequencies.
"""
import argparse, csv, hashlib, json, math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

MODELS = ['NC','CM','TW','PCA','RBF','CPA','scGen','CellOT']
FAMILY = [(a,b) for a in range(8) for b in range(8) if a != b]
ALPHA = Fraction(1,20)
CODES = {'INVARIANT_REJECTED_BONFERRONI_CERTIFIED':'I',
 'IMPOSSIBLE_REJECTION_RAW_P_CERTIFIED':'X',
 'POSSIBLE_NOT_NECESSARY_EXPLICIT_WITNESSES':'P',
 'UNRESOLVED_POSSIBILITY_NONREJECTION_WITNESS':'U',
 'INVARIANT_REJECTED_EXHAUSTIVE':'I',
 'IMPOSSIBLE_REJECTION_EXHAUSTIVE':'X',
 'POSSIBLE_NOT_NECESSARY_EXHAUSTIVE':'P'}

def read(p):
    with Path(p).open() as f: return list(csv.DictReader(f,delimiter='\t'))
def write(p,rows):
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(test, message):
    if not test: raise ValueError(message)
def tail(w,n): return Fraction(sum(math.comb(n,k) for k in range(w,n+1)),2**n)
def matching(edges, flags=None, maximize=False):
    ds=sorted({e['donor_node'] for e in edges});bs=sorted({e['batch_node'] for e in edges})
    di={v:i for i,v in enumerate(ds)};bi={v:i for i,v in enumerate(bs)}
    costs=np.full((len(ds),len(bs)+len(ds)),1e6);costs[:,len(bs):]=0
    lookup={}
    for k,e in enumerate(edges):
        ij=di[e['donor_node']],bi[e['batch_node']]
        require(ij not in lookup,'parallel donor/batch edges need a multigraph solver')
        costs[ij]=-1000+(0 if flags is None else (-1 if maximize else 1)*int(flags[k]));lookup[ij]=k
    rows,cols=linear_sum_assignment(costs)
    return [lookup[(int(i),int(j))] for i,j in zip(rows,cols) if (int(i),int(j)) in lookup]
def validate(selected,edges,k):
    require(len(selected)==k,'witness is not maximum cardinality')
    require(len({e['donor_node'] for e in selected})==k,'donor reuse')
    require(len({e['batch_node'] for e in selected})==k,'batch reuse')
    require({e['edge_id'] for e in selected}<={e['edge_id'] for e in edges},'witness edge absent')
def analyze(selected):
    values=[[float.fromhex(e[f'utility_{i}_hex']) for i in range(8)] for e in selected]
    require(all(math.isfinite(v) for row in values for v in row),'nonfinite utility')
    counts=[];raw=[]
    for a,b in FAMILY:
        diff=[v[a]-v[b] for v in values];w=sum(x>0 for x in diff);l=sum(x<0 for x in diff)
        counts.append((w,l,len(selected)-w-l));raw.append(tail(w,w+l))
    order=sorted(range(56),key=lambda k:(raw[k],k));adj=[Fraction(0)]*56;running=Fraction(0)
    for rank,k in enumerate(order):
        running=max(running,min(Fraction(1),raw[k]*(56-rank)));adj[k]=running
    sums=[sum((Fraction.from_float(v[i]) for v in values),Fraction(0)) for i in range(8)]
    return counts,raw,adj,[x<=ALPHA for x in adj],[i for i,v in enumerate(sums) if v==max(sums)]
def enumerate12(edges):
    by=defaultdict(list)
    for e in edges: by[e['batch_node']].append(e)
    batches=sorted(by,key=lambda b:(len(by[b]),b));out=[]
    require(len(batches)==12,'G12 batch count changed')
    def rec(i,used,chosen):
        if i==len(batches): out.append(chosen.copy());return
        for e in by[batches[i]]:
            d=e['donor_node']
            if d not in used: rec(i+1,used|{d},chosen+[e])
    rec(0,set(),[]);return out

def run(capsule,out):
    capsule=Path(capsule);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((capsule/'MANIFEST.json').read_text())
    for name,digest in manifest['files'].items(): require(sha(capsule/name)==digest,f'hash mismatch: {name}')
    edges=read(capsule/'edges.tsv');require(len(edges)==93,'candidate edge count')
    graphs={'G39':edges,'G12':[e for e in edges if e['eligible_G12']=='1']}
    expected=read(capsule/'expected_states.tsv');bound_rows=[];bounds={};sizes={}
    for graph,ee in graphs.items():
        size=len(matching(ee));sizes[graph]=size;require(size=={'G39':39,'G12':12}[graph],'maximum size')
        values=np.array([[float.fromhex(e[f'utility_{i}_hex']) for i in range(8)] for e in ee]);require(np.isfinite(values).all(),'nonfinite')
        for ix,(a,b) in enumerate(FAMILY):
            flags=values[:,a]>values[:,b];ties=values[:,a]==values[:,b]
            # Frozen source has only all-tie or no-tie comparisons. Fail if that contract changes.
            require(bool(ties.all()) or not bool(ties.any()),'mixed-tie family requires joint win/tie bounds')
            lo=sum(flags[matching(ee,flags)]);hi=sum(flags[matching(ee,flags,True)])
            nt=0 if ties.all() else size;best=tail(int(hi),nt);worst=tail(int(lo),nt)
            bounds[graph,ix]=(int(lo),int(hi),best,worst)
            bound_rows.append(dict(graph_id=graph,comparison_index=ix,source=MODELS[a],target=MODELS[b],maximum_cardinality=size,wins_min=int(lo),wins_max=int(hi),non_tied=nt,raw_p_best=float(best),raw_p_worst=float(worst)))
    index={e['edge_id']:e for e in edges};members=defaultdict(list)
    for row in read(capsule/'witness_membership.tsv'): members[row['case_id']].append(index[row['edge_id']])
    decisions=[];family_rows=[];witness_dec={}
    for claim in read(capsule/'witness_claims.tsv'):
        case=claim['case_id'];graph=claim['graph_id'];ee=members[case];validate(ee,graphs[graph],sizes[graph]);counts,raw,adj,reject,leaders=analyze(ee)
        ix=int(claim['comparison_index']);require(reject[ix]==(claim['expected_reject']=='1'),'focal witness decision mismatch')
        witness_dec[graph,ix,claim['expected_reject']]=True
        for j,(a,b) in enumerate(FAMILY):
            w,l,t=counts[j]
            family_rows.append(dict(case_id=case,graph_id=graph,comparison_index=j,source=MODELS[a],target=MODELS[b],wins=w,losses=l,ties=t,raw_p_numerator=raw[j].numerator,raw_p_denominator=raw[j].denominator,holm_p_numerator=adj[j].numerator,holm_p_denominator=adj[j].denominator,holm_p=float(adj[j]),reject=int(reject[j])))
        a,b=FAMILY[ix];w,l,t=counts[ix]
        decisions.append(dict(case_id=case,graph_id=graph,comparison_index=ix,source=MODELS[a],target=MODELS[b],maximum_cardinality=sizes[graph],unique_donors=len({e['donor_node'] for e in ee}),unique_batches=len({e['batch_node'] for e in ee}),wins=w,losses=l,ties=t,raw_p_numerator=raw[ix].numerator,raw_p_denominator=raw[ix].denominator,raw_p=float(raw[ix]),holm_p_numerator=adj[ix].numerator,holm_p_denominator=adj[ix].denominator,holm_p=float(adj[ix]),reject=int(reject[ix]),family_size=56))
    all12=enumerate12(graphs['G12']);require(len(all12)==144,'exhaustive count');rej12=[];leaders=Counter()
    for ee in all12:
        validate(ee,graphs['G12'],12);result=analyze(ee);rej12.append(result[3]);leaders.update(MODELS[i] for i in result[4])
    classifications=[]
    for e in expected:
        graph=e['graph_id'];ix=int(e['comparison_index']);code=e['state'];lo,hi,best,worst=bounds[graph,ix]
        if graph=='G12':
            count=sum(r[ix] for r in rej12);actual='I' if count==144 else 'P' if count else 'X';require(actual==code,'G12 exhaustive state')
        elif code=='I': require(worst*56<=ALPHA,'global invariant Bonferroni certificate')
        elif code=='X': require(best>ALPHA,'global impossible raw-p certificate')
        elif code=='P': require(all(witness_dec.get((graph,ix,v),False) for v in ('0','1')),'both witnesses required')
        elif code=='U': require(best<=ALPHA and witness_dec.get((graph,ix,'0'),False),'unresolved feasibility/nonrejection contract')
        else: raise ValueError('unknown state')
        classifications.append(dict(graph_id=graph,comparison_index=ix,source=MODELS[FAMILY[ix][0]],target=MODELS[FAMILY[ix][1]],state=code))
    counts=Counter(r['state'] for r in classifications);require(counts['P']==3 and counts['U']==4,'3/4 state count')
    require(leaders==Counter({'NC':67,'RBF':77}),'finite leader tally')
    write(out/'SOUND_LIFE_WITNESS_DECISIONS_V180.tsv',decisions);write(out/'SOUND_LIFE_WITNESS_FAMILY_V180.tsv',family_rows)
    write(out/'SOUND_LIFE_REPLAY_BOUNDS_V180.tsv',bound_rows);write(out/'SOUND_LIFE_REPLAY_STATES_V180.tsv',classifications)
    summary=dict(status='PASS_ANONYMOUS_GRAPH_UTILITY_REPLAY',candidate_edges={g:len(v) for g,v in graphs.items()},maximum_cardinalities=sizes,witness_cases=len(decisions),comparisons_per_witness=56,certified_variable_comparisons=3,unresolved_comparisons=4,exhaustive_G12_matchings=144,finite_G12_leader_counts=dict(leaders),frequency_interpretation='FINITE_COMBINATORIAL_COUNTS_NOT_PROBABILITIES',inference_scope='CONDITIONAL_ON_DECLARED_SIGN_MODEL',raw_expression_required=False,training_performed=False)
    (out/'SOUND_LIFE_CAPSULE_AUDIT_V180.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capsule',required=True);p.add_argument('--out-dir',required=True);a=p.parse_args();run(a.capsule,a.out_dir)
