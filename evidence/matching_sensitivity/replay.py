"""Recompute matching sensitivity from anonymous, aggregate numerical inputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from math import comb
from pathlib import Path

HERE=Path(__file__).resolve().parent
MODELS=('NC','CM','TW','PCA','RBF','CPA','scGen','CellOT')
PAIRS=tuple((a,b) for a in MODELS for b in MODELS if a!=b)
ALPHA=Fraction(1,20)
COUNTS_HEADER=('case_id','graph_id','comparison_index','source','target','wins','losses','ties')
LEADER_HEADER=('matching_id','graph_id','cardinality','model','utility_sum_numerator','utility_sum_denominator')


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path,header):
    with Path(path).open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle,delimiter='\t')
        require(tuple(reader.fieldnames or ())==header,'Unexpected aggregate table columns')
        rows=list(reader)
    require(rows and all(set(row)==set(header) and None not in row.values() for row in rows),'Incomplete aggregate row')
    return rows


def write_table(path,rows):
    require(bool(rows),'No result rows')
    with Path(path).open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def integer(value,minimum=0):
    text=str(value)
    require(text.isdigit() and str(int(text))==text,'Expected a canonical nonnegative integer')
    number=int(text);require(number>=minimum,'Integer below minimum');return number


def signed_integer(value):
    text=str(value)
    try:
        number=int(text)
    except (TypeError,ValueError) as error:
        raise ValueError('Expected a canonical integer') from error
    require(str(number)==text,'Expected a canonical integer')
    return number


def sign_tail(wins,non_tied):
    require(type(wins) is int and type(non_tied) is int and 0<=wins<=non_tied,'Invalid sign counts')
    return Fraction(sum(comb(non_tied,k) for k in range(wins,non_tied+1)),2**non_tied)


def holm(raw):
    require(len(raw)==56 and all(Fraction(0)<=p<=1 for p in raw),'Complete 56-direction family required')
    order=sorted(range(len(raw)),key=lambda i:(raw[i],i))
    adjusted=[Fraction(0)]*len(raw);running=Fraction(0)
    for rank,index in enumerate(order):
        running=max(running,min(Fraction(1),raw[index]*(len(raw)-rank)))
        adjusted[index]=running
    return adjusted


def replay_directions(rows):
    groups=defaultdict(list)
    for row in rows:
        require(row['graph_id'] in ('G39','G12') and row['case_id'],'Unknown graph or empty case')
        groups[row['graph_id'],row['case_id']].append(row)
    require(bool(groups),'No witness families')
    output=[];states=defaultdict(list)
    for (graph,case),family in sorted(groups.items()):
        count=39 if graph=='G39' else 12
        require(len(family)==56,'Each witness must supply all 56 ordered directions')
        indexed={integer(r['comparison_index']):r for r in family}
        require(set(indexed)==set(range(56)) and len(indexed)==len(family),'Duplicate or missing direction')
        triples=[];raw=[]
        for index,(source,target) in enumerate(PAIRS):
            row=indexed[index]
            require((row['source'],row['target'])==(source,target),'Comparison axis differs')
            w,l,t=(integer(row[key]) for key in ('wins','losses','ties'))
            require(w+l+t==count,'Counts do not cover the matching cardinality')
            reverse=indexed[PAIRS.index((target,source))]
            require((w,l,t)==tuple(integer(reverse[key]) for key in ('losses','wins','ties')),'Opposite directions are inconsistent')
            triples.append((w,l,t));raw.append(sign_tail(w,w+l))
        adjusted=holm(raw)
        for index,((source,target),(w,l,t),p,q) in enumerate(zip(PAIRS,triples,raw,adjusted)):
            reject=q<=ALPHA;states[graph,index].append((case,reject))
            output.append(dict(case_id=case,graph_id=graph,comparison_index=index,source=source,target=target,
                wins=w,losses=l,ties=t,non_tied=w+l,raw_p_numerator=p.numerator,raw_p_denominator=p.denominator,
                holm_p_numerator=q.numerator,holm_p_denominator=q.denominator,holm_p=float(q),reject=int(reject)))
    sensitivity=[]
    for (graph,index),values in sorted(states.items()):
        rejecting=[case for case,reject in values if reject];nonrejecting=[case for case,reject in values if not reject]
        sensitivity.append(dict(graph_id=graph,comparison_index=index,source=PAIRS[index][0],target=PAIRS[index][1],
            witness_count=len(values),rejecting_witnesses=len(rejecting),nonrejecting_witnesses=len(nonrejecting),
            matching_dependence_demonstrated=int(bool(rejecting and nonrejecting)),
            rejecting_case=rejecting[0] if rejecting else '',nonrejecting_case=nonrejecting[0] if nonrejecting else ''))
    return output,sensitivity


def replay_leaders(rows):
    groups=defaultdict(dict)
    for row in rows:
        require(row['graph_id']=='G12' and row['matching_id'],'Leader input requires G12 matching summaries')
        require(integer(row['cardinality'])==12,'Leader matching cardinality differs')
        model=row['model'];require(model in MODELS,'Unknown leader model')
        values=groups[row['matching_id']];require(model not in values,'Duplicate model within matching')
        numerator=signed_integer(row['utility_sum_numerator']);denominator=integer(row['utility_sum_denominator'],1)
        values[model]=Fraction(numerator,denominator)/12
    require(bool(groups),'No matching utility summaries')
    output=[];counts={model:0 for model in MODELS}
    for matching,values in sorted(groups.items()):
        require(set(values)==set(MODELS),'Incomplete eight-model matching utility vector')
        best=max(values.values())
        for model in MODELS:
            value=values[model];leader=value==best;counts[model]+=int(leader)
            rank=1+sum(other>value for other in values.values())
            output.append(dict(matching_id=matching,graph_id='G12',model=model,
                mean_utility_numerator=value.numerator,mean_utility_denominator=value.denominator,
                rank_min=rank,is_leader=int(leader)))
    return output,[dict(graph_id='G12',model=model,leader_count=counts[model],matching_count=len(groups)) for model in MODELS]


def run(output,source=HERE/'source'):
    source=Path(source);output=Path(output)
    require(not output.exists(),'Use a new output directory')
    binding=json.loads((HERE/'BINDINGS.json').read_text())
    require(binding['schema']=='MATCHING_AGGREGATE_REPLAY_V1','Unexpected aggregate input schema')
    require(set(binding['files'])=={'witness_counts.tsv','g12_matching_utility_sums.tsv'},'Input roster differs')
    for name,digest in binding['files'].items():require(sha(source/name)==digest,'Changed bound aggregate input: '+name)
    counts=read_table(source/'witness_counts.tsv',COUNTS_HEADER)
    utilities=read_table(source/'g12_matching_utility_sums.tsv',LEADER_HEADER)
    directions,sensitivity=replay_directions(counts);leaders,leader_counts=replay_leaders(utilities)
    cases={(r['graph_id'],r['case_id']) for r in counts};matchings={r['matching_id'] for r in utilities}
    require(len(cases)==binding['witness_cases'] and len(matchings)==binding['g12_matching_count'],'Bound matching/case coverage differs')
    output.mkdir(parents=True)
    for name,rows in [('directional_decisions.tsv',directions),('directional_sensitivity.tsv',sensitivity),
                      ('matching_leaders.tsv',leaders),('leader_counts.tsv',leader_counts)]:write_table(output/name,rows)
    variable=[dict(graph_id=r['graph_id'],source=r['source'],target=r['target']) for r in sensitivity if r['matching_dependence_demonstrated']]
    tally={r['model']:r['leader_count'] for r in leader_counts}
    summary=dict(status='COMPLETE_MATCHING_SENSITIVITY_AGGREGATE_REPLAY',witness_cases=len(cases),directions_per_witness=56,
        matching_dependent_directions_demonstrated=variable,finite_g12_matchings=len(matchings),g12_leader_counts=tally,
        g12_multiple_models_can_lead=sum(count>0 for count in tally.values())>1,
        exact_sign_tails_and_holm=True,alpha_numerator=1,alpha_denominator=20,
        starting_point='Per-witness sign counts and per-matching model utility sums',
        maximum_matching_graph_recomputed=False,root_utility_generation_recomputed=False,
        all_graph_directional_certificates_recomputed=False,
        interpretation='Finite matching sensitivity; conditional sign-model decisions; enumeration counts are not probabilities',
        input_sha256=binding['files'],bindings_sha256=sha(HERE/'BINDINGS.json'),code_sha256=sha(__file__),
        artifacts={name:sha(output/name) for name in ('directional_decisions.tsv','directional_sensitivity.tsv','matching_leaders.tsv','leader_counts.tsv')})
    (output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.output)
