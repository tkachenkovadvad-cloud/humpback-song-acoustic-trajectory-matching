#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

VERSION='2.0'


def gui_inputs():
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
    root=tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
    tables=filedialog.askopenfilenames(title='Select Raven tables',filetypes=[('Raven tables','*.txt *.tsv *.csv'),('All files','*.*')])
    if not tables: root.destroy(); return [],None
    recs=[]
    for i,p in enumerate(tables,1):
        p=Path(p); m=re.match(r'\s*(\d+)',p.name); default=m.group(1) if m else str(i)
        rid=simpledialog.askstring('Record ID',f'Record ID for {p.name}:',initialvalue=default,parent=root) or default
        tr=None
        if messagebox.askyesno('Trajectory outputs',f'Use matcher outputs for record {rid}?',parent=root):
            x=filedialog.askdirectory(title=f'Select trajectory root for record {rid}',parent=root)
            tr=Path(x) if x else None
        recs.append((str(rid),p,tr))
    out=filedialog.askdirectory(title='Select output folder',parent=root)
    root.destroy(); return recs,(Path(out) if out else None)


def read_raven(path):
    for sep in ['\t',None,',']:
        try:
            df=pd.read_csv(path,sep=sep,engine='python',dtype=str,keep_default_na=False)
            df.columns=[str(c).strip() for c in df.columns]
            if 'Selection' in df.columns and any(c.lower()=='match' for c in df.columns): return df
        except Exception: pass
    raise ValueError(f'Cannot read Raven table: {path}')


def col(df,name):
    for c in df.columns:
        if c.strip().lower()==name.lower(): return c
    raise KeyError(f'Missing column {name}; found {list(df.columns)}')


def parse_match(v):
    out=[]
    for g,s in re.findall(r'(?<!\d)(\d+)\s*([lLrR]?)',str(v).strip()):
        z=(int(g),s.lower())
        if z not in out: out.append(z)
    return out


def fnum(v):
    try:return float(str(v).strip())
    except:return math.nan


def build_occurrences(rid,df):
    sc,mc,bc,ec=col(df,'Selection'),col(df,'match'),col(df,'Begin Time (s)'),col(df,'End Time (s)')
    rows=[]
    for ix,r in df.iterrows():
        s=int(float(r[sc])); memberships=parse_match(r[mc]); amb=len(memberships)>1
        if not memberships:
            rows.append(dict(record=rid,source_row=ix+2,selection=s,begin_s=fnum(r[bc]),end_s=fnum(r[ec]),raw_match=r[mc],group=np.nan,side='',membership_count=0,ambiguous_overlap=False))
        else:
            for g,side in memberships:
                rows.append(dict(record=rid,source_row=ix+2,selection=s,begin_s=fnum(r[bc]),end_s=fnum(r[ec]),raw_match=r[mc],group=g,side=side,membership_count=len(memberships),ambiguous_overlap=amb))
    return pd.DataFrame(rows).sort_values(['selection','group'],na_position='last').reset_index(drop=True)


def read_similarity(root):
    if root is None or not root.exists(): return pd.DataFrame()
    preferred=[root/'trajectory_similarity_matches_all_groups.csv',root/'trajectory_similarity_matches_both_records.csv']
    files=[p for p in preferred if p.exists()] or sorted(root.rglob('trajectory_similarity_matches.csv'))
    all=[]
    for p in files:
        try:d=pd.read_csv(p)
        except Exception as e: print('WARNING',p,e); continue
        gc=next((c for c in d.columns if c.lower() in {'matching_group','match_group','group','group_id'}),None)
        d['_group']=pd.to_numeric(d[gc],errors='coerce') if gc else (int(p.parent.name) if p.parent.name.isdigit() else np.nan)
        all.append(d)
    return pd.concat(all,ignore_index=True,sort=False) if all else pd.DataFrame()


def similarity_summary(sim):
    cols=['group','similarity_pairs','median_combined_distance','mean_combined_distance','best_combined_distance','strong_pair_fraction','match_or_strong_fraction','median_permutation_p','median_effect_z','cohesion_score']
    if sim.empty:return pd.DataFrame(columns=cols)
    rows=[]
    for gv,p in sim.groupby('_group',dropna=True):
        dc=next((c for c in ['combined_distance','screen_combined_distance'] if c in p.columns),None)
        d=pd.to_numeric(p[dc],errors='coerce').dropna() if dc else pd.Series(dtype=float)
        dec=p['decision'].astype(str).str.lower() if 'decision' in p else pd.Series(['']*len(p))
        strong=dec.eq('strong_match_candidate'); matched=dec.isin(['strong_match_candidate','match_candidate'])
        pv=pd.to_numeric(p['permutation_p'],errors='coerce').dropna() if 'permutation_p' in p else pd.Series(dtype=float)
        ez=pd.to_numeric(p['effect_z'],errors='coerce').dropna() if 'effect_z' in p else pd.Series(dtype=float)
        comp=[]
        if len(d): comp.append(math.exp(-float(d.median())/max(float(d.quantile(.75)),1e-9)))
        if len(dec): comp.append(float(matched.mean()))
        rows.append(dict(group=int(gv),similarity_pairs=len(p),median_combined_distance=d.median() if len(d) else np.nan,mean_combined_distance=d.mean() if len(d) else np.nan,best_combined_distance=d.min() if len(d) else np.nan,strong_pair_fraction=strong.mean() if len(dec) else np.nan,match_or_strong_fraction=matched.mean() if len(dec) else np.nan,median_permutation_p=pv.median() if len(pv) else np.nan,median_effect_z=ez.median() if len(ez) else np.nan,cohesion_score=np.mean(comp) if comp else np.nan))
    return pd.DataFrame(rows,columns=cols).sort_values('group')


def nodes_table(rid,occ,sim):
    m=occ.dropna(subset=['group']).copy(); m['group']=m['group'].astype(int); rows=[]
    for g,p in m.groupby('group'):
        sels=sorted(set(p.selection.astype(int)))
        rows.append(dict(record=rid,group=g,selection_count=len(sels),occurrence_rows=len(p),selections=';'.join(map(str,sels)),first_selection=min(sels),last_selection=max(sels),first_time_s=p.begin_s.min(),last_time_s=p.end_s.max(),ambiguous_selection_count=p.loc[p.ambiguous_overlap,'selection'].nunique()))
    n=pd.DataFrame(rows)
    return n.merge(sim,on='group',how='left').sort_values('group') if len(n) and len(sim) else n


def memberships(occ):
    d=defaultdict(list)
    for r in occ.dropna(subset=['group']).itertuples(index=False):
        d[int(r.selection)].append(dict(group=int(r.group),side=str(r.side),ambiguous=bool(r.ambiguous_overlap),begin=float(r.begin_s),end=float(r.end_s)))
    return d


def edge_occurrences(rid,occ,max_unmatched=0):
    mp=memberships(occ); matched=sorted(mp); rows=[]
    for i,a in enumerate(matched[:-1]):
        b=matched[i+1]; between=b-a-1
        if between>max_unmatched: continue
        for x in mp[a]:
            for y in mp[b]:
                amb=x['ambiguous'] or y['ambiguous']
                rows.append(dict(record=rid,source_group=x['group'],target_group=y['group'],source_selection=a,target_selection=b,source_side=x['side'],target_side=y['side'],gap_selections=between,gap_s=y['begin']-x['end'],ambiguous_overlap=amb,primary_support=not amb))
    return pd.DataFrame(rows)


def edge_summary(e,min_support):
    cols=['record','source_group','target_group','support_total','support_primary','support_ambiguous','selection_pairs','median_gap_s','min_gap_s','max_gap_s','mean_gap_s','repeated_primary_edge']
    if e.empty:return pd.DataFrame(columns=cols)
    rows=[]
    for (rid,a,b),p in e.groupby(['record','source_group','target_group']):
        pri=p[p.primary_support]; amb=p[p.ambiguous_overlap]
        pairs=';'.join(f'{int(r.source_selection)}->{int(r.target_selection)}'+('*' if r.ambiguous_overlap else '') for r in p.itertuples())
        rows.append(dict(record=rid,source_group=int(a),target_group=int(b),support_total=len(p),support_primary=len(pri),support_ambiguous=len(amb),selection_pairs=pairs,median_gap_s=p.gap_s.median(),min_gap_s=p.gap_s.min(),max_gap_s=p.gap_s.max(),mean_gap_s=p.gap_s.mean(),repeated_primary_edge=len(pri)>=min_support))
    return pd.DataFrame(rows,columns=cols).sort_values(['support_primary','source_group','target_group'],ascending=[False,True,True])


def primary_runs(occ):
    mp=memberships(occ); runs=[]; cur=[]
    for s in sorted(set(occ.selection.astype(int))):
        valid=[x for x in mp.get(s,[]) if not x['ambiguous']]
        if len(valid)==1: cur.append((s,valid[0]['group']))
        else:
            if cur:runs.append(cur);cur=[]
    if cur:runs.append(cur)
    return runs


def _max_nonoverlapping_occurrences(occurrences):
    """
    Greedy interval scheduling gives the maximum number of non-overlapping
    occurrences because all intervals for one pattern have equal length.
    Occurrences from different primary runs are automatically independent.
    """
    chosen = []
    by_run = defaultdict(list)
    for item in occurrences:
        by_run[item["run_id"]].append(item)

    for run_id, items in by_run.items():
        items = sorted(items, key=lambda z: (z["end_index"], z["start_index"]))
        last_end = -1
        for item in items:
            if item["start_index"] > last_end:
                chosen.append(item)
                last_end = item["end_index"]
    return chosen


def themes_table(rid, occ, nodes, min_support, min_len, max_len):
    """
    Find maximal repeated contiguous group sequences.

    Support is based on the maximum number of NON-OVERLAPPING occurrences.
    Thus one long alternating passage cannot inflate support merely because
    many overlapping windows fit inside it.
    """
    occurrence_map = defaultdict(list)

    runs = primary_runs(occ)
    for run_id, run in enumerate(runs, start=1):
        groups = [g for _, g in run]
        selections = [s for s, _ in run]
        upper = min(max_len, len(groups))

        for length in range(min_len, upper + 1):
            for start_i in range(0, len(groups) - length + 1):
                end_i = start_i + length - 1
                pattern = tuple(groups[start_i:start_i + length])
                occurrence_map[pattern].append({
                    "run_id": run_id,
                    "start_index": start_i,
                    "end_index": end_i,
                    "start_selection": selections[start_i],
                    "end_selection": selections[end_i],
                })

    candidate_info = {}
    for pattern, occurrences in occurrence_map.items():
        independent = _max_nonoverlapping_occurrences(occurrences)
        independent_support = len(independent)
        if independent_support < min_support:
            continue
        candidate_info[pattern] = {
            "raw_occurrence_count": len(occurrences),
            "independent_support": independent_support,
            "independent_occurrences": independent,
            "all_occurrences": occurrences,
        }

    maximal = {}
    for pattern, info in candidate_info.items():
        dominated_by = None
        for longer, longer_info in candidate_info.items():
            if len(longer) <= len(pattern):
                continue
            if longer_info["independent_support"] < info["independent_support"]:
                continue
            if any(
                longer[i:i + len(pattern)] == pattern
                for i in range(len(longer) - len(pattern) + 1)
            ):
                dominated_by = longer
                break
        if dominated_by is None:
            maximal[pattern] = info

    cohesion = (
        dict(zip(nodes.group, nodes.cohesion_score))
        if len(nodes) and "cohesion_score" in nodes
        else {}
    )

    rows = []
    for pattern, info in maximal.items():
        cohesion_values = [cohesion.get(group, np.nan) for group in pattern]
        cohesion_values = [v for v in cohesion_values if pd.notna(v)]
        mean_cohesion = float(np.mean(cohesion_values)) if cohesion_values else np.nan

        independent_ranges = ";".join(
            f'run{item["run_id"]}:{item["start_selection"]}-{item["end_selection"]}'
            for item in info["independent_occurrences"]
        )
        all_ranges = ";".join(
            f'run{item["run_id"]}:{item["start_selection"]}-{item["end_selection"]}'
            for item in info["all_occurrences"]
        )

        cohesion_factor = 0.5 + 0.5 * mean_cohesion if pd.notna(mean_cohesion) else 1.0
        score = len(pattern) * info["independent_support"] * cohesion_factor

        rows.append({
            "record": rid,
            "group_sequence": " -> ".join(map(str, pattern)),
            "length": len(pattern),
            "independent_support": info["independent_support"],
            "raw_occurrence_count": info["raw_occurrence_count"],
            "overlapping_occurrences_removed": (
                info["raw_occurrence_count"] - info["independent_support"]
            ),
            "independent_selection_ranges": independent_ranges,
            "all_selection_ranges": all_ranges,
            "mean_group_cohesion": mean_cohesion,
            "candidate_score": score,
        })

    columns = [
        "record", "candidate_rank", "group_sequence", "length",
        "independent_support", "raw_occurrence_count",
        "overlapping_occurrences_removed", "independent_selection_ranges",
        "all_selection_ranges", "mean_group_cohesion", "candidate_score",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    result = pd.DataFrame(rows).sort_values(
        ["candidate_score", "length", "independent_support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    result.insert(1, "candidate_rank", np.arange(1, len(result) + 1))
    return result[columns]

def graph_outputs(out,rid,nodes,edges):
    try:
        import networkx as nx
    except ImportError:
        print('WARNING: pip install networkx matplotlib for GraphML/PNG');return
    G=nx.DiGraph(record=rid)
    for r in nodes.itertuples(index=False):
        attrs={k:('' if pd.isna(v) else v.item() if hasattr(v,'item') else v) for k,v in r._asdict().items() if k not in {'record','group'}}
        G.add_node(str(int(r.group)),**attrs)
    for r in edges.itertuples(index=False):
        attrs={k:('' if pd.isna(v) else v.item() if hasattr(v,'item') else v) for k,v in r._asdict().items() if k not in {'record','source_group','target_group'}}
        G.add_edge(str(int(r.source_group)),str(int(r.target_group)),**attrs)
    nx.write_graphml(G,out/'matching_group_transition_graph.graphml')
    try:
        import matplotlib.pyplot as plt
        pos=nx.spring_layout(G,seed=42); widths=[.8+.8*int(G[u][v].get('support_primary',1)) for u,v in G.edges]
        fig,ax=plt.subplots(figsize=(16,12)); nx.draw(G,pos,with_labels=True,node_size=900,width=widths,arrowsize=18,ax=ax)
        labs={(u,v):str(G[u][v].get('support_primary',0)) for u,v in G.edges}; nx.draw_networkx_edge_labels(G,pos,edge_labels=labs,font_size=8,ax=ax)
        ax.set_title('Matching-group transition graph\nEdge label = primary support');fig.tight_layout();fig.savefig(out/'matching_group_transition_graph.png',dpi=250,bbox_inches='tight');plt.close(fig)
    except Exception as e: print('WARNING: PNG skipped:',e)


def process(rid,table,trroot,outroot,args):
    print(f'\n=== Record {rid} ==='); out=outroot/f'record_{rid}';out.mkdir(parents=True,exist_ok=True)
    occ=build_occurrences(rid,read_raven(table)); sim=similarity_summary(read_similarity(trroot)); nodes=nodes_table(rid,occ,sim)
    eo=edge_occurrences(rid,occ,args.max_unmatched_between); edges=edge_summary(eo,args.min_support); themes=themes_table(rid,occ,nodes,args.min_support,args.min_theme_length,args.max_theme_length)
    files={'occ':out/'selection_group_occurrences.csv','nodes':out/'graph_nodes_matching_groups.csv','eo':out/'graph_edge_occurrences.csv','edges':out/'graph_edges_summary.csv','themes':out/'candidate_themes_maximal_nonoverlapping.csv','sim':out/'group_similarity_summary.csv'}
    occ.to_csv(files['occ'],index=False);nodes.to_csv(files['nodes'],index=False);eo.to_csv(files['eo'],index=False);edges.to_csv(files['edges'],index=False);themes.to_csv(files['themes'],index=False);sim.to_csv(files['sim'],index=False);graph_outputs(out,rid,nodes,edges)
    print('Groups:',len(nodes),'Edges:',len(edges),'Repeated edges:',int(edges.repeated_primary_edge.sum()) if len(edges) else 0,'Candidate themes:',len(themes));print('Output:',out)
    return nodes,edges,themes,occ


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--table',action='append',type=Path);ap.add_argument('--trajectory-root',action='append',type=Path);ap.add_argument('--record-id',action='append');ap.add_argument('--output',type=Path);ap.add_argument('--min-support',type=int,default=2);ap.add_argument('--min-theme-length',type=int,default=2);ap.add_argument('--max-theme-length',type=int,default=12);ap.add_argument('--max-unmatched-between',type=int,default=0);a=ap.parse_args()
    print('Theme graph builder',VERSION)
    print('Theme mining: maximal repeated contiguous sequences with NON-OVERLAPPING support')
    if a.table:
        roots=a.trajectory_root or [];ids=a.record_id or [];recs=[]
        for i,t in enumerate(a.table):
            m=re.match(r'\s*(\d+)',t.name);rid=ids[i] if i<len(ids) else (m.group(1) if m else str(i+1));recs.append((rid,t,roots[i] if i<len(roots) else None))
        out=a.output or Path.cwd()/'theme_graph_outputs'
    else:
        recs,out=gui_inputs()
        if not recs or out is None: print('No inputs selected.');return 0
    out.mkdir(parents=True,exist_ok=True); N=[];E=[];T=[];O=[]
    for r,t,tr in recs:
        n,e,th,o=process(r,t,tr,out,a);N.append(n);E.append(e);T.append(th);O.append(o)
    pd.concat(N,ignore_index=True,sort=False).to_csv(out/'ALL_RECORDS_graph_nodes.csv',index=False);pd.concat(E,ignore_index=True,sort=False).to_csv(out/'ALL_RECORDS_graph_edges.csv',index=False);pd.concat(T,ignore_index=True,sort=False).to_csv(out/'ALL_RECORDS_candidate_themes_maximal_nonoverlapping.csv',index=False);pd.concat(O,ignore_index=True,sort=False).to_csv(out/'ALL_RECORDS_selection_group_occurrences.csv',index=False)
    print('\nDONE:',out);return 0

if __name__=='__main__': raise SystemExit(main())
