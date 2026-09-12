#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, base64, json, math, sys, warnings, webbrowser, zipfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import plotly.graph_objects as go
try:
    import umap
except Exception:
    umap = None

@dataclass
class Config:
    n_fft:int=8192; win_length:int=8192; hop_length:int=1536
    n_mfcc:int=20; fmin:float=20.0; fmax:float=8000.0
    novelty_smooth_ms:float=80.0; min_event_ms:float=120.0
    peak_prominence_quantile:float=0.68; max_nodes:int=40; min_nodes:int=4
    pca_components:int=10; random_state:int=42

def interp_nan(x):
    x=np.asarray(x,float); good=np.isfinite(x)
    if good.all(): return x
    if good.sum()==0: return np.zeros_like(x)
    idx=np.arange(len(x)); return np.interp(idx,idx[good],x[good])

def smooth(x,w):
    w=max(1,int(w))
    if w<=1:return x
    return np.convolve(x,np.ones(w)/w,mode='same')

def spec_entropy(S):
    p=S/np.maximum(S.sum(axis=0,keepdims=True),1e-12)
    return -(p*np.log2(np.maximum(p,1e-12))).sum(axis=0)

def extract(y,sr,cfg):
    if y.ndim>1:y=y.mean(axis=1)
    y=y.astype(np.float32,copy=False)
    fmax=min(cfg.fmax,sr/2-1)
    Sc=librosa.stft(y,n_fft=cfg.n_fft,hop_length=cfg.hop_length,win_length=cfg.win_length,window='hann',center=True)
    Sm=np.abs(Sc); Sp=Sm**2; freqs=librosa.fft_frequencies(sr=sr,n_fft=cfg.n_fft)
    m=(freqs>=cfg.fmin)&(freqs<=fmax); Sb=Sm[m]; Sbp=Sp[m]; bf=freqs[m]
    rms=librosa.feature.rms(S=Sm,frame_length=cfg.n_fft)[0]
    cent=librosa.feature.spectral_centroid(S=Sb,freq=bf)[0]
    bw=librosa.feature.spectral_bandwidth(S=Sb,freq=bf)[0]
    roll=librosa.feature.spectral_rolloff(S=Sb,sr=sr,roll_percent=.85)[0]
    flat=librosa.feature.spectral_flatness(S=np.maximum(Sb,1e-12))[0]
    ent=spec_entropy(Sbp)
    ns=Sb/np.maximum(np.linalg.norm(Sb,axis=0,keepdims=True),1e-12)
    flux=np.sqrt(np.sum(np.diff(ns,axis=1,prepend=ns[:,:1])**2,axis=0))
    mel=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=cfg.n_fft,hop_length=cfg.hop_length,win_length=cfg.win_length,n_mels=64,fmin=cfg.fmin,fmax=fmax,power=2.0)
    mfcc=librosa.feature.mfcc(S=librosa.power_to_db(mel,ref=np.max),n_mfcc=cfg.n_mfcc)
    try:
        frame_length=max(cfg.win_length,2048)
        pyin_fmin=max(float(cfg.fmin), (2.0 * float(sr) / frame_length) + 0.1)
        f0,vf,vp=librosa.pyin(y,fmin=pyin_fmin,fmax=fmax,sr=sr,frame_length=frame_length,hop_length=cfg.hop_length,center=True)
        f0=interp_nan(f0); vp=interp_nan(vp)
    except Exception as e:
        warnings.warn(f'pYIN failed: {e}'); f0=np.zeros_like(rms); vp=np.zeros_like(rms)
    n=min([len(rms),len(cent),len(bw),len(roll),len(flat),len(ent),len(flux),len(f0),len(vp),mfcc.shape[1]])
    t=librosa.frames_to_time(np.arange(n),sr=sr,hop_length=cfg.hop_length)
    d={'frame_index':np.arange(n),'time_s':t,'rms':rms[:n],'rms_db':librosa.amplitude_to_db(np.maximum(rms[:n],1e-12),ref=np.max),
       'spectral_centroid_hz':cent[:n],'spectral_bandwidth_hz':bw[:n],'rolloff85_hz':roll[:n],'spectral_flatness':flat[:n],
       'spectral_entropy':ent[:n],'spectral_flux':flux[:n],'pitch_hz':f0[:n],'voiced_probability':vp[:n]}
    for i in range(cfg.n_mfcc):d[f'mfcc_{i+1:02d}']=mfcc[i,:n]
    df=pd.DataFrame(d)
    cols=['rms_db','spectral_centroid_hz','spectral_bandwidth_hz','rolloff85_hz','spectral_flatness','spectral_entropy','spectral_flux','pitch_hz','voiced_probability']+[f'mfcc_{i+1:02d}' for i in range(cfg.n_mfcc)]
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
    return df,X

def nodes_from(df,X,sr,cfg):
    Z=RobustScaler(quantile_range=(10,90)).fit_transform(X)

    # Frame-to-frame acoustic change ("novelty")
    novelty=np.linalg.norm(np.diff(Z,axis=0,prepend=Z[:1]),axis=1)
    smooth_frames=max(1,round(cfg.novelty_smooth_ms/1000*sr/cfg.hop_length))
    novelty=smooth(novelty,smooth_frames)

    min_dist=max(1,round(cfg.min_event_ms/1000*sr/cfg.hop_length))
    duration_s=float(df.time_s.iloc[-1]-df.time_s.iloc[0]) if len(df)>1 else 0.0

    # For a long signal, four nodes are not useful. Set an adaptive minimum,
    # but still place boundaries on actual novelty maxima rather than equal slices.
    desired_min=min(cfg.max_nodes, max(8, round(duration_s/1.0)))

    pos=novelty[novelty>0]
    prom=float(np.quantile(pos,0.35)) if len(pos) else 0.0
    peaks,props=find_peaks(
        novelty,
        distance=min_dist,
        prominence=max(prom,1e-9)
    )

    # If the strict prominence threshold found too few changes, supplement it
    # with the strongest local novelty maxima while respecting minimum spacing.
    if len(peaks) < desired_min-1:
        all_peaks,_=find_peaks(novelty,distance=min_dist)
        ranked=all_peaks[np.argsort(novelty[all_peaks])[::-1]]

        chosen=list(map(int,peaks))
        for p in ranked:
            p=int(p)
            if all(abs(p-q)>=min_dist for q in chosen):
                chosen.append(p)
            if len(chosen)>=desired_min-1:
                break
        peaks=np.array(sorted(chosen),dtype=int)

    # Keep only the strongest boundaries if there are too many.
    if len(peaks)>cfg.max_nodes-1:
        strongest=np.argsort(novelty[peaks])[-(cfg.max_nodes-1):]
        peaks=np.sort(peaks[strongest])

    bounds=np.unique(np.r_[0,peaks,len(df)-1]).astype(int)

    rows=[]; vecs=[]
    fcols=[c for c in df.columns if c not in ('frame_index','time_s')]

    for nid,(a,b) in enumerate(zip(bounds[:-1],bounds[1:]),1):
        if b<=a:
            continue
        seg=df.iloc[a:b+1]
        sx=X[a:b+1]
        pk=a+int(np.argmax(seg.rms.to_numpy()))

        r={
            'node_id':nid,
            'start_frame':a,
            'peak_frame':pk,
            'end_frame':b,
            'start_time_s':float(df.iloc[a].time_s),
            'peak_time_s':float(df.iloc[pk].time_s),
            'end_time_s':float(df.iloc[b].time_s),
            'duration_s':float(df.iloc[b].time_s-df.iloc[a].time_s),
            'n_frames':b-a+1,
            'novelty_at_start':float(novelty[a]),
            'novelty_at_peak':float(novelty[pk])
        }

        for c in fcols:
            r[c]=float(np.nanmedian(seg[c]))

        r['peak_rms']=float(df.iloc[pk].rms)
        r['peak_rms_db']=float(df.iloc[pk].rms_db)
        rows.append(r)
        vecs.append(np.nanmedian(sx,axis=0))

    return pd.DataFrame(rows),np.asarray(vecs)

def project_frames(X,cfg,max_embed_points=1200):
    Xs=RobustScaler(quantile_range=(10,90)).fit_transform(X)
    npc=max(1,min(cfg.pca_components,len(X)-1,X.shape[1]))
    Xp=PCA(n_components=npc,random_state=cfg.random_state).fit_transform(Xs)
    n=len(Xp)
    idx=np.arange(n) if n<=max_embed_points else np.unique(np.linspace(0,n-1,max_embed_points).round().astype(int))
    if umap is not None and len(idx)>=6:
        emb=umap.UMAP(n_components=3,n_neighbors=min(20,len(idx)-1),min_dist=.03,metric='euclidean',random_state=cfg.random_state).fit_transform(Xp[idx])
        method=f'PCA({npc}) → UMAP3D on {len(idx)} temporal samples → interpolation'
    else:
        emb=np.zeros((len(idx),3)); emb[:,:min(3,Xp.shape[1])]=Xp[idx,:3]
        method=f'PCA({npc}) → XYZ'
    full=np.column_stack([np.interp(np.arange(n),idx,emb[:,j]) for j in range(3)])
    win=max(3,min(21,(n//250)*2+3))
    if win%2==0: win+=1
    for j in range(3): full[:,j]=smooth(full[:,j],win)
    return full,method

def normsize(v,lo=8,hi=28):
    v=np.asarray(v,float); a=np.nanmin(v); b=np.nanmax(v)
    if not np.isfinite(a) or not np.isfinite(b) or math.isclose(a,b):return np.full_like(v,(lo+hi)/2)
    return lo+(v-a)/(b-a)*(hi-lo)


def shared_project(datasets,cfg,max_fit_points=4000):
    """
    Fit one shared feature space for all selected signals.

    Robust scaling is followed by clipping so silent/boundary frames cannot
    create enormous PCA coordinates. PCA scores are standardized before UMAP
    or before the PCA-only fallback.
    """
    all_X=np.vstack([d['X'] for d in datasets])

    scaler=RobustScaler(quantile_range=(10,90))
    all_scaled=scaler.fit_transform(all_X)
    all_scaled=np.nan_to_num(all_scaled,nan=0.0,posinf=0.0,neginf=0.0)
    all_scaled=np.clip(all_scaled,-8.0,8.0)

    npc=max(1,min(cfg.pca_components,len(all_scaled)-1,all_scaled.shape[1]))
    pca=PCA(n_components=npc,random_state=cfg.random_state)
    all_pca=pca.fit_transform(all_scaled)

    # Put PCA dimensions onto comparable scales.
    pca_scaler=RobustScaler(quantile_range=(10,90))
    all_pca_scaled=pca_scaler.fit_transform(all_pca)
    all_pca_scaled=np.clip(
        np.nan_to_num(all_pca_scaled,nan=0.0,posinf=0.0,neginf=0.0),
        -8.0,8.0
    )

    total=len(all_pca_scaled)
    fit_idx=np.arange(total) if total<=max_fit_points else np.unique(
        np.linspace(0,total-1,max_fit_points).round().astype(int)
    )

    if umap is not None and len(fit_idx)>=6:
        reducer=umap.UMAP(
            n_components=3,
            n_neighbors=min(25,len(fit_idx)-1),
            min_dist=.05,
            metric='euclidean',
            random_state=cfg.random_state,
            transform_seed=cfg.random_state
        )
        reducer.fit(all_pca_scaled[fit_idx])
        all_xyz=reducer.transform(all_pca_scaled)
        method=f'Shared clipped RobustScaler → PCA({npc}) → UMAP3D'
    else:
        all_xyz=np.zeros((total,3))
        all_xyz[:,:min(3,all_pca_scaled.shape[1])]=all_pca_scaled[:,:3]
        method=f'Shared clipped RobustScaler → standardized PCA({npc}) → XYZ (UMAP unavailable)'

    pos=0
    for d in datasets:
        n=len(d['fr'])
        xyz=all_xyz[pos:pos+n].copy()

        # Gentle temporal smoothing only; does not alter the common embedding.
        win=max(3,min(21,(n//250)*2+3))
        if win%2==0:
            win+=1
        for j in range(3):
            xyz[:,j]=smooth(xyz[:,j],win)

        d['xyz']=xyz
        d['fr'][['X','Y','Z']]=xyz
        pos+=n

    return method

def add_event_coordinates(d):
    fr=d['fr']; nd=d['nd']; xyz=d['xyz']
    peak_idx=nd.peak_frame.astype(int).clip(0,len(fr)-1).to_numpy()
    nd[['peak_X','peak_Y','peak_Z']]=xyz[peak_idx]
    cents=[]
    for _,r in nd.iterrows():
        a=max(0,int(r.start_frame)); b=min(len(fr)-1,int(r.end_frame))
        cents.append(np.median(xyz[a:b+1],axis=0))
    cents=np.asarray(cents)
    nd[['centroid_X','centroid_Y','centroid_Z']]=cents
    nd[['X','Y','Z']]=cents


def export_signal(d,method):
    path=d['path']; fr=d['fr']; nd=d['nd']; xyz=d['xyz']
    stem=path.with_suffix('')
    fcsv=Path(str(stem)+'_frames_shared_space.csv')
    ecsv=Path(str(stem)+'_events_centroids_shared_space.csv')
    scsv=Path(str(stem)+'_summary_shared_space.csv')

    fr2=fr.copy()
    fr2.insert(0,'source_file',path.name)
    fr2.to_csv(fcsv,index=False)

    ev=nd.copy()
    ev.insert(0,'source_file',path.name)
    ev.to_csv(ecsv,index=False)

    edges=np.linalg.norm(np.diff(xyz,axis=0),axis=1) if len(xyz)>1 else np.array([])
    pd.DataFrame([{
        'source_file':path.name,
        'sample_rate_hz':d['sr'],
        'duration_s':d['duration'],
        'projection':method,
        'node_count':len(nd),
        'trajectory_path_length':float(edges.sum()) if len(edges) else 0,
        'mean_edge_length':float(edges.mean()) if len(edges) else 0
    }]).to_csv(scsv,index=False)

    return fcsv,ecsv,scsv


def layout_offsets(datasets):
    robust_spans=[]
    for d in datasets:
        xyz=d['xyz']
        lo=np.quantile(xyz,0.02,axis=0)
        hi=np.quantile(xyz,0.98,axis=0)
        robust_spans.append(hi-lo)

    spans=np.asarray(robust_spans)
    step_x=max(2.0,float(np.nanmedian(spans[:,0]))*1.6)
    step_y=max(2.0,float(np.nanmedian(spans[:,1]))*1.6)
    cols=math.ceil(math.sqrt(len(datasets)))

    return np.asarray([
        [(i%cols)*step_x,(i//cols)*step_y,0.0]
        for i in range(len(datasets))
    ],float)

def make_combined_html(datasets,method,out):
    offsets=layout_offsets(datasets)
    fig=go.Figure()
    payload=[]
    trace_map=[]

    for i,d in enumerate(datasets):
        xyz=d['xyz']; nd=d['nd']; name=d['path'].name; off=offsets[i]
        shown=xyz+off

        line_idx=len(fig.data)
        fig.add_trace(go.Scatter3d(
            x=shown[:,0],y=shown[:,1],z=shown[:,2],
            mode='lines',line={'width':4},
            name=name,hoverinfo='skip'
        ))

        marker_idx=len(fig.data)
        nxyz=nd[['X','Y','Z']].to_numpy(float)+off
        fig.add_trace(go.Scatter3d(
            x=nxyz[:,0],y=nxyz[:,1],z=nxyz[:,2],
            mode='markers',
            marker={'size':7,'symbol':'circle','opacity':.85},
            name=f'{name} events',
            customdata=np.c_[np.full(len(nd),i),nd.peak_time_s.to_numpy(float)],
            hovertext=[
                f'<b>{name}</b><br>event {int(r.node_id)}<br>{r.start_time_s:.3f}–{r.end_time_s:.3f} s'
                for _,r in nd.iterrows()
            ],
            hoverinfo='text',
            showlegend=False
        ))
        trace_map.append([line_idx,marker_idx])

        audio_uri='data:audio/wav;base64,'+base64.b64encode(d['path'].read_bytes()).decode('ascii')
        payload.append({
            'name':name,
            'times':d['fr'].time_s.round(8).tolist(),
            'x':xyz[:,0].round(8).tolist(),
            'y':xyz[:,1].round(8).tolist(),
            'z':xyz[:,2].round(8).tolist(),
            'events':nd[['X','Y','Z']].round(8).to_numpy().tolist(),
            'offset':off.round(8).tolist(),
            'audio':audio_uri,
            'duration':float(d['duration']),
            'frames_csv':d['path'].with_suffix('').name+'_frames_shared_space.csv',
            'events_csv':d['path'].with_suffix('').name+'_events_centroids_shared_space.csv',
            'summary_csv':d['path'].with_suffix('').name+'_summary_shared_space.csv'
        })

    played_idx=len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=[],y=[],z=[],mode='lines',
        line={'width':8},name='Played',hoverinfo='skip'
    ))

    moving_idx=len(fig.data)
    first=(datasets[0]['xyz'][0]+offsets[0]).tolist()
    fig.add_trace(go.Scatter3d(
        x=[first[0]],y=[first[1]],z=[first[2]],
        mode='markers',
        marker={'size':12,'symbol':'circle'},
        name='Playback',hoverinfo='skip'
    ))

    fig.update_layout(
        title=f'Multiple signals in one shared acoustic space<br><sup>{method}</sup>',
        scene={
            'dragmode':'orbit',
            'xaxis_title':'Shared X',
            'yaxis_title':'Shared Y',
            'zaxis_title':'Shared Z',
            'aspectmode':'data'
        },
        margin={'l':0,'r':0,'t':85,'b':0},
        legend={'orientation':'h'},
        uirevision='keep-camera',
        height=820
    )

    ph=fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id='multiPlot',
        config={
            'responsive':True,
            'displaylogo':False,
            'scrollZoom':True,
            'displayModeBar':True,
            'modeBarButtonsToAdd':['pan3d','zoom3d','resetCameraDefault3d'],
            'doubleClick':'reset+autosize'
        }
    )

    options=''.join(
        f'<option value="{i}">{d["path"].name}</option>'
        for i,d in enumerate(datasets)
    )

    html=f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Shared acoustic trajectories</title>
<style>
body{{margin:0;font-family:Arial;background:#111;color:#eee}}
.controls{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px 16px;background:#1b1b1b;position:sticky;top:0;z-index:10}}
select,button{{padding:7px 10px}}
audio{{width:min(620px,65vw)}}
#timeLabel{{min-width:145px}}
</style>
</head>
<body>
<div class="controls">
<label>Play signal: <select id="signalSelect">{options}</select></label>
<audio id="audio" controls preload="auto"></audio>
<span id="timeLabel"></span>
<button id="side">Side by side</button>
<button id="shared">True shared space</button>
<button id="overlay">Overlay starts</button>
<button id="orbitMode">Mouse: orbit</button>
<button id="panMode">Mouse: pan</button>
<button id="downloadFrames">CSV: current frames</button>
<button id="downloadEvents">CSV: current events</button>
<button id="downloadSummary">CSV: current summary</button>
<a id="downloadAll" href="all_signals_events_centroids_shared_space.csv" download>
<button type="button">CSV: all events</button>
</a>
<a id="downloadAllPerSignal" href="all_signals_per_selection_csv.zip" download>
<button type="button">CSV: all selections (frames + events + summary)</button>
</a>
</div>
{ph}
<script>
const signals={json.dumps(payload)};
const traceMap={json.dumps(trace_map)};
const playedTrace={played_idx};
const movingTrace={moving_idx};

const plot=document.getElementById('multiPlot');
const sel=document.getElementById('signalSelect');
const audio=document.getElementById('audio');
const lab=document.getElementById('timeLabel');

let mode='side';
let active=0;
let raf=null;
let lastI=-1;

function coordsFor(k,which){{
  const s=signals[k];
  const x=s.x.slice(),y=s.y.slice(),z=s.z.slice();

  if(which==='side'){{
    const xs=x.slice().sort((a,b)=>a-b);
    const ys=y.slice().sort((a,b)=>a-b);
    const zs=z.slice().sort((a,b)=>a-b);
    const mid=Math.floor(x.length/2);
    const cx=xs[mid],cy=ys[mid],cz=zs[mid];
    for(let i=0;i<x.length;i++){{
      x[i]=x[i]-cx+s.offset[0];
      y[i]=y[i]-cy+s.offset[1];
      z[i]=z[i]-cz+s.offset[2];
    }}
  }} else if(which==='overlay'){{
    const x0=x[0],y0=y[0],z0=z[0];
    for(let i=0;i<x.length;i++){{
      x[i]-=x0;
      y[i]-=y0;
      z[i]-=z0;
    }}
  }}
  return {{x,y,z}};
}}

function eventCoordsFor(k,which){{
  const s=signals[k];
  const e=s.events.map(v=>v.slice());

  if(which==='side'){{
    const xs=s.x.slice().sort((a,b)=>a-b);
    const ys=s.y.slice().sort((a,b)=>a-b);
    const zs=s.z.slice().sort((a,b)=>a-b);
    const mid=Math.floor(s.x.length/2);
    const cx=xs[mid],cy=ys[mid],cz=zs[mid];
    e.forEach(v=>{{
      v[0]=v[0]-cx+s.offset[0];
      v[1]=v[1]-cy+s.offset[1];
      v[2]=v[2]-cz+s.offset[2];
    }});
  }} else if(which==='overlay'){{
    const x0=s.x[0],y0=s.y[0],z0=s.z[0];
    e.forEach(v=>{{
      v[0]-=x0;
      v[1]-=y0;
      v[2]-=z0;
    }});
  }}
  return {{
    x:e.map(v=>v[0]),
    y:e.map(v=>v[1]),
    z:e.map(v=>v[2])
  }};
}}

function applyMode(which){{
  mode=which;
  const upd={{x:[],y:[],z:[]}};
  const ids=[];

  for(let k=0;k<signals.length;k++){{
    const c=coordsFor(k,which);
    const e=eventCoordsFor(k,which);

    upd.x.push(c.x); upd.y.push(c.y); upd.z.push(c.z);
    ids.push(traceMap[k][0]);

    upd.x.push(e.x); upd.y.push(e.y); upd.z.push(e.z);
    ids.push(traceMap[k][1]);
  }}

  Plotly.restyle(plot,upd,ids);
  lastI=-1;
  renderOnce();
}}

function segmentAt(t){{
  const ts=signals[active].times;
  let lo=0,hi=ts.length-1;

  while(lo<hi){{
    const m=Math.floor((lo+hi)/2);
    if(ts[m]<t)lo=m+1;
    else hi=m;
  }}

  const i=Math.max(0,lo-1);
  const j=Math.min(ts.length-1,i+1);
  const dt=ts[j]-ts[i];
  const q=dt>0?Math.max(0,Math.min(1,(t-ts[i])/dt)):0;
  return [i,j,q];
}}

function renderOnce(){{
  const s=signals[active];
  const c=coordsFor(active,mode);
  const t=audio.currentTime||0;
  const [i,j,q]=segmentAt(t);

  const x=c.x[i]+(c.x[j]-c.x[i])*q;
  const y=c.y[i]+(c.y[j]-c.y[i])*q;
  const z=c.z[i]+(c.z[j]-c.z[i])*q;

  Plotly.restyle(plot,{{x:[[x]],y:[[y]],z:[[z]]}},[movingTrace]);

  if(i!==lastI){{
    Plotly.restyle(plot,{{
      x:[c.x.slice(0,i+1).concat([x])],
      y:[c.y.slice(0,i+1).concat([y])],
      z:[c.z.slice(0,i+1).concat([z])]
    }},[playedTrace]);
    lastI=i;
  }}

  lab.textContent=`${{t.toFixed(3)}} / ${{s.duration.toFixed(3)}} s`;
}}

function loop(){{
  renderOnce();
  if(!audio.paused&&!audio.ended){{
    raf=requestAnimationFrame(loop);
  }}
}}

function selectSignal(k){{
  active=Number(k);
  audio.pause();
  audio.src=signals[active].audio;
  audio.load();
  lastI=-1;
  Plotly.restyle(plot,{{x:[[]],y:[[]],z:[[]]}},[playedTrace]);
  renderOnce();
}}

function downloadRelative(filename){{
  const a=document.createElement('a');
  a.href=filename;
  a.download=filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
}}

document.getElementById('downloadFrames').onclick=()=>downloadRelative(signals[active].frames_csv);
document.getElementById('downloadEvents').onclick=()=>downloadRelative(signals[active].events_csv);
document.getElementById('downloadSummary').onclick=()=>downloadRelative(signals[active].summary_csv);

sel.onchange=()=>selectSignal(sel.value);

audio.addEventListener('play',()=>{{
  cancelAnimationFrame(raf);
  raf=requestAnimationFrame(loop);
}});
audio.addEventListener('pause',()=>{{
  cancelAnimationFrame(raf);
  renderOnce();
}});
audio.addEventListener('seeked',()=>{{
  lastI=-1;
  renderOnce();
}});

document.getElementById('side').onclick=()=>applyMode('side');
document.getElementById('shared').onclick=()=>applyMode('shared');
document.getElementById('overlay').onclick=()=>applyMode('overlay');
document.getElementById('orbitMode').onclick=()=>Plotly.relayout(plot,{{'scene.dragmode':'orbit'}});
document.getElementById('panMode').onclick=()=>Plotly.relayout(plot,{{'scene.dragmode':'pan'}});

plot.on('plotly_click',e=>{{
  if(!e.points||!e.points.length)return;
  const cd=e.points[0].customdata;
  if(cd&&cd.length>=2){{
    selectSignal(cd[0]);
    sel.value=String(cd[0]);
    audio.currentTime=Number(cd[1]);
    audio.play();
  }}
}});

selectSignal(0);
applyMode('side');
</script>
</body>
</html>'''

    out.write_text(html,encoding='utf-8')


def choose_wavs():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root=tk.Tk()
        root.withdraw()
        root.attributes('-topmost',True)
        selected=filedialog.askopenfilenames(
            title='Select WAV files',
            filetypes=[('WAV audio','*.wav'),('All files','*.*')]
        )
        root.destroy()
        return [Path(p) for p in selected]
    except Exception as e:
        print(f'Could not open file picker: {e}',file=sys.stderr)
        return []


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('wav',nargs='*',type=Path)
    ap.add_argument('--fmax',type=float,default=8000)
    ap.add_argument('--max-nodes',type=int,default=40)
    ap.add_argument('--min-event-ms',type=float,default=120)
    a=ap.parse_args()

    paths=a.wav if a.wav else choose_wavs()
    if not paths:
        print('No WAV selected.')
        return 0

    cfg=Config(
        fmax=a.fmax,
        max_nodes=a.max_nodes,
        min_event_ms=a.min_event_ms
    )

    try:
        datasets=[]

        for p in paths:
            if not p.exists():
                raise FileNotFoundError(p)

            y,sr=sf.read(p,always_2d=False)
            mono=y.mean(axis=1) if getattr(y,'ndim',1)>1 else y
            dur=len(mono)/sr

            print(f'Reading: {p}')
            print(f'Sample rate: {sr} Hz | Duration: {dur:.3f} s')

            fr,X=extract(mono,sr,cfg)
            nd,_=nodes_from(fr,X,sr,cfg)

            datasets.append({
                'path':p,
                'sr':sr,
                'duration':dur,
                'fr':fr,
                'X':X,
                'nd':nd
            })

        method=shared_project(datasets,cfg)

        all_events=[]
        for d in datasets:
            add_event_coordinates(d)
            files=export_signal(d,method)

            ev=d['nd'].copy()
            ev.insert(0,'source_file',d['path'].name)
            all_events.append(ev)

            print(f"Natural nodes: {len(d['nd'])}")
            print("Created:")
            for f in files:
                print(f"  {f}")

        common_dir=datasets[0]['path'].parent

        combined_csv=common_dir/'all_signals_events_centroids_shared_space.csv'
        pd.concat(all_events,ignore_index=True).to_csv(combined_csv,index=False)

        # One-click browser download containing separate CSV files for every selection.
        # The same files also remain individually available beside the WAV files.
        zip_path=common_dir/'all_signals_per_selection_csv.zip'
        with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as zf:
            for d in datasets:
                base=d['path'].with_suffix('').name
                for suffix in (
                    '_frames_shared_space.csv',
                    '_events_centroids_shared_space.csv',
                    '_summary_shared_space.csv',
                ):
                    csv_path=common_dir/f'{base}{suffix}'
                    if csv_path.exists():
                        zf.write(csv_path,arcname=csv_path.name)

        html=common_dir/'shared_acoustic_trajectories.html'
        make_combined_html(datasets,method,html)

        print("Combined outputs:")
        print(f"  {html}")
        print(f"  {combined_csv}")
        print(f"  {zip_path}")

        try:
            webbrowser.open(html.resolve().as_uri())
        except Exception:
            pass

        return 0

    except Exception as e:
        print(f'ERROR: {e}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
