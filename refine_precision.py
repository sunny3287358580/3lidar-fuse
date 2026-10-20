#!/usr/bin/env python3
"""Accuracy-first, fixed extrinsic calibration: left->center and right->center.

Uses PCL GICP baselines, double-precision covariance GICP with IRLS+LM,
multiscale optimization, multiple starts, independent-capture transfer,
and comparable fixed-region evaluation. No left/right correspondences.
Run: OPENBLAS_NUM_THREADS=1 python3 -u refine_precision.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from refine_extrinsics import read_initial, read_pcd, discover, voxelize, transform, write_pcd, pose_difference


def rigid(t):
    t = np.array(t, dtype=np.float64, copy=True)
    u, _, vt = np.linalg.svd(t[:3, :3])
    d = np.eye(3); d[-1, -1] = np.linalg.det(u @ vt)
    t[:3, :3] = u @ d @ vt
    t[3] = [0, 0, 0, 1]
    return t


def skew_jac(p):
    a = np.zeros((len(p), 3, 6))
    a[:, 0, 1], a[:, 0, 2] = p[:, 2], -p[:, 1]
    a[:, 1, 0], a[:, 1, 2] = -p[:, 2], p[:, 0]
    a[:, 2, 0], a[:, 2, 1] = p[:, 1], -p[:, 0]
    a[:, :, 3:] = np.eye(3)
    return a


def retract(t, delta):
    d = np.eye(4)
    d[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix()
    d[:3, 3] = delta[3:]
    return d @ t


def geometry(p, voxel, k=20, epsilon=.001):
    p = voxelize(p, voxel)
    tree = cKDTree(p)
    distances, ids = tree.query(p, k=min(k, len(p)), workers=2)
    q = p[ids]; q -= q.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', q, q) / q.shape[1]
    val, vec = np.linalg.eigh(cov)
    n = vec[:, :, 0]
    cov = np.eye(3)[None] - (1-epsilon) * n[:, :, None] * n[:, None, :]
    curvature = val[:, 0] / np.maximum(val.sum(1), 1e-15)
    quality = (curvature < .08) & (distances[:, -1] < max(.3, 5*voxel))
    return {'p':p, 'tree':tree, 'cov':cov, 'n':n, 'quality':quality}


def matches(data, t, gate, trim, robust, symmetric=False):
    """Freeze correspondences/covariances in each LM inner solve."""
    pieces=[]
    for capture, (src, tgt) in data.items():
        xp = transform(src['p'], t)
        ns = src['n'] @ t[:3,:3].T
        pairs=[]
        dist, idx = tgt['tree'].query(xp, workers=2)
        pairs.append((np.arange(len(xp)),idx,dist))
        if symmetric:
            local=(tgt['p']-t[:3,3])@t[:3,:3]
            dist, idx=src['tree'].query(local,workers=2)
            pairs.append((idx,np.arange(len(tgt['p'])),dist))
        for si, ti, dist in pairs:
            valid=(dist<gate)
            if robust>0:
                valid &= src['quality'][si]&tgt['quality'][ti]
                valid &= np.abs(np.sum(ns[si]*tgt['n'][ti],axis=1))>.65
            si,ti=si[valid],ti[valid]
            if len(si)<100:raise RuntimeError(f'Too few correspondences: {capture}, {len(si)}')
            x,y=xp[si],tgt['p'][ti]
            cov=t[:3,:3]@src['cov'][si]@t[:3,:3].T+tgt['cov'][ti]
            ev,vec=np.linalg.eigh(cov)
            w=(vec/np.sqrt(np.maximum(ev,1e-12))[:,None,:])@vec.transpose(0,2,1)
            e=np.einsum('nij,nj->ni',w,x-y)
            norm=np.linalg.norm(e,axis=1)
            keep=norm<=np.quantile(norm,trim)
            x,y,w,norm=x[keep],y[keep],w[keep],norm[keep]
            # Cauchy IRLS on entire 3D Mahalanobis residual, not each component.
            scale=max(float(np.median(norm))*robust,.01)
            rw=1/np.sqrt(1+(norm/scale)**2) if robust>0 else np.ones(len(norm))
            w=w*(rw/np.sqrt(len(x)))[:,None,None]
            pieces.append((x,y,w))
    return tuple(np.concatenate([p[i] for p in pieces]) for i in range(3))


def refine(data_pyramid, seed, max_outer, trim, robust, symmetric=False, tag=''):
    t=rigid(seed);history=[]
    for voxel,gate,data in data_pyramid:
        stable=0
        print(f'  {tag} voxel={voxel:g}, gate={gate:g}',flush=True)
        for iteration in range(max_outer):
            x,y,w=matches(data,t,gate,trim,robust,symmetric)
            def residual(delta):
                r=Rotation.from_rotvec(delta[:3]).as_matrix()
                e=x@r.T+delta[3:]-y
                return np.einsum('nij,nj->ni',w,e).ravel()
            def jacobian(delta):
                rot=Rotation.from_rotvec(delta[:3]).as_matrix()
                z=x@rot.T
                angle=np.linalg.norm(delta[:3])
                a,b,c=delta[:3]
                skew=np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])
                if angle<1e-6:
                    left_j=np.eye(3)+.5*skew+skew@skew/6
                else:
                    left_j=np.eye(3)+(1-np.cos(angle))/angle**2*skew+(angle-np.sin(angle))/angle**3*(skew@skew)
                j=skew_jac(z)
                j[:,:,:3]=j[:,:,:3]@left_j
                return np.einsum('nij,njk->nik',w,j).reshape(-1,6)
            # LM solves an actual nonlinear rotation with an analytic SO(3)
            # Jacobian; the outer loop refreshes covariance/correspondences.
            fit=least_squares(residual,np.zeros(6),method='lm',jac=jacobian,
                              ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=100,
                              x_scale=np.array([.1,.1,.1,1,1,1]))
            delta=fit.x
            if not np.isfinite(delta).all():raise RuntimeError('Non-finite LM update')
            # An unexpectedly large step indicates correspondence failure.
            if np.linalg.norm(delta[:3])>np.deg2rad(3) or np.linalg.norm(delta[3:])>.15:
                raise RuntimeError(f'Excessive LM step in {tag}: {delta}')
            t=retract(t,delta)
            angle=float(np.rad2deg(np.linalg.norm(delta[:3])));shift=float(np.linalg.norm(delta[3:]))
            j=skew_jac(x)
            j=np.einsum('nij,njk->nik',w,j).reshape(-1,6)
            h=j.T@j;d=np.sqrt(np.maximum(np.diag(h),1e-15));ev=np.linalg.eigvalsh(h/np.outer(d,d))
            row={'voxel_m':voxel,'gate_m':gate,'iteration':iteration,'pairs':len(x),
                 'translation_step_m':shift,'rotation_step_deg':angle,
                 'fixed_pair_cost':float(fit.cost),'condition_scaled':float(ev[-1]/max(ev[0],1e-15)),
                 'lm_status':int(fit.status),'lm_nfev':int(fit.nfev)}
            history.append(row)
            if iteration%25==0:
                print(f'    {iteration:03d}: step {shift*1000:.4f} mm, {angle:.6f} deg, pairs {len(x)}',flush=True)
            stable=stable+1 if shift<1e-5 and angle<.0002 else 0
            # Detect nearest-neighbor limit cycles at sub-0.1mm scale rather
            # than misreporting an iteration cap as numerical convergence.
            plateau=False
            if iteration>=30:
                recent=[r for r in history[-20:] if r['voxel_m']==voxel]
                plateau=(len(recent)==20 and max(r['translation_step_m'] for r in recent)<.0001
                         and max(r['rotation_step_deg'] for r in recent)<.002
                         and np.ptp([r['fixed_pair_cost'] for r in recent])<1e-4*max(np.mean([r['fixed_pair_cost'] for r in recent]),1e-12))
            if stable>=3:break
            if plateau:break
        history[-1]['stage_stop']='small_steps' if stable>=3 else ('sub_0.1mm_plateau' if plateau else 'iteration_limit')
    return rigid(t),history


def evaluation_cache(raw,sensor,initial):
    cache={}
    for c,clouds in raw.items():
        a,b=geometry(clouds[sensor],.02),geometry(clouds['center'],.02)
        x=transform(a['p'],initial)
        dist,idx=b['tree'].query(x,workers=2)
        # Select the region once, from the supplied coarse extrinsics.
        srcmask=(dist<.25)&a['quality']&b['quality'][idx]
        dist,idx=a['tree'].query((b['p']-initial[:3,3])@initial[:3,:3],workers=2)
        tgtmask=(dist<.25)&b['quality']&a['quality'][idx]
        cache[c]=(a,b,srcmask,tgtmask)
    return cache


def evaluate(cache,t):
    rows=[]
    for capture,(a,b,sm,tm) in cache.items():
        world=transform(a['p'],t)
        for direction in ['sensor_to_center','center_to_sensor']:
            if direction=='sensor_to_center':
                dist,idx=b['tree'].query(world,workers=2)
                plane=np.abs(np.sum((world-b['p'][idx])*b['n'][idx],axis=1))
                mask=sm
            else:
                dist,idx=a['tree'].query((b['p']-t[:3,3])@t[:3,:3],workers=2)
                plane=np.abs(np.sum((b['p']-world[idx])*(a['n'][idx]@t[:3,:3].T),axis=1))
                mask=tm
            d,p=dist[mask],plane[mask]
            rows.append({'capture':capture,'direction':direction,'fixed_roi_points':len(d),
                         'nn_rmse_clipped_10cm_m':float(np.sqrt(np.mean(np.minimum(d,.1)**2))),
                         'plane_rmse_clipped_10cm_m':float(np.sqrt(np.mean(np.minimum(p,.1)**2))),
                         'plane_median_m':float(np.median(p)),
                         'nn_median_m':float(np.median(d)),
                         'within_2cm':float(np.mean(d<.02)),'within_5cm':float(np.mean(d<.05)),
                         'within_10cm':float(np.mean(d<.10)),
                         'all_points_within_5cm':float(np.mean(dist<.05))})
    # Symmetric, fixed-region score; equal weight per capture/direction.
    score=float(np.mean([r['nn_rmse_clipped_10cm_m'] for r in rows]))
    return {'score_m':score,'rows':rows}


def mean_pose(ts):
    out=np.eye(4);out[:3,3]=np.mean([t[:3,3] for t in ts],axis=0)
    out[:3,:3]=Rotation.from_matrix(np.array([rigid(t)[:3,:3] for t in ts])).mean().as_matrix()
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--output',type=Path,default=None)
    ap.add_argument('--captures',nargs='+',default=['2-1','2-2'])
    ap.add_argument('--outer',type=int,default=180)
    ap.add_argument('--starts',type=int,default=3)
    ap.add_argument('--sensors',nargs='+',choices=['left','right'],default=['left','right'])
    ap.add_argument('--skip-validation',action='store_true')
    ap.add_argument('--resume',action='store_true',help='Reuse candidates only if input files, settings and source hashes match.')
    args=ap.parse_args();root=args.root.resolve();out=args.output or root/'calibration_precision'
    if args.outer<1 or args.starts<1 or args.starts>3:
        ap.error('outer must be positive and starts must be in 1..3')
    out.mkdir(exist_ok=True,parents=True);initial=read_initial(root/'redme.txt')
    files=discover(root,args.captures);raw={c:{s:read_pcd(p) for s,p in d.items()} for c,d in files.items()}
    manifest={'input_hashes':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
              for d in files.values() for p in d.values()},
              'initial_hash':hashlib.sha256((root/'redme.txt').read_bytes()).hexdigest(),
              'script_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_hash':hashlib.sha256((root/'refine_extrinsics.py').read_bytes()).hexdigest(),
              'cpp_hash':hashlib.sha256((root/'cpp/refine_gicp.cpp').read_bytes()).hexdigest(),
              'alternative_seed_hash':hashlib.sha256((root/'calibration_refined/convergence_check.json').read_bytes()).hexdigest()
                 if (root/'calibration_refined/convergence_check.json').exists() else None,
              'outer':args.outer,'starts':args.starts,'captures':args.captures,'sensors':args.sensors}
    manifest_path=out/'cache_manifest.json'
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text())!=manifest:
            ap.error('Cannot resume: input/settings/source fingerprint differs or manifest is missing. Run without --resume or select a new output directory.')
    else:
        if list(out.glob('candidate_*.txt')) or (out/'final_report.json').exists():
            ap.error('Output directory already contains results. Use a new --output directory, or --resume with a matching manifest.')
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    start=time.time();report={'settings':{'outer':args.outer,'starts':args.starts,'captures':args.captures,
          'method':'independent side-center, PCL GICP seeds, covariance-GICP+IRLS+LM, double precision',
          'scales_m':[.04,.02,.01,0],'gates_m':[.18,.12,.08,.06],'trim':.90,'cauchy_scale_multiplier':2.,
          'covariance_normal_eigenvalue':.001,'covariance_tangent_eigenvalues':[1.,1.],
          'score':'equal capture/direction mean of fixed-ROI 10cm-clipped nearest-point RMSE'},
          'convention':'p_center = T_center_from_sensor @ homogeneous(p_sensor), meters; center fixed identity',
          'inputs':{c:{s:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for s,p in d.items()} for c,d in files.items()},
          'validation_status':'skipped' if args.skip_validation else 'capture-transfer checks; not independent ground truth',
          'versions':{'python':sys.version,'numpy':np.__version__},
          'initial':{s:t.tolist() for s,t in initial.items()},'sensors':{}}
    final={'center':np.eye(4)}
    for sensor in args.sensors:
        print(f'\n=== {sensor.upper()} -> CENTER ===',flush=True)
        np.savetxt(out/f'initial_{sensor}.txt',initial[sensor],fmt='%.12f')
        pcl=[]
        for c in raw:
            dest=out/f'pcl_{sensor}_{c}.txt'
            log=out/f'pcl_{sensor}_{c}.log'
            complete=out/f'pcl_{sensor}_{c}.complete'
            if not (args.resume and dest.exists() and complete.exists()):
                with log.open('w') as f:
                    subprocess.run([str(root/'cpp/refine_gicp'),str(files[c][sensor]),str(files[c]['center']),
                                    str(out/f'initial_{sensor}.txt'),str(dest)],stdout=f,stderr=subprocess.STDOUT,check=True)
                complete.write_text('All five GICP scales completed successfully.\n')
            pcl.append(rigid(np.loadtxt(dest)))
        cache=evaluation_cache(raw,sensor,initial[sensor])
        candidates={'initial':initial[sensor],**{f'pcl_{c}':t for c,t in zip(raw,pcl)},'pcl_mean':mean_pose(pcl)}
        pyramid=[]
        for voxel,gate in zip([.04,.02,.01,0],[.18,.12,.08,.06]):
            data={c:(geometry(clouds[sensor],voxel),geometry(clouds['center'],voxel)) for c,clouds in raw.items()}
            pyramid.append((voxel,gate,data))
        seeds=[('pcl_mean',mean_pose(pcl)),('initial',initial[sensor])]
        # Alternate PCL/plane basins are assessed, never selected by training loss alone.
        alternative=root/'calibration_refined/convergence_check.json'
        if alternative.exists():
            seeds.append(('plane_basin',np.array(json.loads(alternative.read_text())['refined'][sensor])))
        else:
            seeds.append(('pcl_first',pcl[0]))
        histories={}
        for label,seed in seeds[:args.starts]:
            existing=out/f'candidate_{sensor}_lm_{label}.txt'
            history_path=out/f'history_{sensor}_lm_{label}.json'
            if args.resume and existing.exists():
                t=rigid(np.loadtxt(existing))
                h=json.loads(history_path.read_text()) if history_path.exists() else [{'note':'Resumed completed candidate from earlier numerical-Jacobian run.'}]
            else:
                t,h=refine(pyramid,seed,args.outer,.90,2.,False,f'{sensor}/{label}')
                history_path.write_text(json.dumps(h,indent=2)+'\n')
            candidates[f'lm_{label}']=t;histories[f'lm_{label}']=h
            np.savetxt(out/f'candidate_{sensor}_lm_{label}.txt',t,fmt='%.12f')
            print(f'  {label}: eval_score={1000*evaluate(cache,t)["score_m"]:.3f} mm',flush=True)
        evaluations={label:evaluate(cache,t) for label,t in candidates.items()}
        # A planar quality filter can discard useful edge constraints in low
        # overlap. Also run classical (all-point) covariance GICP with LM.
        full_data={c:(geometry(clouds[sensor],0),geometry(clouds['center'],0)) for c,clouds in raw.items()}
        for label,gate,trim in [('gicp_lm_full',.06,1.),('gicp_lm_8cm',.08,1.),('gicp_lm_trim',.06,.95)]:
            path=out/f'candidate_{sensor}_{label}.txt'
            if args.resume and path.exists():
                t=rigid(np.loadtxt(path))
                history_path=out/f'history_{sensor}_{label}.json'
                histories[label]=json.loads(history_path.read_text()) if history_path.exists() else []
            else:
                t,h=refine([(0,gate,full_data)],mean_pose(pcl),args.outer,trim,0.,False,sensor+'/'+label)
                histories[label]=h
                np.savetxt(path,t,fmt='%.12f')
                (out/f'history_{sensor}_{label}.json').write_text(json.dumps(h,indent=2)+'\n')
            candidates[label]=t;evaluations[label]=evaluate(cache,t)
        eligible=['initial','pcl_mean']+[x for x in candidates if x.startswith(('lm_','gicp_lm_'))]
        best=min(eligible,key=lambda x:evaluations[x]['score_m'])
        final[sensor]=candidates[best]
        sensor_report={'selected':best,'refined':final[sensor].tolist(),
                       'correction_from_initial':pose_difference(initial[sensor],final[sensor]),
                       'candidates':{label:{'matrix':t.tolist(),'evaluation':evaluations[label]} for label,t in candidates.items()},
                       'optimization_history':histories,'independent':{}}
        if not args.skip_validation:
            for c in raw:
                partial=[(v,g,{c:data[c]}) for v,g,data in pyramid]
                if best.startswith('gicp_lm_'):
                    gate=.08 if best=='gicp_lm_8cm' else .06
                    trim=.95 if best=='gicp_lm_trim' else 1.
                    fit,h=refine([(0,gate,{c:full_data[c]})],final[sensor],args.outer,trim,0.,False,f'{sensor}/validation_{c}')
                else:
                    fit,h=refine(partial,final[sensor],args.outer,.90,2.,False,f'{sensor}/validation_{c}')
                sensor_report['independent'][c]={'matrix':fit.tolist(),'difference_from_joint':pose_difference(fit,final[sensor]),
                                                'transfer_evaluation':evaluate(cache,fit),'history':h}
                np.savetxt(out/f'independent_{sensor}_{c}.txt',fit,fmt='%.12f')
        report['sensors'][sensor]=sensor_report
        np.savetxt(out/f'T_center_from_{sensor}.txt',final[sensor],fmt='%.12f')
        np.savetxt(out/f'T_{sensor}_from_center.txt',np.linalg.inv(final[sensor]),fmt='%.12f')
        report['elapsed_seconds']=time.time()-start
        (out/'precision_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(f'SELECTED {sensor}: {best}, score {1000*evaluations[best]["score_m"]:.3f}mm\n{final[sensor]}',flush=True)
    for c,clouds in raw.items():
        if set(final)!=set(clouds):continue
        dest=out/c;dest.mkdir(exist_ok=True)
        write_pcd(dest/'merged_initial.pcd',clouds,initial)
        write_pcd(dest/'merged_refined.pcd',clouds,final)
        for s in clouds:write_pcd(dest/f'{s}_aligned.pcd',{s:clouds[s]},final)
    (out/'extrinsics.json').write_text(json.dumps({s:t.tolist() for s,t in final.items()},indent=2)+'\n')
    summary={'method':report['settings']['method'],'convention':report['convention'],
             'validation_status':report['validation_status'],'sensors':{}}
    for s,v in report['sensors'].items():
        rec={'selected':v['selected'],'matrix':v['refined'],'correction':v['correction_from_initial'],
             'candidates':v['candidates'],'validation':v['independent']}
        if len(v['independent'])==2:
            a,b=v['independent'].values()
            rec['validation_between_captures']=pose_difference(np.array(a['matrix']),np.array(b['matrix']))
        summary['sensors'][s]=rec
    (out/'final_report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    print(f'COMPLETE {time.time()-start:.1f}s: {out}',flush=True)


if __name__=='__main__':main()
