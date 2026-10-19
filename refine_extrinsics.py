#!/usr/bin/env python3
"""Refine fixed left/right -> center extrinsics from multiple PCD captures.

Requires numpy + scipy; no Open3D, ROS, GUI or network is needed.
Convention: p_center = T_center_from_sensor @ [p_sensor, 1].
Both captures share two rigid transforms; capture coordinates are never mixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def read_pcd(path):
    """Read XYZ from binary, ASCII, or binary_compressed PCD (arbitrary fields)."""
    with Path(path).open('rb') as f:
        h = {}
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f'Incomplete PCD header: {path}')
            words = line.decode('ascii').strip().split()
            if words and not words[0].startswith('#'):
                h[words[0]] = words[1:]
            if words and words[0] == 'DATA':
                break
        names = h['FIELDS']
        sizes = list(map(int, h['SIZE']))
        counts = list(map(int, h.get('COUNT', ['1'] * len(names))))
        n = int(h['POINTS'][0])
        fmts = [np.dtype('<' + {'F': 'f', 'I': 'i', 'U': 'u'}[t] + str(s))
                for t, s in zip(h['TYPE'], sizes)]
        dtype = np.dtype([(name, dt, (cnt,)) if cnt > 1 else (name, dt)
                          for name, dt, cnt in zip(names, fmts, counts)])
        mode = h['DATA'][0]
        if mode == 'binary':
            data = np.frombuffer(f.read(), dtype=dtype, count=n)
            xyz = np.column_stack([data[k] for k in ('x', 'y', 'z')])
        elif mode == 'ascii':
            data = np.loadtxt(f, ndmin=2)
            offsets = np.cumsum([0] + counts[:-1])
            xyz = data[:, [offsets[names.index(k)] for k in ('x', 'y', 'z')]]
        elif mode == 'binary_compressed':
            import struct
            cs, us = struct.unpack('<II', f.read(8))
            src = f.read(cs)
            out = bytearray()
            i = 0
            while i < len(src):
                c = src[i]
                i += 1
                if c < 32:
                    length = c + 1
                    out.extend(src[i:i + length])
                    i += length
                else:
                    length, offset = c >> 5, (c & 31) << 8
                    if length == 7:
                        length += src[i]
                        i += 1
                    offset += src[i]
                    i += 1
                    start = len(out) - offset - 1
                    for k in range(length + 2):
                        out.append(out[start + k])
            if len(out) != us or us != n * dtype.itemsize:
                raise ValueError(f'Invalid compressed PCD size: {path}')
            fields, offset = {}, 0
            for name, dt, size, cnt in zip(names, fmts, sizes, counts):
                fields[name] = np.frombuffer(out, dtype=dt, count=n * cnt, offset=offset)
                offset += n * size * cnt
            xyz = np.column_stack([fields[k] for k in ('x', 'y', 'z')])
        else:
            raise ValueError(f'Unsupported PCD mode: {mode}')
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def read_initial(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        words = line.split()
        if len(words) != 4:
            continue
        try:
            rows.append([float(x) for x in words])
        except ValueError:
            continue
    if len(rows) != 8:
        raise ValueError('Initial file must contain exactly two 4x4 matrices, LEFT then RIGHT.')
    matrices = {'center': np.eye(4)}
    for name, a in zip(('left', 'right'), np.asarray(rows).reshape(2, 4, 4)):
        if not np.allclose(a[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f'Invalid homogeneous matrix: {name}')
        u, s, vt = np.linalg.svd(a[:3, :3])
        if np.max(np.abs(s - 1)) > 0.01 or np.linalg.det(a[:3, :3]) < 0:
            raise ValueError(f'Initial rotation is not close to SO(3): {name}')
        a[:3, :3] = u @ vt  # restore orthogonality lost in six-decimal text
        matrices[name] = a.copy()
    return matrices


def discover(root, captures):
    result = {}
    for capture in captures:
        files = list((root / capture).glob('*.pcd'))
        selected = {}
        for name, token in [('left', '左'), ('center', '中'), ('right', '右')]:
            found = [p for p in files if token in p.name or name in p.stem.lower()]
            if len(found) != 1:
                raise ValueError(f'{capture}: expected one {name} PCD, found {found}')
            selected[name] = found[0]
        result[capture] = selected
    return result


def transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def voxelize(points, size):
    if size <= 0:
        return points.copy()
    _, inv = np.unique(np.floor(points / size).astype(np.int64), axis=0, return_inverse=True)
    count = np.bincount(inv)
    return np.column_stack([np.bincount(inv, weights=points[:, k]) / count for k in range(3)])


def surface(points, voxel, workers=1):
    p = voxelize(points, voxel)
    tree = cKDTree(p)
    k = min(20, len(p))
    if k < 6:
        raise ValueError('Too few points for surface normal estimation.')
    dist, idx = tree.query(p, k=k, workers=workers)
    near = p[idx]
    near -= near.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', near, near) / k
    val, vec = np.linalg.eigh(cov)
    curvature = val[:, 0] / np.maximum(val.sum(axis=1), 1e-15)
    planarity = (val[:, 1] - val[:, 0]) / np.maximum(val[:, 2], 1e-15)
    valid = (curvature < 0.06) & (planarity > 0.12) & (dist[:, -1] < max(0.35, 5 * voxel))
    return {'p': p, 'n': vec[:, :, 0], 'valid': valid, 'tree': tree}


def increment(matrix, delta):
    q = np.eye(4)
    q[:3, :3] = Rotation.from_rotvec(delta[:3]).as_matrix()
    q[:3, 3] = delta[3:]
    return q @ matrix


def blocks(surfaces, matrices, gate, trim, cross_side, workers):
    """Symmetric point-to-plane residuals, equal weight per capture/direction."""
    pairs = [('left', 'center'), ('right', 'center')]
    if cross_side:
        pairs.append(('left', 'right'))
    for capture, clouds in surfaces.items():
        world = {s: transform(clouds[s]['p'], matrices[s]) for s in clouds}
        normals = {s: clouds[s]['n'] @ matrices[s][:3, :3].T for s in clouds}
        for first, second in pairs:
            for src, tgt in [(first, second), (second, first)]:
                # Query in target local coordinates; no tree rebuilds per iteration.
                local = (world[src] - matrices[tgt][:3, 3]) @ matrices[tgt][:3, :3]
                dist, idx = clouds[tgt]['tree'].query(local, workers=workers)
                good = ((dist < gate) & clouds[src]['valid'] & clouds[tgt]['valid'][idx]
                        & (np.abs(np.sum(normals[src] * normals[tgt][idx], axis=1)) > 0.70))
                ids = np.flatnonzero(good)
                if len(ids) < 50:
                    continue
                x, y, n = world[src][ids], world[tgt][idx[ids]], normals[tgt][idx[ids]]
                residual = np.sum((x - y) * n, axis=1)
                keep = np.abs(residual) <= np.quantile(np.abs(residual), trim)
                x, n, residual = x[keep], n[keep], residual[keep]
                # Exact derivative for rotating target normals, left increments.
                a = np.column_stack([np.cross(x, n), n])
                jac = np.zeros((len(x), 12))
                for sensor, sign in [(src, 1), (tgt, -1)]:
                    if sensor != 'center':
                        j = 0 if sensor == 'left' else 6
                        jac[:, j:j+6] += sign * a
                yield capture, src, tgt, jac, residual


def gicp_blocks(surfaces, matrices, gate, trim, cross_side, workers):
    """Generalized ICP: use both local surface orientations, retain weak tangent constraints."""
    pairs = [('left', 'center'), ('right', 'center')]
    if cross_side:
        pairs.append(('left', 'right'))
    def pose_jacobian(p):
        j = np.zeros((len(p), 3, 6))
        j[:, 0, 1], j[:, 0, 2] = p[:, 2], -p[:, 1]
        j[:, 1, 0], j[:, 1, 2] = -p[:, 2], p[:, 0]
        j[:, 2, 0], j[:, 2, 1] = p[:, 1], -p[:, 0]
        j[:, :, 3:] = np.eye(3)
        return j
    for capture, clouds in surfaces.items():
        world = {s: transform(clouds[s]['p'], matrices[s]) for s in clouds}
        normals = {s: clouds[s]['n'] @ matrices[s][:3, :3].T for s in clouds}
        for first, second in pairs:
            for src, tgt in [(first, second), (second, first)]:
                local = (world[src] - matrices[tgt][:3, 3]) @ matrices[tgt][:3, :3]
                dist, idx = clouds[tgt]['tree'].query(local, workers=workers)
                good = ((dist < gate) & clouds[src]['valid'] & clouds[tgt]['valid'][idx]
                        & (np.abs(np.sum(normals[src] * normals[tgt][idx], axis=1)) > 0.70))
                ids = np.flatnonzero(good)
                if len(ids) < 50:
                    continue
                x, y = world[src][ids], world[tgt][idx[ids]]
                ns, nt = normals[src][ids], normals[tgt][idx[ids]]
                cov = 2 * np.eye(3)[None] - .99 * (ns[:, :, None] * ns[:, None, :] + nt[:, :, None] * nt[:, None, :])
                ev, vec = np.linalg.eigh(cov)
                whitening = (vec / np.sqrt(ev)[:, None, :]) @ vec.transpose(0, 2, 1)
                residual = np.einsum('nij,nj->ni', whitening, x-y)
                norms = np.linalg.norm(residual, axis=1)
                keep = norms <= np.quantile(norms, trim)
                x, y, whitening, residual = x[keep], y[keep], whitening[keep], residual[keep]
                jac = np.zeros((len(x), 3, 12))
                for sensor, sign, p in [(src, 1, x), (tgt, -1, y)]:
                    if sensor != 'center':
                        j = 0 if sensor == 'left' else 6
                        jac[:, :, j:j+6] += sign * pose_jacobian(p)
                jac = np.einsum('nij,njk->nik', whitening, jac)
                yield capture, src, tgt, jac, residual


def optimize(raw, initial, scales, gates, iterations, trim, cross_side, workers, verbose=True, method='plane'):
    matrices = {s: t.copy() for s, t in initial.items()}
    history = []
    for voxel, gate in zip(scales, gates):
        surfaces = {c: {s: surface(p, voxel, workers) for s, p in clouds.items()}
                    for c, clouds in raw.items()}
        if verbose:
            print(f'  voxel={voxel:g}m gate={gate:g}m', flush=True)
        for iteration in range(iterations):
            h, g, rms, counts = np.zeros((12, 12)), np.zeros(12), [], {}
            block_function = gicp_blocks if method == 'gicp' else blocks
            for capture, src, tgt, jac, residual in block_function(
                    surfaces, matrices, gate, trim, cross_side, workers):
                counts[f'{capture}:{src}->{tgt}'] = len(residual)
                if residual.ndim == 2:
                    norms = np.linalg.norm(residual, axis=1)
                    scale = max(float(np.median(norms)), .005)
                    w = np.minimum(1., 1.345 * scale / np.maximum(norms, 1e-15)) / len(norms)
                    w = np.repeat(w, 3)
                    jac, residual = jac.reshape(-1, 12), residual.ravel()
                else:
                    scale = np.clip(1.4826 * np.median(np.abs(residual - np.median(residual))), 0.004, gate / 4)
                    w = np.minimum(1., 1.345 * scale / np.maximum(np.abs(residual), 1e-15)) / len(residual)
                h += jac.T @ (jac * w[:, None])
                g += jac.T @ (residual * w)
                rms.append(float(np.sqrt(np.mean(residual ** 2))))
            if len(counts) < 4 * len(raw):
                raise RuntimeError(f'Insufficient surface overlap at voxel {voxel}: {counts}')
            d = np.sqrt(np.maximum(np.diag(h), 1e-15))
            normalized = h / np.outer(d, d)
            ev = np.linalg.eigvalsh(normalized)
            if ev[0] < 1e-7:
                raise RuntimeError('Geometry does not constrain all pose dimensions; use more diverse static surfaces.')
            delta = np.linalg.solve(h + 1e-8 * np.diag(np.diag(h)), -g)
            # Limit each iteration to 2 degrees and 10 cm, not the total correction.
            factor = 1.
            for j in (0, 6):
                factor = min(factor, np.deg2rad(2) / max(np.linalg.norm(delta[j:j+3]), 1e-15),
                             0.10 / max(np.linalg.norm(delta[j+3:j+6]), 1e-15))
            delta *= factor
            for sensor, j in [('left', 0), ('right', 6)]:
                matrices[sensor] = increment(matrices[sensor], delta[j:j+6])
            row = {'voxel_m': voxel, 'iteration': iteration, 'plane_rmse_m': float(np.mean(rms)),
                   'max_rotation_step_deg': float(np.rad2deg(max(np.linalg.norm(delta[:3]), np.linalg.norm(delta[6:9])))),
                   'max_translation_step_m': float(max(np.linalg.norm(delta[3:6]), np.linalg.norm(delta[9:12]))),
                   'normalized_hessian_condition': float(ev[-1] / ev[0]), 'correspondences': counts}
            history.append(row)
            if verbose and (iteration % 10 == 0 or iteration == iterations - 1):
                print(f"    iter={iteration:02d} objective_RMS={row['plane_rmse_m']:.6f} step={1000*row['max_translation_step_m']:.4f}mm", flush=True)
            if row['max_rotation_step_deg'] < 1e-5 and row['max_translation_step_m'] < 2e-6:
                break
    return matrices, history


def metrics(raw, initial, refined, workers):
    """Fixed evaluation samples and common-overlap comparisons; no training gates."""
    rows = []
    for capture, clouds in raw.items():
        ref = surface(clouds['center'], 0.025, workers)
        for sensor in ('left', 'right'):
            src = voxelize(clouds[sensor], 0.025)
            observations = {}
            for stage, matrices in [('initial', initial), ('refined', refined)]:
                p = transform(src, matrices[sensor])
                dist, idx = ref['tree'].query(p, workers=workers)
                plane = np.abs(np.sum((p - ref['p'][idx]) * ref['n'][idx], axis=1))
                observations[stage] = (dist, plane, ref['valid'][idx])
            # Same source points before/after, based on intersection of overlaps.
            common = ((observations['initial'][0] < .20) & (observations['refined'][0] < .20)
                      & observations['initial'][2] & observations['refined'][2])
            for stage, (dist, plane, valid) in observations.items():
                inside = dist < .20
                row = {'capture': capture, 'sensor': sensor, 'stage': stage, 'evaluation_points': len(src),
                       'overlap_5cm': float(np.mean(dist < .05)), 'overlap_10cm': float(np.mean(dist < .10)),
                       'overlap_20cm': float(np.mean(inside)), 'common_points': int(common.sum()),
                       'common_nn_rmse_m': float(np.sqrt(np.mean(dist[common] ** 2))) if common.any() else None,
                       'common_plane_median_m': float(np.median(plane[common])) if common.any() else None,
                       'common_plane_rmse_m': float(np.sqrt(np.mean(plane[common] ** 2))) if common.any() else None,
                       'common_plane_p90_m': float(np.quantile(plane[common], .90)) if common.any() else None}
                rows.append(row)
    return rows


def pose_difference(a, b):
    return {'translation_difference_m': float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            'rotation_difference_deg': float(np.rad2deg(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))}


def write_pcd(path, groups, matrices):
    """Preserve full input geometry; assign colors/index for CloudCompare inspection."""
    dt = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<u4'), ('Original_cloud_index', '<u4')])
    chunks = []
    for index, sensor, color in [(0, 'center', 0xdddddd), (1, 'left', 0xff5555), (2, 'right', 0x5588ff)]:
        if sensor not in groups:
            continue
        xyz = transform(groups[sensor], matrices[sensor])
        out = np.zeros(len(xyz), dtype=dt)
        for j, axis in enumerate(('x', 'y', 'z')):
            out[axis] = xyz[:, j]
        out['rgb'], out['Original_cloud_index'] = color, index
        chunks.append(out)
    data = np.concatenate(chunks)
    header = ('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n'
              'FIELDS x y z rgb Original_cloud_index\nSIZE 4 4 4 4 4\nTYPE F F F U U\n'
              f'COUNT 1 1 1 1 1\nWIDTH {len(data)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
              f'POINTS {len(data)}\nDATA binary\n')
    with Path(path).open('wb') as f:
        f.write(header.encode('ascii'))
        f.write(data.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--initial', type=Path, default=None)
    parser.add_argument('--captures', nargs='+', default=['2-1', '2-2'])
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--scales', nargs='+', type=float, default=[.15, .08, .04, .02])
    parser.add_argument('--gates', nargs='+', type=float, default=[.8, .4, .2, .10])
    parser.add_argument('--iterations', type=int, default=60)
    parser.add_argument('--trim', type=float, default=.85)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--method', choices=['plane', 'gicp'], default='plane')
    parser.add_argument('--cross-side', action='store_true', help='Also constrain overlapping left/right surfaces.')
    parser.add_argument('--skip-validation', action='store_true', help='Skip independent capture fits and cross-capture transfer checks.')
    args = parser.parse_args()
    if len(args.scales) != len(args.gates) or not 0.5 <= args.trim <= 1:
        parser.error('Scales/gates must have equal lengths; trim must be in [0.5, 1].')
    if any(x <= 0 for x in args.scales + args.gates) or args.iterations < 1:
        parser.error('Scales, gates and iterations must be positive.')
    start = time.time()
    args.root = args.root.resolve()
    initial_path = args.initial or args.root / 'redme.txt'
    output = args.output or args.root / 'calibration_refined'
    output.mkdir(parents=True, exist_ok=True)
    files = discover(args.root, args.captures)
    raw = {c: {s: read_pcd(p) for s, p in clouds.items()} for c, clouds in files.items()}
    initial = read_initial(initial_path)
    print('Convention: p_center = T_center_from_sensor @ p_sensor; unit: meter.', flush=True)
    for c, clouds in raw.items():
        print(c, {s: len(p) for s, p in clouds.items()}, flush=True)
    print('Joint fixed-extrinsic fit across captures:', flush=True)
    refined, history = optimize(raw, initial, args.scales, args.gates, args.iterations,
                                args.trim, args.cross_side, args.workers, method=args.method)
    np.set_printoptions(precision=10, suppress=True)
    for sensor in ('left', 'right'):
        print(f'T_center_from_{sensor}:\n{refined[sensor]}', flush=True)
        np.savetxt(output / f'T_center_from_{sensor}.txt', refined[sensor], fmt='%.12f')
        np.savetxt(output / f'T_{sensor}_from_center.txt', np.linalg.inv(refined[sensor]), fmt='%.12f')
    rows = metrics(raw, initial, refined, args.workers)
    independent = {}
    if not args.skip_validation and len(raw) > 1:
        for capture in raw:
            print(f'Independent validation fit: {capture}', flush=True)
            fitted, _ = optimize({capture: raw[capture]}, initial, args.scales, args.gates,
                                 args.iterations, args.trim, args.cross_side, args.workers, verbose=False, method=args.method)
            independent[capture] = {'transforms': {s: t.tolist() for s, t in fitted.items()},
                                    'difference_from_joint': {s: pose_difference(fitted[s], refined[s]) for s in ('left', 'right')},
                                    'transfer_metrics': metrics(raw, initial, fitted, args.workers)}
            for s in ('left', 'right'):
                print(s, independent[capture]['difference_from_joint'][s], flush=True)
    result = {'convention': 'p_center = T_center_from_sensor @ [x_sensor, y_sensor, z_sensor, 1]; meters',
              'initial_file': str(initial_path), 'initial_sha256': hashlib.sha256(initial_path.read_bytes()).hexdigest(),
              'initial': {s: t.tolist() for s, t in initial.items()},
              'refined': {s: t.tolist() for s, t in refined.items()},
              'correction': {s: pose_difference(initial[s], refined[s]) for s in ('left', 'right')},
              'input_files': {c: {s: {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                    for s, p in clouds.items()} for c, clouds in files.items()},
              'settings': {k: v for k, v in vars(args).items() if k not in ('root', 'initial', 'output')},
              'metrics': rows, 'independent_validation': independent, 'history': history,
              'elapsed_seconds': time.time() - start,
              'limitation': 'Geometric alignment estimate, not externally validated sensor accuracy. Assumes fixed mounts, meters, static scene, synchronized captures, and correct LEFT/RIGHT initial order.'}
    (output / 'calibration.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for capture, clouds in raw.items():
        dest = output / capture
        dest.mkdir(exist_ok=True)
        write_pcd(dest / 'merged_initial.pcd', clouds, initial)
        write_pcd(dest / 'merged_refined.pcd', clouds, refined)
        for sensor in ('left', 'center', 'right'):
            write_pcd(dest / f'{sensor}_aligned.pcd', {sensor: clouds[sensor]}, refined)
    print('Evaluation (same source points in common 20cm overlap):', flush=True)
    for row in rows:
        print(row['capture'], row['sensor'], row['stage'],
              f"plane median={1000*row['common_plane_median_m']:.2f}mm RMSE={1000*row['common_plane_rmse_m']:.2f}mm overlap10={100*row['overlap_10cm']:.1f}%", flush=True)
    print(f'Output: {output}; elapsed {time.time()-start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
