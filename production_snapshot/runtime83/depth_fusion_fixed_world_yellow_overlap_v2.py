"""World-fixed PCF lattice + OBBs, resolved on identical subpixel rays.

HUMAN yellow-projected-overlap revision of the preserved completed renderer.
Only P/Q color membership changes; the fixed-world ray/depth kernel, opaque
faces, camera and lattice parameters stay identical. PCF follows the user's
render_pcf.py; this is a new experimental condition format, not a replacement
for frozen training data. Depth is metric ray distance, NEVER camera-z.
"""
from dataclasses import dataclass
import numpy as np
import cv2


@dataclass
class Box:
    center: np.ndarray                 # shared fixed world, metres
    half_size: np.ndarray              # LOCAL half extents, metres
    rotation: np.ndarray               # local-to-world, columns are axes
    label: int                         # 1 source red, 2 target blue

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        self.half_size = np.asarray(self.half_size, dtype=np.float64)
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        if (self.center.shape != (3,) or self.half_size.shape != (3,)
                or self.rotation.shape != (3, 3)
                or not np.isfinite(self.center).all()
                or not np.isfinite(self.half_size).all()
                or np.any(self.half_size <= 0)
                or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(self.rotation), 1, atol=1e-7)
                or self.label not in (1, 2)):
            raise ValueError('Invalid finite metric OBB / proper rotation / label')


def camera_rays(c2w, K, width, height, aa=1):
    """Same pixel-centre AA convention as original Renderer.frame."""
    K = np.array(K, dtype=np.float64, copy=True)
    K[:2] *= aa
    K[0, 2] += (aa - 1) * .5
    K[1, 2] += (aa - 1) * .5
    y, x = np.mgrid[:height * aa, :width * aa]
    rays = np.stack(((x - K[0, 2]) / K[0, 0],
                     (y - K[1, 2]) / K[1, 1], np.ones_like(x)), -1).reshape(-1, 3)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return np.einsum('ni,ji->nj', rays, np.asarray(c2w)[:3, :3])


def field(points, ball=.05, rod=.008):
    q = points - np.floor(points + .5)
    s = q * q
    return np.minimum(np.sqrt(s.sum(-1)) - ball,
                      np.sqrt(np.minimum.reduce([s[..., 0]+s[..., 1],
                                                 s[..., 0]+s[..., 2],
                                                 s[..., 1]+s[..., 2]])) - rod)


def trace_lattice(origin_m, directions, spacing=.5, far_cells=64., steps=4096):
    if spacing <= 0:
        raise ValueError('Spacing must be positive metres')
    # Original half-cell phase, modulo applied ONLY to periodic geometry.
    origin = np.remainder(np.asarray(origin_m) / spacing + .5, 1.)
    t = np.full(len(directions), .002, dtype=np.float64)
    hit = np.zeros(len(t), bool)
    active = np.ones(len(t), bool)
    for _ in range(steps):
        ix = np.flatnonzero(active)
        if not len(ix):
            break
        d = field(origin + t[ix, None] * directions[ix])
        done = np.abs(d) < .00015
        hit[ix[done]] = True
        active[ix[done]] = False
        remain = ix[~done]
        t[remain] += np.maximum(np.abs(d[~done]) * .95, .000075)
        active[remain[t[remain] >= far_cells]] = False
    depth = np.where(hit, t * spacing, np.inf)
    return depth, active, origin, t


def box_intersections(origin, directions, box, near=.000001):
    """Exact slab OBB intersection; camera inside returns forward exit."""
    o = (np.asarray(origin) - box.center) @ box.rotation
    d = np.einsum('ni,ij->nj', directions, box.rotation)
    parallel = np.abs(d) < 1e-12
    safe = np.where(parallel, 1., d)
    a, b = (-box.half_size - o) / safe, (box.half_size - o) / safe
    low, high = np.minimum(a, b), np.maximum(a, b)
    low[parallel] = -np.inf
    high[parallel] = np.inf
    enter, leave = low.max(1), high.min(1)
    depth = np.where(enter >= near, enter, leave)
    valid = (leave >= np.maximum(enter, near)) & ~np.any(
        parallel & (np.abs(o) > box.half_size), axis=1)
    return np.where(valid, depth, np.inf)


def background(directions):
    """Original direction_rgb background and periodic-surface appearance."""
    dx, dy, dz = directions.T
    lon = np.arctan2(dx, dz)
    lat = np.arcsin(np.clip(-dy, -1, 1))
    env = (directions + 1.) * .5
    angle = np.pi / 12
    glon = np.abs(lon / angle - np.floor(lon / angle + .5)) * angle * np.cos(lat)
    glat = np.abs(lat / angle - np.floor(lat / angle + .5)) * angle
    # Angular overlay is set by render() using the actual supersampled fx.
    return env, glon, glat, lon, lat


def prepare_lattice(c2w, K, width=1280, height=704, aa=2, spacing=.5,
                    far_cells=64., steps=4096):
    """Reuse exactly one PCF trace for cases with identical camera/parameters."""
    c2w, K = np.array(c2w, float), np.array(K, float)
    directions = camera_rays(c2w, K, width, height, aa)
    result = trace_lattice(c2w[:3, 3], directions, spacing, far_cells, steps)
    return dict(c2w=c2w, K=K, parameters=(width,height,aa,spacing,far_cells,steps),
                directions=directions, trace=result)


def render(c2w, K, boxes, width=1280, height=704, aa=2, spacing=.5,
           far_cells=64., steps=4096, prepared=None, base_rgb=None):
    c2w, K = np.asarray(c2w, float), np.asarray(K, float)
    if (c2w.shape != (4, 4) or K.shape != (3, 3)
            or not isinstance(aa, int) or aa < 1 or width <= 0 or height <= 0):
        raise ValueError('Expected 4x4 c2w, 3x3 K and integer AA >=1')
    if not np.isfinite(c2w).all() or not np.isfinite(K).all():
        raise ValueError('Nonfinite camera')
    if (not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(c2w[:3, :3]), 1, atol=1e-6)
            or not np.allclose(c2w[3], [0, 0, 0, 1])):
        raise ValueError('Camera must be rigid, no scale')
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[0, 1]) > 1e-9
            or abs(K[1, 0]) > 1e-9 or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError('Original PCF pinhole model requires positive focal length and zero skew')
    origin = c2w[:3, 3]
    selected = None
    pixel_selection = None
    if base_rgb is not None:
        if prepared is not None:
            raise ValueError('Sparse background mode cannot consume a full-frame trace cache')
        if np.shape(base_rgb) != (height,width,3) or base_rgb.dtype != np.uint8:
            raise ValueError('Original native-resolution uint8 RGB background required')
        full_directions=camera_rays(c2w,K,width,height,aa)
        silhouette=np.zeros(len(full_directions),bool)
        for b in boxes:
            depth=box_intersections(origin,full_directions,b,near=spacing*.002)
            silhouette |= np.isfinite(depth) & (depth < spacing*far_cells)
        # Include ALL AA samples of each candidate pixel. Resolve exactly as
        # full rendering; reuse original PCF RGB only outside every box pixel.
        pixel_selection=silhouette.reshape(height,aa,width,aa).any(axis=(1,3))
        sample_selection=np.repeat(np.repeat(pixel_selection,aa,axis=0),aa,axis=1).ravel()
        selected=np.flatnonzero(sample_selection)
        directions=full_directions[selected]
        del full_directions
        pd,unresolved,phase,cell_t=trace_lattice(origin,directions,spacing,far_cells,steps)
    elif prepared is None:
        directions = camera_rays(c2w, K, width, height, aa)
        pd, unresolved, phase, cell_t = trace_lattice(origin, directions, spacing, far_cells, steps)
    else:
        if (prepared['parameters'] != (width,height,aa,spacing,far_cells,steps)
                or not np.array_equal(prepared['c2w'], c2w)
                or not np.array_equal(prepared['K'], K)):
            raise ValueError('Cached PCF must have identical camera, K and all render parameters')
        directions = prepared['directions']
        pd, unresolved, phase, cell_t = prepared['trace']
    bd = np.full(len(pd), np.inf)
    owner = np.full(len(pd), -1, dtype=int)
    for i, b in enumerate(boxes):
        d = box_intersections(origin, directions, b, near=spacing*.002)
        d = np.where(d < far_cells*spacing, d, np.inf)
        use = d < bd
        bd[use], owner[use] = d[use], i
    # One depth test on the SAME normalized rays and metric units.
    visible_box = (bd < pd) & np.isfinite(bd) & ~unresolved
    visible_lattice = np.isfinite(pd) & ~visible_box
    env, glon, glat, _, _ = background(directions)
    edge = .75 / (K[0, 0] * aa)
    grid = 1 - np.clip((np.minimum(glon, glat)-edge)/(edge+1e-7), 0, 1)
    grid *= np.minimum(1, np.sqrt(directions[:, 0]**2+directions[:, 2]**2)/.1)
    env += (1-env) * grid[:, None] * .55
    rgb = env.copy()
    ix = np.flatnonzero(np.isfinite(pd))
    p = phase + cell_t[ix, None] * directions[ix]
    normal = np.stack([field(p + np.eye(3)[i]*.0003) -
                       field(p - np.eye(3)[i]*.0003) for i in range(3)], 1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    diffuse = .55 + .45 * np.maximum(0, np.einsum('ni,i->n', normal, [.36, -.80, -.48]))
    q = p - np.floor(p+.5)
    sphere = np.linalg.norm(q, axis=1)-.05 < .0005
    surface = np.where(sphere[:, None], [.92, .28, .16], [.26, .29, .31]) * diffuse[:, None]
    fog = np.exp(-np.maximum(cell_t[ix]-2, 0)/8)
    rgb[ix] = env[ix]*(1-fog[:, None]) + surface*fog[:, None]
    labels = np.zeros(len(pd), np.uint8)
    bx = np.flatnonzero(visible_box)
    if len(bx):
        points = origin + bd[bx, None] * directions[bx]
        # HUMAN correction: P/Q are editing markers, so their visible projected
        # overlap stays yellow even when the two world volumes are separated.
        # Each marker is tested against the SAME PCF ray depth independently.
        # A nearer lattice still occludes the marker; no transparent PCF blend.
        for b in boxes:
            marker_depth = box_intersections(origin, directions[bx], b, near=spacing*.002)
            marker_visible = ((marker_depth < far_cells*spacing)
                              & (marker_depth < pd[bx]) & ~unresolved[bx])
            labels[bx[marker_visible]] |= b.label
        assert np.all(labels[bx] != 0)
        palette = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0.]])
        for i, b in enumerate(boxes):
            sel = owner[bx] == i
            ids = bx[sel]
            local = np.einsum('ni,ij->nj', points[sel]-b.center, b.rotation)
            face = np.argmax(np.abs(local)/b.half_size, axis=1)
            n = np.eye(3)[face] * np.sign(local[np.arange(len(local)), face])[:, None]
            n = np.einsum('ni,ji->nj', n, b.rotation)
            lighting = .65 + .35*np.maximum(0, np.einsum('ni,i->n', n, [.36, -.80, -.48]))
            # Physical edge width in local metres; never screen-space overlay.
            dist_to_faces = np.sort(np.maximum(b.half_size-np.abs(local), 0), axis=1)
            wire = dist_to_faces[:, 1] < min(b.half_size)*.045
            lighting = np.where(wire, 1., lighting)
            rgb[ids] = palette[labels[ids]] * lighting[:, None]
    shape = (height*aa, width*aa)
    samples=np.clip(rgb*255,0,255).astype(np.uint8)
    if selected is not None:
        dense=np.zeros((shape[0]*shape[1],3),np.uint8)
        dense[selected]=samples
        image=dense.reshape(*shape,3)
    else:
        image=samples.reshape(*shape,3)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA) if aa > 1 else image
    if selected is not None:
        output=base_rgb.copy()
        output[pixel_selection]=image[pixel_selection]
        image=output
        def expand(values,default):
            dense=np.full(shape[0]*shape[1],default,dtype=values.dtype)
            dense[selected]=values
            return dense.reshape(shape)
        return image,dict(pcf_depth_m=expand(pd,np.nan),box_depth_m=expand(bd,np.inf),
            box_visible=expand(visible_box,False),pcf_visible=expand(visible_lattice,False),
            labels=expand(labels,0),unresolved=expand(unresolved,False),
            depth_computed=sample_selection.reshape(shape))
    return image,dict(pcf_depth_m=pd.reshape(shape),box_depth_m=bd.reshape(shape),
        box_visible=visible_box.reshape(shape),pcf_visible=visible_lattice.reshape(shape),
        labels=labels.reshape(shape),unresolved=unresolved.reshape(shape))
