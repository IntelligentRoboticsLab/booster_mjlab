"""Bake the K1 velocity policy into static assets for the in-browser demo.

The project page runs the robot entirely in the visitor's browser: MuJoCo's
official WebAssembly build steps the physics, a tiny JS evaluator runs the
policy MLP, and three.js draws the geoms. This script produces everything that
code needs from the same sources as training:

* ``k1_web.xml`` - the mjlab K1 entity (PD ``<position>`` actuators, collision
  config, keyframe) plus a floor, with the training-time solver settings.
* ``robot.glb`` - the full-resolution visual meshes, baked per body and
  Draco-compressed (needs ``npx``; falls back to an uncompressed GLB).
* ``meshes/*.stl`` - the collision meshes, decimated (MuJoCo only uses their
  convex hull) so the physics model compiles fast.
* ``policy.bin`` / ``scene.json`` - the exported ONNX MLP as raw float32
  weights, and the metadata the runtime needs (joint addresses, PD gains,
  action scale, observation layout, command limits).

Usage (``fast-simplification`` is optional; without it meshes are copied at
full resolution)::

    uv run --with fast-simplification export-web-demo path/to/policy.onnx
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnx
import trimesh
import tyro
from mjlab.entity import Entity
from onnx import numpy_helper

from booster_mjlab.robots.booster_k1.k1_constants import K1_XML, get_k1_robot_cfg

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "website" / "static" / "demo"

# Mirrors the velocity task's SimulationCfg (velocity_env_cfg.py, flat variant).
SIM_TIMESTEP = 0.005
DECIMATION = 4
SOLVER_ITERATIONS = 10
LS_ITERATIONS = 20
CCD_ITERATIONS = 60

# The flat terrain's contact params sit at the middle of the training
# randomization ranges (randomize_terrain_contact in velocity_env_cfg.py).
FLOOR_SOLREF = (0.018, 1.0)
FLOOR_SOLIMP = (0.9, 0.965, 0.0065, 0.5, 2.0)

# Joystick limits. The curriculum's final stage trains x in [-1.5, 1.75],
# y in [-1.75, 1.75] and yaw in [-1.5, 1.5]; stay a little inside that.
COMMAND_LIMITS = {"vx": (-1.0, 1.5), "vy": (-1.0, 1.0), "wz": (-1.5, 1.5)}

# Collision-mesh decimation: keep this fraction of faces, never fewer than the floor.
MESH_KEEP_FRACTION = 0.1
MESH_MIN_FACES = 1000


@dataclass(frozen=True)
class Config:
    policy: Path
    """Exported velocity policy (.onnx) with mjlab metadata."""

    output_dir: Path = DEFAULT_OUTPUT_DIR
    """Where to write the demo assets."""

    keep_fraction: float = MESH_KEEP_FRACTION
    """Fraction of triangles to keep per collision mesh when fast-simplification is installed."""


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def _attr(node: onnx.NodeProto, name: str, default):
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return default


def export_policy(onnx_path: Path, out_dir: Path) -> dict:
    """Flatten a feed-forward ONNX graph into a layer list plus a float32 blob.

    Supports the ops produced by mjlab's MLP export (input normalizer ``Sub`` /
    ``Div`` followed by ``Gemm`` + activation). Anything else raises so a new
    architecture is noticed instead of silently mis-evaluated.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    inits = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    metadata = {p.key: p.value for p in model.metadata_props}

    (graph_input,) = graph.input
    in_dim = graph_input.type.tensor_type.shape.dim[-1].dim_value
    blob = bytearray()
    layers: list[dict] = []

    def push(arr: np.ndarray) -> int:
        offset = len(blob) // 4
        blob.extend(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        return offset

    current = graph_input.name
    for node in graph.node:
        if current not in node.input:
            raise ValueError(
                f"Non-sequential graph at node {node.name} ({node.op_type})"
            )
        (output,) = node.output
        if node.op_type in ("Sub", "Div", "Add", "Mul"):
            a, b = node.input
            tensor_is_first = a == current
            const = inits[b if tensor_is_first else a].reshape(-1)
            if const.shape[0] != in_dim and const.shape[0] != 1:
                raise ValueError(f"{node.op_type} operand has shape {const.shape}")
            layers.append(
                {
                    "op": node.op_type.lower(),
                    "offset": push(np.broadcast_to(const, (in_dim,))),
                    "size": in_dim,
                    "tensor_first": tensor_is_first,
                }
            )
        elif node.op_type == "Gemm":
            x, w_name, b_name = node.input
            if x != current:
                raise ValueError("Gemm with the running tensor as a weight")
            w = inits[w_name]
            if not _attr(node, "transB", 0):
                w = w.T
            if _attr(node, "transA", 0):
                raise ValueError("transA is not supported")
            alpha = float(_attr(node, "alpha", 1.0))
            beta = float(_attr(node, "beta", 1.0))
            out_dim, k = w.shape
            if k != in_dim:
                raise ValueError(f"Gemm expects {k} inputs, running dim is {in_dim}")
            layers.append(
                {
                    "op": "gemm",
                    "in": in_dim,
                    "out": out_dim,
                    "w_offset": push(alpha * w),  # row-major (out, in)
                    "b_offset": push(beta * inits[b_name].reshape(-1)),
                }
            )
            in_dim = out_dim
        elif node.op_type == "Elu":
            layers.append({"op": "elu", "alpha": float(_attr(node, "alpha", 1.0))})
        elif node.op_type in ("Relu", "Tanh", "Sigmoid", "Identity"):
            layers.append({"op": node.op_type.lower()})
        else:
            raise ValueError(f"Unsupported op {node.op_type} in {onnx_path}")
        current = output

    (graph_output,) = graph.output
    if current != graph_output.name:
        raise ValueError("Graph output is not the last node's output")

    (out_dir / "policy.bin").write_bytes(bytes(blob))
    return {
        "file": "policy.bin",
        "input_dim": graph_input.type.tensor_type.shape.dim[-1].dim_value,
        "output_dim": in_dim,
        "layers": layers,
        "metadata": metadata,
    }


def evaluate_policy(policy: dict, blob: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Reference evaluation of the flattened graph; the JS runtime mirrors this."""
    x = obs.astype(np.float32)
    for layer in policy["layers"]:
        op = layer["op"]
        if op in ("sub", "div", "add", "mul"):
            c = blob[layer["offset"] : layer["offset"] + layer["size"]]
            if op == "sub":
                x = x - c if layer["tensor_first"] else c - x
            elif op == "div":
                x = x / c if layer["tensor_first"] else c / x
            elif op == "add":
                x = x + c
            else:
                x = x * c
        elif op == "gemm":
            w = blob[layer["w_offset"] : layer["w_offset"] + layer["in"] * layer["out"]]
            b = blob[layer["b_offset"] : layer["b_offset"] + layer["out"]]
            x = w.reshape(layer["out"], layer["in"]) @ x + b
        elif op == "elu":
            a = layer["alpha"]
            x = np.where(x > 0, x, a * (np.exp(np.minimum(x, 0)) - 1))
        elif op == "relu":
            x = np.maximum(x, 0)
        elif op == "tanh":
            x = np.tanh(x)
        elif op == "sigmoid":
            x = 1 / (1 + np.exp(-x))
    return x


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #


def _decimate(mesh: trimesh.Trimesh, keep_fraction: float) -> trimesh.Trimesh:
    try:
        import fast_simplification
    except ImportError:
        return mesh
    target = max(MESH_MIN_FACES, int(len(mesh.faces) * keep_fraction))
    if target >= len(mesh.faces):
        return mesh
    verts, faces = fast_simplification.simplify(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        target_count=target,
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _load_mesh(source_dir: Path, mesh: mujoco.MjsMesh) -> trimesh.Trimesh:
    tm = trimesh.load(source_dir / mesh.file, force="mesh")
    assert isinstance(tm, trimesh.Trimesh)
    if not np.allclose(mesh.scale, 1.0):
        tm.apply_scale(np.asarray(mesh.scale))
    return tm


def _draco_compress(glb: Path) -> None:
    """Shrink the GLB with Draco via gltf-transform (needs node/npx); best effort."""
    cmd = ["npx", "--yes", "@gltf-transform/cli@4", "draco", str(glb), str(glb)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as error:
        print(
            f"warning: Draco compression skipped ({error}); shipping uncompressed GLB"
        )


def bake_visual_meshes(
    spec: mujoco.MjSpec, source_dir: Path, out_dir: Path
) -> list[dict]:
    """Move the visual meshes out of the MJCF into a full-resolution GLB.

    Visual geoms carry no mass or contacts, so MuJoCo does not need them. Baking
    them per (body, material) in the body frame lets the page draw the original
    CAD triangles from body poses while the physics model stays small.
    """
    mesh_by_name = {m.name: m for m in spec.meshes}
    parts: dict[tuple[str, tuple[float, ...]], list[trimesh.Trimesh]] = {}
    for geom in list(spec.geoms):
        visual = geom.group < 3 and geom.contype == 0 and geom.conaffinity == 0
        if not visual or geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        tm = _load_mesh(source_dir, mesh_by_name[geom.meshname])
        transform = np.eye(4)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, np.asarray(geom.quat))
        transform[:3, :3] = rot.reshape(3, 3)
        transform[:3, 3] = geom.pos
        tm.apply_transform(transform)
        rgba = spec.material(geom.material).rgba if geom.material else geom.rgba
        parts.setdefault((geom.parent.name, tuple(float(v) for v in rgba)), []).append(
            tm
        )
        spec.delete(geom)

    scene = trimesh.Scene()
    nodes: list[dict] = []
    triangles = 0
    for (body, rgba), meshes in parts.items():
        merged = trimesh.util.concatenate(meshes)
        merged.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=rgba, metallicFactor=0.15, roughnessFactor=0.55
            )
        )
        node = f"{body}__{len(nodes)}"
        scene.add_geometry(merged, node_name=node, geom_name=node)
        nodes.append({"node": node, "body": body})
        triangles += len(merged.faces)
    glb = out_dir / "robot.glb"
    scene.export(glb)
    _draco_compress(glb)
    print(
        f"visual meshes: {triangles} triangles -> {glb.name} ({glb.stat().st_size / 1e6:.2f} MB)"
    )
    return nodes


def export_scene(
    out_dir: Path, keep_fraction: float
) -> tuple[mujoco.MjSpec, list[str], list[dict]]:
    """Write the web MJCF, its collision meshes and the visual GLB."""
    spec = Entity(get_k1_robot_cfg()).spec
    source_dir = K1_XML.parent / spec.meshdir
    render_nodes = bake_visual_meshes(spec, source_dir, out_dir)

    # Remaining mesh references are collision shapes (or primitives fitted to a
    # mesh): MuJoCo only keeps their convex hull, so a decimated copy is as good
    # as the original and far smaller.
    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    used = {g.meshname for g in spec.geoms if g.meshname}
    mesh_files: list[str] = []
    before = after = 0
    for mesh in list(spec.meshes):
        if mesh.name not in used:
            spec.delete(mesh)
            continue
        tm = _load_mesh(source_dir, mesh)
        slim = _decimate(tm, keep_fraction)
        before += len(tm.faces)
        after += len(slim.faces)
        name = f"{mesh.name}.stl"
        slim.export(mesh_dir / name)
        mesh.file = name
        mesh.scale = [1.0, 1.0, 1.0]
        mesh_files.append(name)
    spec.meshdir = "meshes"
    print(f"collision meshes: {before} -> {after} triangles")

    # The skybox texture only bloats the compiled model; the page draws its own sky.
    for texture in list(spec.textures):
        spec.delete(texture)

    floor = spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.93, 0.93, 0.93, 1],
        # Mirrors terrain_collisions(): zero friction so the robot's coefficients
        # apply, solmix=1 so the floor's compliance dominates.
        contype=1,
        conaffinity=1,
        condim=3,
        priority=0,
        friction=[0, 0, 0],
        solmix=1.0,
    )
    floor.solref[:] = FLOOR_SOLREF
    floor.solimp[:] = FLOOR_SOLIMP

    opt = spec.option
    opt.timestep = SIM_TIMESTEP
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    opt.iterations = SOLVER_ITERATIONS
    opt.ls_iterations = LS_ITERATIONS
    opt.tolerance = 1e-8
    opt.ls_tolerance = 0.01
    opt.ccd_iterations = CCD_ITERATIONS
    opt.impratio = 1.0

    (out_dir / "k1_web.xml").write_text(spec.to_xml())
    return spec, mesh_files, render_nodes


def build_scene_metadata(
    out_dir: Path, mesh_files: list[str], render_nodes: list[dict], policy: dict
) -> tuple[dict, mujoco.MjModel]:
    """Compile the exported XML and resolve every id the JS runtime needs."""
    model = mujoco.MjModel.from_xml_path(str(out_dir / "k1_web.xml"))
    md = policy["metadata"]
    joint_names = md["joint_names"].split(",")
    floats = lambda key: [float(v) for v in md[key].split(",")]  # noqa: E731
    kp, kd = floats("joint_stiffness"), floats("joint_damping")
    default, scale = floats("default_joint_pos"), floats("action_scale")

    joints = []
    for i, name in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid < 0 or aid < 0:
            raise ValueError(f"Joint/actuator {name} missing from the web model")
        # Sanity: the compiled PD gains must match what the policy was trained with.
        if not np.isclose(model.actuator_gainprm[aid, 0], kp[i]) or not np.isclose(
            -model.actuator_biasprm[aid, 2], kd[i]
        ):
            raise ValueError(f"PD gains for {name} differ between model and policy")
        joints.append(
            {
                "name": name,
                "qpos_adr": int(model.jnt_qposadr[jid]),
                "dof_adr": int(model.jnt_dofadr[jid]),
                "ctrl_id": aid,
                "default": default[i],
                "action_scale": scale[i],
                "kp": kp[i],
                "kd": kd[i],
            }
        )

    obs_names = md["observation_names"].split(",")
    expected = [
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    ]
    if obs_names != expected:
        raise ValueError(f"Observation layout {obs_names} differs from {expected}")
    history = [float(v) for v in md["observation_terms_history_length"].split(",")]
    if any(history):
        raise ValueError("Observation history is not supported by the web runtime")

    gyro = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
    if min(gyro, trunk, key) < 0:
        raise ValueError("Gyro sensor, Trunk body or init_state keyframe missing")

    for node in render_nodes:
        node["body_id"] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, node["body"]
        )
        if node["body_id"] < 0:
            raise ValueError(f"Body {node['body']} missing from the web model")

    scene = {
        "source_run": md.get("run_path", "unknown"),
        "xml": "k1_web.xml",
        "meshes": [f"meshes/{f}" for f in mesh_files],
        "render": {"glb": "robot.glb", "nodes": render_nodes},
        "sim_dt": SIM_TIMESTEP,
        "decimation": DECIMATION,
        "control_dt": SIM_TIMESTEP * DECIMATION,
        "trunk_body_id": trunk,
        "gyro_adr": int(model.sensor_adr[gyro]),
        "keyframe_id": key,
        "observation_names": obs_names,
        "command_limits": {k: list(v) for k, v in COMMAND_LIMITS.items()},
        "joints": joints,
        "policy": {k: v for k, v in policy.items() if k != "metadata"},
    }
    return scene, model


# --------------------------------------------------------------------------- #
# Verification rollout
# --------------------------------------------------------------------------- #


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world vector v into the frame of quaternion q (w, x, y, z)."""
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, v, np.array([q[0], -q[1], -q[2], -q[3]]))
    return out


def verify_rollout(
    scene: dict, model: mujoco.MjModel, blob: np.ndarray, seconds: float
) -> None:
    """Run the exported assets with plain MuJoCo + numpy and check the robot walks.

    This is the same loop the JS runtime implements, so passing here means the
    exported data is self-consistent (addresses, gains, scales, obs layout).
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, scene["keyframe_id"])
    joints = scene["joints"]
    qpos_adr = np.array([j["qpos_adr"] for j in joints])
    dof_adr = np.array([j["dof_adr"] for j in joints])
    ctrl_id = np.array([j["ctrl_id"] for j in joints])
    default = np.array([j["default"] for j in joints])
    scale = np.array([j["action_scale"] for j in joints])
    gyro = scene["gyro_adr"]
    trunk = scene["trunk_body_id"]
    command = np.array([0.6, 0.0, 0.0])
    last_action = np.zeros(len(joints), dtype=np.float32)
    data.ctrl[ctrl_id] = default
    mujoco.mj_forward(model, data)

    steps = int(seconds / scene["control_dt"])
    start = data.xpos[trunk].copy()
    for _ in range(steps):
        gravity = _quat_rotate_inverse(data.xquat[trunk], np.array([0, 0, -1.0]))
        obs = np.concatenate(
            [
                data.sensordata[gyro : gyro + 3],
                gravity,
                data.qpos[qpos_adr] - default,
                data.qvel[dof_adr],
                last_action,
                command,
            ]
        ).astype(np.float32)
        action = evaluate_policy(scene["policy"], blob, obs)
        last_action = action
        data.ctrl[ctrl_id] = default + action * scale
        for _ in range(scene["decimation"]):
            mujoco.mj_step(model, data)

    delta = data.xpos[trunk] - start
    up = -_quat_rotate_inverse(data.xquat[trunk], np.array([0, 0, -1.0]))[2]
    print(
        f"rollout {seconds:.1f}s @ {command}: moved {delta[0]:.2f} m fwd, {delta[1]:.2f} m side, up={up:.2f}"
    )
    if up < 0.9 or delta[0] < 0.35 * seconds * command[0]:
        raise RuntimeError(
            "Verification rollout failed: the robot fell or did not track the command"
        )


def run(cfg: Config) -> None:
    out_dir = cfg.output_dir
    if out_dir.exists():
        shutil.rmtree(out_dir / "meshes", ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = export_policy(cfg.policy, out_dir)
    _, mesh_files, render_nodes = export_scene(out_dir, cfg.keep_fraction)
    scene, model = build_scene_metadata(out_dir, mesh_files, render_nodes, policy)
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=1))

    blob = np.frombuffer((out_dir / "policy.bin").read_bytes(), dtype=np.float32)
    verify_rollout(scene, model, blob, seconds=4.0)

    total = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"wrote {out_dir} ({total / 1e6:.1f} MB)")


def main() -> None:
    run(tyro.cli(Config))


if __name__ == "__main__":
    main()
