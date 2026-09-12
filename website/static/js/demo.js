// Interactive K1 demo: MuJoCo (WebAssembly) physics + the exported velocity
// policy + three.js rendering, all in the browser. Assets come from
// static/demo/, produced by `uv run export-web-demo`.
//
// Heavy dependencies (the 10 MB MuJoCo module, three.js) are imported lazily
// when the visitor presses "Load", so the rest of the page stays light.

const ASSET_DIR = './static/demo/';
const MUJOCO_BASE = 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.13.0/';
const DRACO_DECODER_PATH = 'https://cdn.jsdelivr.net/npm/three@0.186.0/examples/jsm/libs/draco/gltf/';

// ---------------------------------------------------------------------------
// Policy: sequential evaluation of the layer list written by export_web_demo.
// ---------------------------------------------------------------------------
class Policy {
  constructor(spec, blob) {
    this.layers = spec.layers;
    this.inputDim = spec.input_dim;
    this.outputDim = spec.output_dim;
    this.blob = blob;
    this.scratch = [new Float32Array(1024), new Float32Array(1024)];
  }

  static async load(spec) {
    const response = await fetch(ASSET_DIR + spec.file);
    if (!response.ok) throw new Error(`Failed to fetch ${spec.file}`);
    return new Policy(spec, new Float32Array(await response.arrayBuffer()));
  }

  run(obs) {
    let x = obs;
    let n = this.inputDim;
    let which = 0;
    for (const layer of this.layers) {
      const out = this.scratch[which];
      switch (layer.op) {
        case 'sub': case 'div': case 'add': case 'mul': {
          const c = this.blob;
          const o = layer.offset;
          for (let i = 0; i < n; i++) {
            const a = layer.tensor_first ? x[i] : c[o + i];
            const b = layer.tensor_first ? c[o + i] : x[i];
            out[i] = layer.op === 'sub' ? a - b : layer.op === 'div' ? a / b : layer.op === 'add' ? a + b : a * b;
          }
          break;
        }
        case 'gemm': {
          const w = this.blob;
          const rows = layer.out;
          const cols = layer.in;
          for (let r = 0; r < rows; r++) {
            let acc = w[layer.b_offset + r];
            const base = layer.w_offset + r * cols;
            for (let c = 0; c < cols; c++) acc += w[base + c] * x[c];
            out[r] = acc;
          }
          n = rows;
          break;
        }
        case 'elu':
          for (let i = 0; i < n; i++) out[i] = x[i] > 0 ? x[i] : layer.alpha * (Math.exp(x[i]) - 1);
          break;
        case 'relu':
          for (let i = 0; i < n; i++) out[i] = Math.max(0, x[i]);
          break;
        case 'tanh':
          for (let i = 0; i < n; i++) out[i] = Math.tanh(x[i]);
          break;
        case 'sigmoid':
          for (let i = 0; i < n; i++) out[i] = 1 / (1 + Math.exp(-x[i]));
          break;
        default:
          for (let i = 0; i < n; i++) out[i] = x[i];
      }
      x = out;
      which ^= 1;
    }
    return x.subarray(0, this.outputDim);
  }
}

// ---------------------------------------------------------------------------
// Simulation: MuJoCo model + policy, stepped at the training control rate.
// ---------------------------------------------------------------------------
class Sim {
  constructor(mujoco, model, scene, policy) {
    this.mujoco = mujoco;
    this.model = model;
    this.scene = scene;
    this.policy = policy;
    this.data = new mujoco.MjData(model);
    this.joints = scene.joints;
    this.obs = new Float32Array(policy.inputDim);
    this.lastAction = new Float32Array(this.joints.length);
    this.command = [0, 0, 0];
    this.time = 0;
    this.fallen = false;
    this.reset();
  }

  static async load(mujoco, scene, policy) {
    // MuJoCo reads the model from Emscripten's in-memory filesystem.
    const fs = mujoco.FS;
    try { fs.mkdir('/k1'); fs.mkdir('/k1/meshes'); } catch (_) { /* already there */ }
    const xml = await fetch(ASSET_DIR + scene.xml).then(r => { if (!r.ok) throw new Error('xml'); return r.text(); });
    fs.writeFile('/k1/' + scene.xml, xml);
    await Promise.all(scene.meshes.map(async path => {
      const bytes = await fetch(ASSET_DIR + path).then(r => { if (!r.ok) throw new Error(path); return r.arrayBuffer(); });
      fs.writeFile('/k1/' + path, new Uint8Array(bytes));
    }));
    const model = mujoco.MjModel.from_xml_path('/k1/' + scene.xml);
    return new Sim(mujoco, model, scene, policy);
  }

  reset() {
    const { mujoco, model, data, scene } = this;
    mujoco.mj_resetDataKeyframe(model, data, scene.keyframe_id);
    const ctrl = data.ctrl;
    for (const j of this.joints) ctrl[j.ctrl_id] = j.default;
    this.lastAction.fill(0);
    mujoco.mj_forward(model, data);
    this.time = 0;
    this.substeps = 0;
    this.fallen = false;
    this.fallTime = 0;
    this.snapshot();
  }

  // Gravity direction (0, 0, -1) expressed in the trunk frame.
  projectedGravity(out) {
    const q = this.data.xquat;
    const b = this.scene.trunk_body_id * 4;
    const w = q[b], x = q[b + 1], y = q[b + 2], z = q[b + 3];
    // R^T * (0, 0, -1) is minus the third row of R(q).
    out[0] = -2 * (x * z - w * y);
    out[1] = -2 * (y * z + w * x);
    out[2] = -(1 - 2 * (x * x + y * y));
    return out;
  }

  // Observe, run the policy and write PD targets. Runs every `decimation` physics steps.
  control() {
    const { data, scene, obs, joints } = this;
    const qpos = data.qpos, qvel = data.qvel, sensordata = data.sensordata;
    const gyro = scene.gyro_adr;
    let k = 0;
    obs[k++] = sensordata[gyro]; obs[k++] = sensordata[gyro + 1]; obs[k++] = sensordata[gyro + 2];
    const g = this.projectedGravity(this._g || (this._g = new Float64Array(3)));
    obs[k++] = g[0]; obs[k++] = g[1]; obs[k++] = g[2];
    for (const j of joints) obs[k++] = qpos[j.qpos_adr] - j.default;
    for (const j of joints) obs[k++] = qvel[j.dof_adr];
    for (let i = 0; i < joints.length; i++) obs[k++] = this.lastAction[i];
    obs[k++] = this.command[0]; obs[k++] = this.command[1]; obs[k++] = this.command[2];

    const action = this.policy.run(obs);
    const ctrl = data.ctrl;
    for (let i = 0; i < joints.length; i++) {
      const j = joints[i];
      this.lastAction[i] = action[i];
      ctrl[j.ctrl_id] = j.default + action[i] * j.action_scale;
    }

    // Upright cosine below 0.5 (tilt > 60 deg) counts as a fall; the caller resets.
    if (-g[2] < 0.5) this.fallTime += scene.control_dt; else this.fallTime = 0;
    this.fallen = this.fallTime > 0.4;
  }

  // One physics step (sim_dt). Keeps the previous body poses so the renderer can
  // interpolate between physics states instead of showing 5 ms jumps.
  substep() {
    if (this.substeps % this.scene.decimation === 0) this.control();
    this.snapshot();
    this.mujoco.mj_step(this.model, this.data);
    this.substeps++;
    this.time += this.scene.sim_dt;
  }

  snapshot() {
    const nbody = this.model.nbody;
    if (!this.prevXpos) { this.prevXpos = new Float64Array(nbody * 3); this.prevXquat = new Float64Array(nbody * 4); }
    this.prevXpos.set(this.data.xpos.subarray(0, nbody * 3));
    this.prevXquat.set(this.data.xquat.subarray(0, nbody * 4));
  }

  trunkPosition() {
    const p = this.data.xpos, b = this.scene.trunk_body_id * 3;
    return [p[b], p[b + 1], p[b + 2]];
  }
}

// ---------------------------------------------------------------------------
// Rendering: three.js meshes driven by interpolated MuJoCo body poses.
// ---------------------------------------------------------------------------
class View {
  constructor(THREE, OrbitControls, canvas, sim) {
    this.THREE = THREE;
    this.sim = sim;
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf7f7f7);
    scene.fog = new THREE.Fog(0xf7f7f7, 9, 22);
    this.scene = scene;

    // MuJoCo is z-up, three.js is y-up: everything simulated lives under this group.
    this.world = new THREE.Group();
    this.world.rotation.x = -Math.PI / 2;
    scene.add(this.world);

    this.camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.05, 60);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.enablePan = false;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 8;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.03;
    this.controls.target.set(0, 0.45, 0);
    this.camera.position.set(1.7, 0.95, 2.0);

    scene.add(new THREE.HemisphereLight(0xffffff, 0xd9d9d9, 1.35));
    const sun = new THREE.DirectionalLight(0xffffff, 1.9);
    sun.position.set(2.5, 6, 3.5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 1; sun.shadow.camera.far = 20;
    const s = 3.2;
    sun.shadow.camera.left = -s; sun.shadow.camera.right = s; sun.shadow.camera.top = s; sun.shadow.camera.bottom = -s;
    sun.shadow.bias = -0.0005;
    sun.shadow.normalBias = 0.02;
    scene.add(sun); scene.add(sun.target);
    this.sun = sun;

    // Ground with a 1 m grid drawn as a repeating texture (crisp up close, fades with distance).
    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(200, 200),
      new THREE.MeshStandardMaterial({ map: this.gridTexture(renderer), roughness: 1, metalness: 0 }));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);
    this.ground = ground;

    this.geoms = this.buildGeoms();
    this.bodies = [];
    this.tmpMatrix = new THREE.Matrix4();
    this.tmpVec = new THREE.Vector3();
    this.tmpPos = new THREE.Vector3();
    this.tmpQuat = new THREE.Quaternion();
    this.qa = new THREE.Quaternion();
    this.qb = new THREE.Quaternion();
    this.unit = new THREE.Vector3(1, 1, 1);
  }

  gridTexture(renderer) {
    const { THREE } = this;
    const size = 256;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#f0f0f0';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#c4c4c4';
    ctx.fillRect(0, 0, size, 3);
    ctx.fillRect(0, 0, 3, size);
    const texture = new THREE.CanvasTexture(canvas);
    texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(200, 200); // one tile per metre on the 200 m plane
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
    return texture;
  }

  // Full-resolution visual meshes, baked per body by export_web_demo. They are
  // posed from body frames, so the physics model only needs collision meshes.
  async loadRobot(render, GLTFLoader, DRACOLoader, dracoPath) {
    const { THREE } = this;
    const loader = new GLTFLoader();
    const draco = new DRACOLoader();
    draco.setDecoderPath(dracoPath);
    loader.setDRACOLoader(draco);
    const gltf = await loader.loadAsync(ASSET_DIR + render.glb);
    const bodyOf = new Map(render.nodes.map(n => [n.node, n.body_id]));
    const found = [];
    gltf.scene.traverse(obj => {
      if (!obj.isMesh) return;
      let node = obj;
      while (node && !bodyOf.has(node.name)) node = node.parent;
      if (!node) return;
      const geometry = obj.geometry.index ? obj.geometry.toNonIndexed() : obj.geometry;
      geometry.deleteAttribute('normal');
      geometry.computeVertexNormals(); // flat shading suits CAD parts
      const source = Array.isArray(obj.material) ? obj.material[0] : obj.material;
      // MuJoCo rgba values are display colors, not linear light: convert like the MuJoCo viewer shows them.
      const material = new THREE.MeshStandardMaterial({ color: source.color.clone().convertSRGBToLinear(), roughness: 0.55, metalness: 0.15 });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.matrixAutoUpdate = false;
      found.push({ mesh, id: bodyOf.get(node.name) });
    });
    if (!found.length) throw new Error('robot.glb has no meshes matching scene.json');
    for (const f of found) this.world.add(f.mesh);
    this.bodies = found;
    draco.dispose();
  }

  buildGeoms() {
    const { THREE, sim } = this;
    const m = sim.model;
    const type = m.geom_type, size = m.geom_size, dataid = m.geom_dataid, group = m.geom_group;
    const rgba = m.geom_rgba, matid = m.geom_matid, matRgba = m.mat_rgba;
    const gpos = m.geom_pos, gquat = m.geom_quat;
    const vert = m.mesh_vert, face = m.mesh_face, vadr = m.mesh_vertadr, fadr = m.mesh_faceadr, vnum = m.mesh_vertnum, fnum = m.mesh_facenum;
    const PLANE = 0, SPHERE = 2, CAPSULE = 3, ELLIPSOID = 4, CYLINDER = 5, BOX = 6, MESH = 7;
    const meshCache = new Map();
    const materials = new Map();
    const out = [];
    for (let g = 0; g < m.ngeom; g++) {
      if (group[g] >= 3 || type[g] === PLANE) continue; // collision-only geoms and the floor
      let geometry;
      const sx = size[g * 3], sy = size[g * 3 + 1], sz = size[g * 3 + 2];
      switch (type[g]) {
        case SPHERE: geometry = new THREE.SphereGeometry(sx, 24, 16); break;
        case CAPSULE: geometry = new THREE.CapsuleGeometry(sx, 2 * sy, 8, 24).rotateX(Math.PI / 2); break;
        case ELLIPSOID: geometry = new THREE.SphereGeometry(1, 24, 16).scale(sx, sy, sz); break;
        case CYLINDER: geometry = new THREE.CylinderGeometry(sx, sx, 2 * sy, 32).rotateX(Math.PI / 2); break;
        case BOX: geometry = new THREE.BoxGeometry(2 * sx, 2 * sy, 2 * sz); break;
        case MESH: {
          const id = dataid[g];
          if (!meshCache.has(id)) {
            const positions = vert.subarray(vadr[id] * 3, (vadr[id] + vnum[id]) * 3);
            const indices = face.subarray(fadr[id] * 3, (fadr[id] + fnum[id]) * 3);
            const indexed = new THREE.BufferGeometry();
            indexed.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
            indexed.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1));
            const flat = indexed.toNonIndexed(); // STL parts read best with flat shading
            flat.computeVertexNormals();
            indexed.dispose();
            meshCache.set(id, flat);
          }
          geometry = meshCache.get(id);
          break;
        }
        default: continue;
      }
      const c = matid[g] >= 0 ? matRgba.subarray(matid[g] * 4, matid[g] * 4 + 4) : rgba.subarray(g * 4, g * 4 + 4);
      const key = `${c[0]},${c[1]},${c[2]},${c[3]}`;
      if (!materials.has(key)) {
        materials.set(key, new THREE.MeshStandardMaterial({
          color: new THREE.Color(c[0], c[1], c[2]).convertSRGBToLinear(),
          roughness: 0.55, metalness: 0.15, transparent: c[3] < 1, opacity: c[3],
        }));
      }
      const mesh = new THREE.Mesh(geometry, materials.get(key));
      mesh.castShadow = true;
      mesh.receiveShadow = false;
      mesh.matrixAutoUpdate = false;
      this.world.add(mesh);
      const local = new THREE.Matrix4().compose(
        new THREE.Vector3(gpos[g * 3], gpos[g * 3 + 1], gpos[g * 3 + 2]),
        new THREE.Quaternion(gquat[g * 4 + 1], gquat[g * 4 + 2], gquat[g * 4 + 3], gquat[g * 4]),
        new THREE.Vector3(1, 1, 1));
      out.push({ mesh, id: g, local });
    }
    return out;
  }

  resize(width, height) {
    if (width < 2 || height < 2) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  // World pose of a body, interpolated between the previous and current physics
  // state (alpha in [0, 1)) so motion looks smooth at any frame rate.
  bodyPose(id, alpha, position, quaternion) {
    const { sim } = this;
    const p0 = sim.prevXpos, q0 = sim.prevXquat, p1 = sim.data.xpos, q1 = sim.data.xquat;
    const p = id * 3, q = id * 4;
    position.set(
      p0[p] + (p1[p] - p0[p]) * alpha,
      p0[p + 1] + (p1[p + 1] - p0[p + 1]) * alpha,
      p0[p + 2] + (p1[p + 2] - p0[p + 2]) * alpha);
    // MuJoCo quaternions are (w, x, y, z); three.js takes (x, y, z, w).
    this.qa.set(q0[q + 1], q0[q + 2], q0[q + 3], q0[q]);
    this.qb.set(q1[q + 1], q1[q + 2], q1[q + 3], q1[q]);
    quaternion.copy(this.qa).slerp(this.qb, alpha);
  }

  render(dt, alpha = 1) {
    const { sim, tmpPos, tmpQuat, tmpMatrix } = this;
    for (const { mesh, id } of this.bodies) {
      this.bodyPose(id, alpha, tmpPos, tmpQuat);
      mesh.matrix.compose(tmpPos, tmpQuat, this.unit);
      mesh.matrixWorldNeedsUpdate = true;
    }
    for (const { mesh, id, local } of this.geoms) {
      this.bodyPose(sim.model.geom_bodyid[id], alpha, tmpPos, tmpQuat);
      tmpMatrix.compose(tmpPos, tmpQuat, this.unit).multiply(local);
      mesh.matrix.copy(tmpMatrix);
      mesh.matrixWorldNeedsUpdate = true;
    }

    // Follow the trunk: move the orbit target (and the camera with it) smoothly.
    this.bodyPose(sim.scene.trunk_body_id, alpha, tmpPos, tmpQuat);
    this.tmpVec.set(tmpPos.x, tmpPos.z - 0.1, -tmpPos.y); // z-up -> y-up
    const follow = 1 - Math.exp(-dt * 6);
    const delta = this.tmpVec.sub(this.controls.target).multiplyScalar(follow);
    this.controls.target.add(delta);
    this.camera.position.add(delta);
    this.controls.update();

    this.sun.position.set(this.controls.target.x + 2.5, 6, this.controls.target.z + 3.5);
    this.sun.target.position.copy(this.controls.target);
    // Keep the (finite) ground under the robot; integer snapping keeps the grid aligned.
    this.ground.position.x = Math.round(this.controls.target.x);
    this.ground.position.z = Math.round(this.controls.target.z);
    this.renderer.render(this.scene, this.camera);
  }
}

// ---------------------------------------------------------------------------
// Input: virtual sticks, keyboard, gamepad.
// ---------------------------------------------------------------------------
class Joystick {
  constructor(element) {
    this.element = element;
    this.knob = element.querySelector('.joystick-knob');
    this.x = 0; this.y = 0;
    this.active = false;
    element.addEventListener('pointerdown', e => { this.active = true; element.setPointerCapture(e.pointerId); this.move(e); element.classList.add('active'); });
    element.addEventListener('pointermove', e => { if (this.active) this.move(e); });
    const release = () => { this.active = false; this.set(0, 0); element.classList.remove('active'); };
    element.addEventListener('pointerup', release);
    element.addEventListener('pointercancel', release);
    element.addEventListener('lostpointercapture', release);
  }
  move(e) {
    const r = this.element.getBoundingClientRect();
    const radius = r.width / 2;
    let dx = (e.clientX - (r.left + radius)) / (radius * 0.72);
    let dy = -(e.clientY - (r.top + radius)) / (radius * 0.72);
    const len = Math.hypot(dx, dy);
    if (len > 1) { dx /= len; dy /= len; }
    this.set(dx, dy);
  }
  set(x, y) {
    this.x = x; this.y = y;
    const r = this.element.getBoundingClientRect();
    const k = r.width * 0.36;
    this.knob.style.transform = `translate(${x * k}px, ${-y * k}px)`;
  }
}

class Input {
  constructor(root, limits) {
    this.limits = limits;
    this.keys = new Set();
    this.stick = new Joystick(root.querySelector('.joystick'));
    this.target = [0, 0, 0];
    this.command = [0, 0, 0];
    root.addEventListener('keydown', e => {
      if (e.target !== root && e.target.tagName === 'BUTTON') return;
      if (this.handled(e.code)) { this.keys.add(e.code); e.preventDefault(); }
    });
    root.addEventListener('keyup', e => this.keys.delete(e.code));
    root.addEventListener('blur', () => this.keys.clear());
    this.usedGamepad = false;
  }
  handled(code) {
    return ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(code);
  }
  // Axes in [-1, 1]: forward, left, turn-left (robot frame convention). The
  // stick walks forward/back and turns; keys and gamepads always go full speed.
  readSticks() {
    let fwd = this.stick.y, left = 0, turn = -this.stick.x;
    const k = this.keys;
    const kf = (k.has('KeyW') || k.has('ArrowUp') ? 1 : 0) - (k.has('KeyS') || k.has('ArrowDown') ? 1 : 0);
    const kl = (k.has('KeyA') ? 1 : 0) - (k.has('KeyD') ? 1 : 0);
    const kt = (k.has('KeyQ') || k.has('ArrowLeft') ? 1 : 0) - (k.has('KeyE') || k.has('ArrowRight') ? 1 : 0);
    fwd += kf; left += kl; turn += kt;
    const pads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (const pad of pads) {
      if (!pad || !pad.connected) continue;
      const dz = v => (Math.abs(v) < 0.12 ? 0 : v);
      const gx = dz(pad.axes[0] || 0), gy = dz(pad.axes[1] || 0), rx = dz(pad.axes[2] || 0);
      if (gx || gy || rx) { this.usedGamepad = true; fwd += -gy; left += -gx; turn += -rx; }
    }
    const clamp = v => Math.max(-1, Math.min(1, v));
    return [clamp(fwd), clamp(left), clamp(turn)];
  }
  // Map sticks to velocities within the trained ranges and slew toward them.
  update(dt) {
    const [f, l, t] = this.readSticks();
    const L = this.limits;
    this.target[0] = f >= 0 ? f * L.vx[1] : -f * L.vx[0];
    this.target[1] = l >= 0 ? l * L.vy[1] : -l * L.vy[0];
    this.target[2] = t >= 0 ? t * L.wz[1] : -t * L.wz[0];
    const a = 1 - Math.exp(-dt / 0.18);
    for (let i = 0; i < 3; i++) this.command[i] += (this.target[i] - this.command[i]) * a;
    return this.command;
  }
}

// ---------------------------------------------------------------------------
// Page wiring.
// ---------------------------------------------------------------------------
function setupDemo(root) {
  const canvas = root.querySelector('canvas');
  const startOverlay = root.querySelector('.demo-start');
  const startButton = root.querySelector('.demo-start-button');
  const status = root.querySelector('.demo-status');
  const hud = root.querySelector('.demo-hud');
  const readout = root.querySelector('.demo-readout');
  const resetButton = root.querySelector('.demo-reset');
  const debug = { started: false, steps: 0, errors: [] };
  window.k1demo = debug;

  const setStatus = (text, isError = false) => {
    status.textContent = text;
    status.classList.toggle('is-error', isError);
  };

  startButton.addEventListener('click', async () => {
    startButton.disabled = true;
    root.classList.add('loading');
    try {
      setStatus('Loading MuJoCo (WebAssembly)…');
      const [{ default: loadMujoco }, THREE, { OrbitControls }, { GLTFLoader }, { DRACOLoader }, scene] = await Promise.all([
        import('@mujoco/mujoco'),
        import('three'),
        import('three/addons/controls/OrbitControls.js'),
        import('three/addons/loaders/GLTFLoader.js'),
        import('three/addons/loaders/DRACOLoader.js'),
        fetch(ASSET_DIR + 'scene.json').then(r => { if (!r.ok) throw new Error('scene.json'); return r.json(); }),
      ]);
      const mujoco = await loadMujoco({ locateFile: path => MUJOCO_BASE + path });
      setStatus('Loading robot and policy…');
      const policy = await Policy.load(scene.policy);
      const sim = await Sim.load(mujoco, scene, policy);
      const view = new View(THREE, OrbitControls, canvas, sim);
      await view.loadRobot(scene.render, GLTFLoader, DRACOLoader, DRACO_DECODER_PATH);
      const input = new Input(root, scene.command_limits);
      run({ sim, view, input, scene, THREE });
      startOverlay.hidden = true;
      hud.hidden = false;
      root.classList.remove('loading');
      root.classList.add('running');
      root.focus({ preventScroll: true });
      debug.view = view;
      debug.started = true;
    } catch (error) {
      console.error(error);
      debug.errors.push(String(error));
      setStatus(`Could not start the demo: ${error.message || error}`, true);
      startButton.disabled = false;
      root.classList.remove('loading');
    }
  });

  function run(ctx) {
    const { sim, view, input, scene } = ctx;
    const observer = new ResizeObserver(() => {
      const r = canvas.getBoundingClientRect();
      view.resize(Math.round(r.width * view.renderer.getPixelRatio()), Math.round(r.height * view.renderer.getPixelRatio()));
    });
    observer.observe(canvas);
    let visible = true;
    new IntersectionObserver(entries => { visible = entries.some(e => e.isIntersecting); }, { threshold: 0.05 }).observe(root);

    resetButton.addEventListener('click', () => { sim.reset(); root.focus({ preventScroll: true }); });
    canvas.addEventListener('pointerdown', () => root.focus({ preventScroll: true }));

    let last = performance.now();
    let accumulator = 0;
    let fallenSince = null;
    const dt = scene.sim_dt;
    const maxSteps = 4 * scene.decimation; // at most four control steps per frame
    const frame = now => {
      requestAnimationFrame(frame);
      const elapsed = Math.min((now - last) / 1000, 0.25);
      last = now;
      if (!visible || document.hidden) { accumulator = 0; return; }

      const command = input.update(elapsed);
      sim.command = command;
      accumulator += elapsed;
      let steps = 0;
      while (accumulator >= dt && steps < maxSteps) {
        sim.substep();
        accumulator -= dt;
        steps++;
      }
      if (steps === maxSteps) accumulator = 0; // dropped frames: do not try to catch up
      debug.steps = sim.substeps;

      if (sim.fallen) {
        if (fallenSince === null) fallenSince = now;
        if (now - fallenSince > 1200) { sim.reset(); fallenSince = null; }
        setStatus('Fell over, resetting…');
      } else {
        fallenSince = null;
        setStatus(input.usedGamepad ? 'Gamepad connected' : '');
      }
      debug.trunk = sim.trunkPosition(); debug.command = command.slice(); debug.time = sim.time;
      const [vx, vy, wz] = command;
      readout.textContent = `vx ${vx.toFixed(2)} m/s   vy ${vy.toFixed(2)} m/s   yaw ${wz.toFixed(2)} rad/s`;
      view.render(elapsed, accumulator / dt);
    };
    requestAnimationFrame(frame);
  }
}

document.querySelectorAll('.demo').forEach(setupDemo);
