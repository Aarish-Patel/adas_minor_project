// 3D scene: arena, car, LiDAR, predicted path, tracked objects, camera rig.
// Simulation coordinates (x forward, y left, z up) map to three.js as (x, z, -y).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

export const LEVEL_COLORS = [0x34d399, 0xfacc15, 0xfb923c, 0xef4444];
const S2T = (x, y, z = 0) => new THREE.Vector3(x, z, -y);

function canvasTexture(size, draw, repeat) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  draw(c.getContext('2d'), size);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 8;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat, repeat); }
  return t;
}

function floorTexture() {
  return canvasTexture(1024, (g, s) => {
    const grad = g.createLinearGradient(0, 0, s, s);
    grad.addColorStop(0, '#182136'); grad.addColorStop(1, '#141c2e');
    g.fillStyle = grad; g.fillRect(0, 0, s, s);
    for (let i = 0; i < 9000; i++) {                       // fine speckle, like sealed concrete
      g.fillStyle = `rgba(255,255,255,${Math.random() * 0.025})`;
      g.fillRect(Math.random() * s, Math.random() * s, 2, 2);
    }
    const step = s / 40;                                    // 10 cm cells over a 4 m tile
    g.lineWidth = 1;
    for (let i = 0; i <= 40; i++) {
      g.strokeStyle = i % 10 === 0 ? 'rgba(120,160,255,0.20)' : 'rgba(120,160,255,0.05)';
      g.lineWidth = i % 10 === 0 ? 2.5 : 1;
      g.beginPath(); g.moveTo(i * step, 0); g.lineTo(i * step, s); g.stroke();
      g.beginPath(); g.moveTo(0, i * step); g.lineTo(s, i * step); g.stroke();
    }
  }, 15);
}

function glowTexture(color = '34,211,238') {
  return canvasTexture(256, (g, s) => {
    const r = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    r.addColorStop(0, `rgba(${color},0.22)`); r.addColorStop(0.55, `rgba(${color},0.07)`);
    r.addColorStop(1, `rgba(${color},0)`);
    g.fillStyle = r; g.fillRect(0, 0, s, s);
  });
}

function dotTexture() {
  return canvasTexture(64, (g, s) => {
    const r = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(0.5, 'rgba(255,255,255,0.85)');
    r.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = r; g.fillRect(0, 0, s, s);
  });
}

export class World3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.85;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0f1c);
    this.scene.fog = new THREE.Fog(0x111b32, 6, 22);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.18;

    this.camera = new THREE.PerspectiveCamera(52, 1, 0.02, 60);
    this.camera.position.set(-0.8, 0.5, 0.0);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.camMode = 'chase';

    this.viewCam = new THREE.PerspectiveCamera(53.6, 4 / 3, 0.02, 30);   // the car's own camera (68 deg horizontal, 4:3)
    this.camRT = new THREE.WebGLRenderTarget(640, 480, { colorSpace: THREE.SRGBColorSpace });
    this.camBuf = new Uint8Array(640 * 480 * 4);

    this.buildSky();
    this.buildLights();
    this.buildFloor();
    this.worldGroup = new THREE.Group();
    this.scene.add(this.worldGroup);
    this.dynamic = new Map();
    this.markers = [];
    this.car = null;
    this.veh = null;

    this.lidarPoints = null;
    this.trackPool = [];
    this.time = 0;
    this.level = 0;
    this.camTarget = new THREE.Vector3();
    this.camPos = new THREE.Vector3();
    this.ready = false;

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(512, 512), 0.4, 0.55, 1.15);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  buildSky() {
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide, depthWrite: false, fog: false,
      uniforms: { top: { value: new THREE.Color(0x03060f) }, mid: { value: new THREE.Color(0x0d1730) }, hor: { value: new THREE.Color(0x1d2f57) } },
      vertexShader: 'varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }',
      fragmentShader: `uniform vec3 top; uniform vec3 mid; uniform vec3 hor; varying vec3 vP;
        float h(vec3 p){ return normalize(p).y; }
        void main(){ float t = clamp(h(vP), 0.0, 1.0);
          vec3 c = mix(hor, mid, smoothstep(0.0, 0.25, t)); c = mix(c, top, smoothstep(0.2, 0.9, t));
          gl_FragColor = vec4(c, 1.0); }`,
    });
    this.sky = new THREE.Mesh(new THREE.SphereGeometry(50, 32, 16), mat);
    this.scene.add(this.sky);
  }

  buildLights() {
    this.scene.add(new THREE.HemisphereLight(0xaec3f0, 0x10172a, 0.32));
    this.sun = new THREE.DirectionalLight(0xffefd6, 1.15);
    this.sun.position.set(3, 6, 2.5);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(2048, 2048);
    const sc = this.sun.shadow.camera;
    sc.left = -3.2; sc.right = 3.2; sc.top = 3.2; sc.bottom = -3.2; sc.near = 0.5; sc.far = 18;
    this.sun.shadow.bias = -0.0004;
    this.sun.shadow.normalBias = 0.01;
    this.scene.add(this.sun);
    this.scene.add(this.sun.target);
    const rim = new THREE.DirectionalLight(0x4f74d9, 0.45);
    rim.position.set(-4, 2.5, -3);
    this.scene.add(rim);
  }

  buildFloor() {
    const geo = new THREE.PlaneGeometry(60, 60);
    const mat = new THREE.MeshStandardMaterial({ map: floorTexture(), roughness: 0.78, metalness: 0.15 });
    const floor = new THREE.Mesh(geo, mat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
    this.floor = floor;
  }

  // ------------------------------------------------------------------ world
  clearWorld() {
    this.worldGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material && !o.material.userData.shared) {
        (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
      }
    });
    this.worldGroup.clear();
    this.dynamic.clear();
    this.markers = [];
  }

  setWorld(desc) {
    this.clearWorld();
    this.veh = desc.vehicle;
    if (!this.car) this.buildCar();
    const foam = new THREE.MeshStandardMaterial({ color: 0xaeb8cc, roughness: 0.7, metalness: 0.02 });
    foam.userData.shared = true;
    const card = new THREE.MeshStandardMaterial({ color: 0xb98555, roughness: 0.85, metalness: 0.0 });
    card.userData.shared = true;
    const tape = new THREE.MeshStandardMaterial({ color: 0xe9d9b0, roughness: 0.6 });
    tape.userData.shared = true;

    for (const o of desc.objects) {
      if (o.k === 'wall') {
        const dx = o.x2 - o.x1, dy = o.y2 - o.y1;
        const len = Math.hypot(dx, dy);
        const m = new THREE.Mesh(new RoundedBoxGeometry(len + 0.02, o.h, 0.03, 3, 0.006), foam);
        m.position.copy(S2T((o.x1 + o.x2) / 2, (o.y1 + o.y2) / 2, o.h / 2));
        m.rotation.y = Math.atan2(dy, dx);
        m.castShadow = m.receiveShadow = true;
        this.worldGroup.add(m);
        const stripe = new THREE.Mesh(new THREE.BoxGeometry(len, 0.006, 0.032),
          new THREE.MeshBasicMaterial({ color: 0x38bdf8 }));
        stripe.position.copy(m.position); stripe.position.y = o.h + 0.001;
        stripe.rotation.y = m.rotation.y;
        this.worldGroup.add(stripe);
      } else if (o.k === 'box' && o.style === 'car') {
        const car = this.makeLeaderCar({ l: o.l, w: o.w, c: o.c });
        car.position.copy(S2T(o.cx, o.cy));
        car.rotation.y = o.hd;
        car.traverse((m) => { if (m.isMesh) m.receiveShadow = true; });
        this.worldGroup.add(car);
      } else if (o.k === 'box') {
        const g = new THREE.Group();
        const m = new THREE.Mesh(new RoundedBoxGeometry(o.l, o.h, o.w, 3, 0.008), card);
        m.position.y = o.h / 2;
        m.castShadow = m.receiveShadow = true;
        g.add(m);
        const band = new THREE.Mesh(new THREE.BoxGeometry(o.l * 0.16, o.h + 0.002, o.w + 0.002), tape);
        band.position.y = o.h / 2;
        g.add(band);
        g.position.copy(S2T(o.cx, o.cy));
        g.rotation.y = o.hd;
        this.worldGroup.add(g);
      } else if (o.k === 'cone') {
        this.worldGroup.add(this.makeCone(o));
      } else if (o.k === 'ped') {
        const p = this.makePedestrian(o.r);
        this.dynamic.set(o.id, p);
        this.worldGroup.add(p);
      } else if (o.k === 'leader') {
        const c = this.makeLeaderCar(o);
        this.dynamic.set(o.id, c);
        this.worldGroup.add(c);
      } else if (o.k === 'marker') {
        this.worldGroup.add(this.makeMarker(o));
      } else if (o.k === 'line') {
        this.worldGroup.add(this.makeLine(o));
      }
    }
  }

  makeCone(o) {
    const g = new THREE.Group();
    const h = 0.13;
    const body = new THREE.Mesh(new THREE.ConeGeometry(o.r * 1.5, h, 28),
      new THREE.MeshStandardMaterial({ color: 0xff6a1a, roughness: 0.45 }));
    body.position.y = h / 2 + 0.006;
    body.castShadow = true;
    const band = new THREE.Mesh(new THREE.CylinderGeometry(o.r * 0.86, o.r * 1.02, 0.028, 28),
      new THREE.MeshStandardMaterial({ color: 0xf5f5f5, roughness: 0.5 }));
    band.position.y = h * 0.58;
    const base = new THREE.Mesh(new THREE.BoxGeometry(o.r * 3.6, 0.012, o.r * 3.6),
      new THREE.MeshStandardMaterial({ color: 0xff6a1a, roughness: 0.5 }));
    base.position.y = 0.006;
    g.add(body, band, base);
    g.position.copy(S2T(o.x, o.y));
    return g;
  }

  makePedestrian(r) {
    const g = new THREE.Group();
    const cloth = new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.6 });
    const skin = new THREE.MeshStandardMaterial({ color: 0xf1c9a5, roughness: 0.7 });
    const body = new THREE.Mesh(new THREE.CapsuleGeometry(r * 0.62, 0.12, 6, 14), cloth);
    body.position.y = 0.095;
    body.castShadow = true;
    const head = new THREE.Mesh(new THREE.SphereGeometry(r * 0.5, 20, 16), skin);
    head.position.y = 0.205;
    head.castShadow = true;
    const ring = new THREE.Mesh(new THREE.RingGeometry(r * 1.15, r * 1.32, 40),
      new THREE.MeshBasicMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.45, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2; ring.position.y = 0.002;
    g.add(body, head, ring);
    g.userData = { body, head, phase: Math.random() * 6 };
    return g;
  }

  makeLeaderCar(o) {
    const g = new THREE.Group();
    const body = new THREE.Mesh(new RoundedBoxGeometry(o.l, 0.05, o.w, 4, 0.014),
      new THREE.MeshPhysicalMaterial({ color: new THREE.Color(o.c || '#2563eb'), roughness: 0.28, metalness: 0.5, clearcoat: 1 }));
    body.position.y = 0.045; body.castShadow = true;
    const top = new THREE.Mesh(new RoundedBoxGeometry(o.l * 0.5, 0.035, o.w * 0.8, 3, 0.012),
      new THREE.MeshPhysicalMaterial({ color: 0x1e3a8a, roughness: 0.2, metalness: 0.6 }));
    top.position.set(-o.l * 0.05, 0.088, 0); top.castShadow = true;
    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.014, o.w * 0.8),
      new THREE.MeshBasicMaterial({ color: 0xff2a2a }));
    tail.position.set(-o.l / 2, 0.05, 0);
    g.add(body, top, tail);
    for (const sx of [-1, 1]) for (const sz of [-1, 1]) {
      const w = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.022, 22),
        new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.9 }));
      w.rotation.x = Math.PI / 2;
      w.position.set(sx * o.l * 0.32, 0.028, sz * (o.w / 2));
      g.add(w);
    }
    return g;
  }

  makeMarker(o) {
    const tex = new THREE.TextureLoader().load(`markers/marker_${o.id}.png`);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.NearestFilter;
    const grp = new THREE.Group();
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(o.size * 1.25, o.size * 1.25),
      new THREE.MeshBasicMaterial({ map: tex }));
    const backing = new THREE.Mesh(new THREE.BoxGeometry(o.size * 1.3, o.size * 1.3, 0.006),
      new THREE.MeshStandardMaterial({ color: 0xf3f4f6, roughness: 0.6 }));
    backing.position.z = -0.004;
    const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.006, o.z, 12),
      new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.6, roughness: 0.4 }));
    stand.position.set(0, -o.z / 2, -0.008);
    grp.add(plane, backing, stand);
    grp.position.copy(S2T(o.x, o.y, o.z));
    grp.rotation.y = o.yaw + Math.PI / 2;      // plane faces +Z locally; face along the marker yaw
    grp.userData = { marker: o };
    this.markers.push(grp);
    return grp;
  }

  makeLine(o) {
    const g = new THREE.Group();
    const mat = new THREE.MeshBasicMaterial({ color: new THREE.Color(o.c) });
    for (let i = 0; i < o.pts.length - 1; i++) {
      const [x1, y1] = o.pts[i], [x2, y2] = o.pts[i + 1];
      const len = Math.hypot(x2 - x1, y2 - y1), ang = Math.atan2(y2 - y1, x2 - x1);
      const n = o.dashed ? Math.floor(len / 0.16) : 1;
      for (let k = 0; k < n; k++) {
        const segLen = o.dashed ? 0.09 : len;
        const t = o.dashed ? (k * 0.16 + 0.05) / len : 0.5;
        const m = new THREE.Mesh(new THREE.PlaneGeometry(segLen, o.w), mat);
        m.rotation.x = -Math.PI / 2;
        m.rotation.z = ang;
        m.position.copy(S2T(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, 0.0015));
        g.add(m);
      }
    }
    return g;
  }

  // ------------------------------------------------------------------ car
  buildCar() {
    const v = this.veh;
    const car = new THREE.Group();
    const L = v.front - v.rear, W = v.width;
    const cx = (v.front + v.rear) / 2;

    const bodyMat = new THREE.MeshPhysicalMaterial({ color: 0x1f2a3d, roughness: 0.32, metalness: 0.55, clearcoat: 0.8 });
    const body = new THREE.Mesh(new RoundedBoxGeometry(L, 0.036, W * 0.86, 5, 0.012), bodyMat);
    body.position.set(cx, 0.036, 0);
    body.castShadow = true; body.receiveShadow = true;
    const deck = new THREE.Mesh(new RoundedBoxGeometry(L * 0.78, 0.014, W * 0.7, 4, 0.006),
      new THREE.MeshPhysicalMaterial({ color: 0x334155, roughness: 0.4, metalness: 0.4 }));
    deck.position.set(cx - 0.005, 0.061, 0);
    deck.castShadow = true;
    const accent = new THREE.Mesh(new THREE.BoxGeometry(L * 0.96, 0.004, 0.006),
      new THREE.MeshStandardMaterial({ color: 0x2dd4bf, emissive: 0x2dd4bf, emissiveIntensity: 1.6 }));
    accent.position.set(cx, 0.0565, W * 0.31);
    const accent2 = accent.clone(); accent2.position.z = -W * 0.31;
    car.add(body, deck, accent, accent2);

    const tireMat = new THREE.MeshStandardMaterial({ color: 0x0b0f17, roughness: 0.95 });
    const hubMat = new THREE.MeshStandardMaterial({ color: 0xcbd5e1, metalness: 0.9, roughness: 0.25 });
    const wheel = () => {
      const g = new THREE.Group();
      const tire = new THREE.Mesh(new THREE.CylinderGeometry(0.034, 0.034, 0.026, 32), tireMat);
      tire.rotation.x = Math.PI / 2; tire.castShadow = true;
      const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.028, 20), hubMat);
      hub.rotation.x = Math.PI / 2;
      const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.004, 0.03), hubMat);
      spoke.position.z = 0;
      const spin = new THREE.Group();
      spin.add(tire, hub, spoke);
      g.add(spin); g.userData.spin = spin;
      return g;
    };
    this.wheels = { rear: [], front: [] };
    const outer = W / 2 - 0.014;
    for (const side of [-1, 1]) {
      const r = wheel(); r.position.set(0, 0.034, side * outer);
      car.add(r); this.wheels.rear.push(r);
      const pivot = new THREE.Group();
      pivot.position.set(v.wheelbase, 0.034, side * v.track / 2);
      const f = wheel(); f.position.z = side * (outer - v.track / 2);
      pivot.add(f);
      car.add(pivot);
      this.wheels.front.push({ pivot, side, wheel: f });
    }

    // LiDAR
    const lidar = new THREE.Group();
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.034, 0.02, 32),
      new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.5, metalness: 0.5 }));
    const head = new THREE.Mesh(new THREE.CylinderGeometry(0.026, 0.026, 0.02, 32),
      new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.3, metalness: 0.7 }));
    head.position.y = 0.02;
    const glow = new THREE.Mesh(new THREE.CylinderGeometry(0.0272, 0.0272, 0.004, 32),
      new THREE.MeshStandardMaterial({ color: 0x22d3ee, emissive: 0x22d3ee, emissiveIntensity: 2.4 }));
    glow.position.y = 0.024;
    const nub = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.01, 0.014),
      new THREE.MeshStandardMaterial({ color: 0x0f172a }));
    nub.position.set(0.024, 0.02, 0);
    head.add(nub);
    lidar.add(base, head, glow);
    lidar.position.set(v.lidar_x, 0.078, 0);
    lidar.traverse((o) => { if (o.isMesh) o.castShadow = true; });
    car.add(lidar);
    this.lidarHead = head;

    // Camera module on the front deck
    const cam = new THREE.Group();
    const camBody = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.02, 0.05),
      new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.4, metalness: 0.5 }));
    const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.007, 0.007, 0.006, 18),
      new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x0ea5e9, emissiveIntensity: 1.2, metalness: 1, roughness: 0.1 }));
    lens.rotation.z = Math.PI / 2; lens.position.set(0.014, 0, 0);
    cam.add(camBody, lens);
    cam.position.set(v.front - 0.035, 0.083, 0);
    car.add(cam);
    this.camMount = new THREE.Object3D();
    this.camMount.position.set(v.front - 0.02, 0.085, 0);
    car.add(this.camMount);

    // Lights
    const headMat = new THREE.MeshStandardMaterial({ color: 0xfffbe6, emissive: 0xfff2b0, emissiveIntensity: 1.3 });
    this.brakeMat = new THREE.MeshStandardMaterial({ color: 0x7f1d1d, emissive: 0xff1a1a, emissiveIntensity: 0.25 });
    for (const s of [-1, 1]) {
      const hl = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.01, 0.026), headMat);
      hl.position.set(v.front - 0.002, 0.04, s * W * 0.3);
      car.add(hl);
      const bl = new THREE.Mesh(new THREE.BoxGeometry(0.006, 0.01, 0.03), this.brakeMat);
      bl.position.set(v.rear + 0.002, 0.04, s * W * 0.3);
      car.add(bl);
    }

    for (const s of [-1, 1]) {
      const beam = new THREE.SpotLight(0xfff1c2, 0.09, 2.4, 0.42, 0.85, 1.6);
      beam.position.set(v.front, 0.05, s * W * 0.3);
      beam.target.position.set(v.front + 1.4, 0.0, s * W * 0.2);
      car.add(beam, beam.target);
    }

    // Status ring under the car
    this.aura = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.24, 64),
      new THREE.MeshBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.0, side: THREE.DoubleSide, depthWrite: false }));
    this.aura.rotation.x = -Math.PI / 2;
    this.aura.position.set(cx, 0.003, 0);
    car.add(this.aura);

    // LiDAR range rings + radar glow (fixed to the LiDAR)
    const radar = new THREE.Group();
    radar.position.set(v.lidar_x, 0.004, 0);
    const disc = new THREE.Mesh(new THREE.CircleGeometry(3.0, 64),
      new THREE.MeshBasicMaterial({ map: glowTexture(), transparent: true, opacity: 0.32, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2;
    radar.add(disc);
    for (const r of [0.2, 0.5, 1.0, 2.0]) {
      const ring = new THREE.Mesh(new THREE.RingGeometry(r - 0.003, r + 0.003, 96),
        new THREE.MeshBasicMaterial({ color: r === 0.2 ? 0xb91c1c : 0x22d3ee, transparent: true, opacity: r === 0.2 ? 0.16 : 0.10, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2;
      radar.add(ring);
    }
    car.add(radar);
    this.radar = radar;

    this.scene.add(car);
    this.car = car;

    // predicted path ribbon
    this.pathGeo = new THREE.BufferGeometry();
    this.pathGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(64 * 3), 3));
    const idx = [];
    for (let i = 0; i < 31; i++) idx.push(i * 2, i * 2 + 1, i * 2 + 2, i * 2 + 1, i * 2 + 3, i * 2 + 2);
    this.pathGeo.setIndex(idx);
    const colours = new Float32Array(64 * 4);
    this.pathGeo.setAttribute('color', new THREE.BufferAttribute(colours, 4));
    this.pathMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, side: THREE.DoubleSide, depthWrite: false });
    this.pathMesh = new THREE.Mesh(this.pathGeo, this.pathMat);
    this.pathMesh.frustumCulled = false;
    this.scene.add(this.pathMesh);

    // contact marker
    this.hit = new THREE.Group();
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.07, 40),
      new THREE.MeshBasicMaterial({ color: 0xef4444, side: THREE.DoubleSide, transparent: true, opacity: 0.9 }));
    ring.rotation.x = -Math.PI / 2;
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.22, 8),
      new THREE.MeshBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.8 }));
    beam.position.y = 0.11;
    this.hit.add(ring, beam);
    this.hit.visible = false;
    this.scene.add(this.hit);

    // LiDAR points
    const n = 900;
    this.lidarGeo = new THREE.BufferGeometry();
    this.lidarGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.lidarGeo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    this.lidarGeo.setDrawRange(0, 0);
    const pm = new THREE.PointsMaterial({ size: 0.022, vertexColors: true, map: dotTexture(), transparent: true,
      opacity: 0.95, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true });
    this.lidarPoints = new THREE.Points(this.lidarGeo, pm);
    this.lidarPoints.frustumCulled = false;
    this.scene.add(this.lidarPoints);
  }

  // ------------------------------------------------------------------ per-frame
  setLidar(flat) {
    const pos = this.lidarGeo.attributes.position, col = this.lidarGeo.attributes.color;
    const n = Math.min(flat.length / 2, pos.count);
    const cx = this.car ? this.car.position.x : 0, cz = this.car ? this.car.position.z : 0;
    const near = new THREE.Color(0xff5a3c), mid = new THREE.Color(0xfacc15), far = new THREE.Color(0x22d3ee), c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const x = flat[i * 2], z = -flat[i * 2 + 1];
      pos.setXYZ(i, x, 0.06, z);
      const d = Math.hypot(x - cx, z - cz);
      if (d < 0.6) c.copy(near).lerp(mid, d / 0.6);
      else if (d < 1.6) c.copy(mid).lerp(far, (d - 0.6) / 1.0);
      else c.copy(far);
      col.setXYZ(i, c.r, c.g, c.b);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    this.lidarGeo.setDrawRange(0, n);
  }

  trackMesh(i) {
    while (this.trackPool.length <= i) {
      const g = new THREE.Group();
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.9, 1, 40),
        new THREE.MeshBasicMaterial({ color: 0xfacc15, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = 0.008;
      const arrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0.02, 0), 0.3, 0xfacc15, 0.05, 0.035);
      g.add(ring, arrow);
      g.userData = { ring, arrow };
      this.scene.add(g);
      this.trackPool.push(g);
    }
    return this.trackPool[i];
  }

  update(snap, dt) {
    if (!this.car) return;
    this.time += dt;
    const c = snap.car;
    this.car.position.copy(S2T(c.x, c.y));
    this.car.rotation.y = c.th;
    this.level = snap.adas.level;
    const colour = LEVEL_COLORS[this.level];

    // wheels
    const spin = (c.v * dt) / 0.034;
    for (const w of this.wheels.rear) w.userData.spin.rotation.z -= spin;
    for (const f of this.wheels.front) {
      f.pivot.rotation.y = f.side < 0 ? c.wl : c.wr;      // left wheel is on the -z side
      f.wheel.userData.spin.rotation.z -= spin;
    }
    this.lidarHead.rotation.y += dt * 12;
    const braking = snap.adas.level === 3 || snap.cmd.out < -5;
    this.brakeMat.emissiveIntensity = braking ? 4.5 : 0.25;

    this.aura.material.color.setHex(colour);
    this.aura.material.opacity = this.level === 0 ? 0.0 : 0.25 + 0.15 * Math.sin(this.time * (4 + this.level * 3));
    const pc = new THREE.Color(colour);
    const col = this.pathGeo.attributes.color;
    for (let i = 0; i < 32; i++) {
      const a = (0.62 - 0.55 * (i / 31)) * (0.75 + 0.1 * this.level);
      for (const j of [i * 2, i * 2 + 1]) col.setXYZW(j, pc.r, pc.g, pc.b, a);
    }
    col.needsUpdate = true;

    // predicted path ribbon
    const path = snap.path;
    const pos = this.pathGeo.attributes.position;
    const halfW = (this.veh.width / 2);
    for (let i = 0; i < path.length; i++) {
      const [x, y] = path[i];
      const j = Math.min(i + 1, path.length - 1), k = Math.max(i - 1, 0);
      let dx = path[j][0] - path[k][0], dy = path[j][1] - path[k][1];
      const l = Math.hypot(dx, dy) || 1; dx /= l; dy /= l;
      const nx = -dy * halfW, ny = dx * halfW;
      pos.setXYZ(i * 2, x + nx, 0.004, -(y + ny));
      pos.setXYZ(i * 2 + 1, x - nx, 0.004, -(y - ny));
    }
    pos.needsUpdate = true;

    // contact marker where the predicted path meets an obstacle
    const D = snap.adas.D;
    if (D > 0 && D < 1.6 && snap.adas.mode !== 'off') {
      const last = path[path.length - 1];
      this.hit.visible = true;
      this.hit.position.set(last[0], 0.004, -last[1]);
      const pulse = 1 + 0.18 * Math.sin(this.time * 7);
      this.hit.scale.set(pulse, 1, pulse);
    } else this.hit.visible = false;

    // dynamic objects
    for (const d of snap.dyn) {
      const m = this.dynamic.get(d.id);
      if (!m) continue;
      m.position.copy(S2T(d.x, d.y));
      m.rotation.y = d.hd;
      if (m.userData.body) {
        const s = Math.sin(this.time * 9 + m.userData.phase);
        m.userData.body.position.y = 0.095 + 0.004 * Math.abs(s);
        m.userData.body.rotation.z = 0.05 * s;
      }
    }

    // tracks
    let n = 0;
    for (const t of snap.tracks) {
      if (!t.moving) continue;
      const g = this.trackMesh(n++);
      g.visible = true;
      g.position.copy(S2T(t.x, t.y));
      const r = t.r + 0.025;
      g.userData.ring.scale.set(r, r, r);
      const sp = Math.hypot(t.vx, t.vy);
      const dir = new THREE.Vector3(t.vx, 0, -t.vy).normalize();
      g.userData.arrow.setDirection(sp > 1e-3 ? dir : new THREE.Vector3(1, 0, 0));
      g.userData.arrow.setLength(Math.max(0.06, sp * 1.2), 0.05, 0.035);
      g.userData.arrow.visible = sp > 0.02;
    }
    for (let i = n; i < this.trackPool.length; i++) this.trackPool[i].visible = false;

    if (snap.lidar) this.setLidar(snap.lidar);

    // sun follows the car so shadows stay sharp
    this.sun.position.set(c.x + 3, 6, -c.y + 2.5);
    this.sun.target.position.set(c.x, 0, -c.y);

    this.updateCamera(c, dt);
  }

  // Screen position (CSS px) of a simulation point at a given height, or null if behind the camera.
  toScreen(x, y, z = 0.2) {
    const v = new THREE.Vector3(x, z, -y).project(this.camera);
    if (v.z > 1 || v.z < -1) return null;
    return [(v.x * 0.5 + 0.5) * this.canvas.clientWidth, (-v.y * 0.5 + 0.5) * this.canvas.clientHeight];
  }

  setCamMode(mode) {
    this.camMode = mode;
    this.controls.enabled = mode === 'orbit';
    if (mode === 'orbit' && this.car) {
      this.controls.target.copy(this.car.position);
    }
  }

  updateCamera(c, dt) {
    const carPos = this.car.position;
    const fwd = new THREE.Vector3(Math.cos(c.th), 0, -Math.sin(c.th));
    const k = 1 - Math.pow(0.001, dt);
    if (this.camMode === 'chase') {
      const want = carPos.clone().addScaledVector(fwd, -0.62).add(new THREE.Vector3(0, 0.3, 0));
      const look = carPos.clone().addScaledVector(fwd, 0.42).add(new THREE.Vector3(0, 0.03, 0));
      this.camPos.lerp(want, k);
      this.camTarget.lerp(look, k);
      this.camera.position.copy(this.camPos);
      this.camera.lookAt(this.camTarget);
      this.camera.fov = 58;
    } else if (this.camMode === 'top') {
      const want = carPos.clone().add(new THREE.Vector3(0, 3.4, 0.001));
      this.camPos.lerp(want, k);
      this.camTarget.lerp(carPos.clone().addScaledVector(fwd, 0.3), k);
      this.camera.position.copy(this.camPos);
      this.camera.lookAt(this.camTarget);
      this.camera.fov = 48;
    } else if (this.camMode === 'cockpit') {
      const p = new THREE.Vector3();
      this.camMount.getWorldPosition(p);
      this.camera.position.copy(p);
      this.camera.lookAt(p.clone().addScaledVector(fwd, 1.0).add(new THREE.Vector3(0, -0.06, 0)));
      this.camera.fov = 68;
    } else {
      this.controls.target.lerp(carPos, k * 0.5);
      this.controls.update();
    }
    this.camera.updateProjectionMatrix();
  }

  snapCamera() {
    if (!this.car) return;
    this.camPos.copy(this.camera.position);
    this.camTarget.copy(this.car.position);
  }

  render() {
    this.composer.render();
  }

  // Hide everything a real camera would not see (LiDAR dots, radar rings, status aura, track markers).
  setOverlays(visible, keepPath = false) {
    this.lidarPoints.visible = visible;
    this.radar.visible = visible;
    this.aura.visible = visible;
    this.hit.visible = visible && this.hit.visible;
    this.pathMesh.visible = visible || keepPath;
    for (const t of this.trackPool) t.userData.hiddenByCam = t.visible;
    if (!visible) for (const t of this.trackPool) t.visible = false;
  }

  restoreOverlays() {
    this.lidarPoints.visible = true;
    this.radar.visible = true;
    this.aura.visible = true;
    this.pathMesh.visible = true;
    for (const t of this.trackPool) t.visible = !!t.userData.hiddenByCam;
  }

  aimCarCamera() {
    const p = new THREE.Vector3(), q = new THREE.Quaternion();
    this.camMount.getWorldPosition(p);
    this.camMount.getWorldQuaternion(q);
    this.viewCam.position.copy(p);
    const look = new THREE.Vector3(1, -0.0875, 0).applyQuaternion(q);
    this.viewCam.up.set(0, 1, 0);
    this.viewCam.lookAt(p.clone().add(look));
  }

  // A 640x480 frame from the car camera as RGBA pixels (top row first), for the OpenCV detector.
  captureCarCamera() {
    if (!this.car) return null;
    this.aimCarCamera();
    this.viewCam.aspect = 640 / 480;
    this.viewCam.updateProjectionMatrix();
    const r = this.renderer;
    const hitWas = this.hit.visible;
    this.setOverlays(false);
    r.setRenderTarget(this.camRT);
    r.render(this.scene, this.viewCam);
    r.setRenderTarget(null);
    this.restoreOverlays();
    this.hit.visible = hitWas;
    r.readRenderTargetPixels(this.camRT, 0, 0, 640, 480, this.camBuf);
    const out = new Uint8ClampedArray(640 * 480 * 4);
    for (let y = 0; y < 480; y++) {
      out.set(this.camBuf.subarray((479 - y) * 640 * 4, (480 - y) * 640 * 4), y * 640 * 4);
    }
    return new ImageData(out, 640, 480);
  }

  // The car's own camera, drawn into a rectangle of the canvas (bottom-left origin, CSS pixels).
  renderCarCamera(x, y, w, h) {
    if (!this.car) return;
    this.aimCarCamera();
    this.viewCam.aspect = w / h;
    this.viewCam.updateProjectionMatrix();

    const r = this.renderer;
    const ratio = r.getPixelRatio();
    r.autoClear = false;
    r.setScissorTest(true);
    r.setViewport(x, y, w, h);
    r.setScissor(x, y, w, h);
    r.clearDepth();
    const hitWas = this.hit.visible;
    this.setOverlays(false, true);
    r.render(this.scene, this.viewCam);
    this.restoreOverlays();
    this.hit.visible = hitWas;
    r.setScissorTest(false);
    const size = r.getSize(new THREE.Vector2());
    r.setViewport(0, 0, size.x, size.y);
    r.autoClear = true;
  }
}
