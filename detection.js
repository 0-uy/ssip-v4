/**
 * detection.js — SSIP v4.2
 * ─────────────────────────────────────────────────────────────────────────
 * Modelos:
 *   yolov8n-pose.onnx  → keypoints 17 puntos por persona (hasta 8)
 *   yolov8n.onnx       → 80 clases COCO (hasta 20 objetos)
 *
 * ═══ ESCENARIOS DETECTADOS ═══════════════════════════════════════════════
 *
 * ── Existentes (v4.1) ────────────────────────────────────────────────────
 *  Z1  Mano en zona de alerta               → LOW
 *  Z2  Permanencia en zona (dwellTime s)    → HIGH
 *  Z3  Mano sale rápido hacia torso         → HIGH  "OBJETO OCULTADO"
 *  P1  Mano en bolsillo / no visible        → HIGH  (12 frames seguidos)
 *  P2  Brazos cruzados + muñeca oculta      → HIGH  (15 frames seguidos)
 *  O1  Mano toca objeto en zona             → HIGH
 *  O2  Objeto tomado (contacto >800ms)      → HIGH
 *
 * ── Nuevos (v4.2) ────────────────────────────────────────────────────────
 *  A   CAJA → BOLSILLO/MANGA
 *      Mano sale de zona de pago → en 2s va a cadera (bolsillo)
 *      o sube por encima del codo (manga). Gate: solo zona type='pago'.
 *
 *  B   PRODUCTO A LA MANGA
 *      Post-contacto: muñeca sube >7% por encima del codo en 3s.
 *      Gate: postContact activo + elbowY conocido.
 *
 *  C   BAG STUFFING
 *      Post-contacto: muñeca queda <14% de un bolso/mochila en 3s.
 *      Gate: postContact activo + BAG_CLASS presente en frame.
 *
 *  D   OCULTAMIENTO BAJO ROPA
 *      Post-contacto: muñeca en torso central + conf < 0.40 (oculta bajo ropa).
 *      Gate: postContact activo + keypoints hombro y cadera visibles.
 *
 *  E   MERODEO
 *      3+ entradas a misma zona de alerta en 90s sin pasar por caja.
 *      Gate: zone.type !== 'pago' + visitedPay === false.
 *
 *  F   TRASPASO ENTRE PERSONAS
 *      Objeto desaparece + otra persona a <22% de distancia.
 *      Gate: _detectHandoff() > 0.
 *
 *  G   ARREBATO RÁPIDO
 *      Contacto entre 300ms y 700ms + objeto desaparece.
 *      El "obj gone" normal requiere >800ms; este detecta el grab & go.
 *
 *  H   ESCANEO PREVIO
 *      Cabeza gira rápidamente izq-der (nariz X varía >0.12 en 1.5s)
 *      solo cuando la persona está cerca de un objeto de alerta o en zona.
 *      Gate: doble condición para eliminar falsos positivos (caminar).
 *
 *  I   CUERPO COMO PANTALLA
 *      Nariz no visible (conf < KP_THRESH) mientras muñeca está en zona.
 *      Indica que da la espalda a la cámara para ocultar la acción.
 *      Gate: inZone activo + nose.c < KP_THRESH.
 *
 *  J   AGACHARSE Y OCULTAR
 *      Nariz baja >15% relativa al torso durante postContact activo.
 *      Indica que se agacha para meter objeto en bolso al piso / media.
 *      Gate: postContact activo + elbowY conocido (evita false pos).
 *
 * ═══ ARQUITECTURA ════════════════════════════════════════════════════════
 *  · Máquina de estados por track (postContact, cajaExit, badges[])
 *  · Todos los checks son O(1) o O(N_objDets≤20) — sin impacto en FPS
 *  · Contadores con decay (-2 por frame limpio) → robustez ante ruido
 *  · Cooldown por clave única → sin spam
 *  · Gates explícitos → sin falsos positivos en escenarios legítimos
 */

const POSE_MODEL = './yolov8n-pose.onnx';
const OBJ_MODEL  = './yolov8n.onnx';
const INPUT_W    = 640;
const INPUT_H    = 640;
const CONF_POSE  = 0.30;
const CONF_OBJ   = 0.35;
const KP_THRESH  = 0.25;
const IOU_THRESH = 0.45;

const KP = {
  NOSE:0, L_EYE:1, R_EYE:2, L_EAR:3, R_EAR:4,
  L_SHOULDER:5, R_SHOULDER:6, L_ELBOW:7, R_ELBOW:8,
  L_WRIST:9, R_WRIST:10, L_HIP:11, R_HIP:12,
  L_KNEE:13, R_KNEE:14, L_ANKLE:15, R_ANKLE:16,
};

const BONES = [
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],
];

const OBJ_CLASSES = {
  24:'mochila', 25:'paraguas', 26:'bolso', 27:'corbata', 28:'valija',
  39:'botella', 40:'copa', 41:'taza', 42:'tenedor', 43:'cuchillo',
  44:'cuchara', 45:'tazón', 46:'banana', 47:'manzana',
  56:'silla', 57:'sofá', 63:'laptop', 64:'mouse', 65:'control',
  66:'teclado', 67:'celular', 73:'libro', 74:'reloj', 75:'jarrón', 76:'tijera',
};

const ALERT_CLASSES = new Set([24,26,28,39,41,43,63,67,73,74,75,76]);
const BAG_CLASSES   = new Set([24, 26, 28]); // mochila, bolso, valija

// Timings
const POST_CONTACT_MS = 3000;   // ventana post-contacto para detectar destino
const CAJA_HEIST_MS   = 2000;   // ventana caja-heist para detectar destino
const PROWL_WINDOW_MS = 90000;  // ventana merodeo
const PROWL_THRESH    = 3;      // entradas para considerar merodeo
const SCAN_WINDOW_MS  = 1500;   // ventana de escaneo de cabeza
const GRAB_MIN_MS     = 300;    // mínimo contacto para arrebato
const GRAB_MAX_MS     = 700;    // máximo contacto para arrebato (>700 es "tomado")

let _poseSession = null;
let _objSession  = null;
let _posePromise = null;
let _objPromise  = null;

// ── Helpers ───────────────────────────────────────────────────────────────
const _ok  = p => p && p.c >= KP_THRESH;
const _d   = (ax,ay,bx,by) => Math.hypot(ax-bx, ay-by);
const _mid = (a,b) => ({ x: (a.x+b.x)/2, y: (a.y+b.y)/2 });

export class DetectionEngine {
  constructor(canvas, zoneManager, alertManager, config = {}) {
    this.canvas       = canvas;
    this.ctx          = canvas.getContext('2d');
    this.zoneManager  = zoneManager;
    this.alertManager = alertManager;
    this.config = {
      movementThreshold: config.movementThreshold ?? 50,
      dwellTime:         config.dwellTime         ?? 3,
      cooldown:          config.cooldown          ?? 8,
    };
    this.active        = false;
    this._off          = document.createElement('canvas');
    this._off.width    = INPUT_W;
    this._off.height   = INPUT_H;
    this._offCtx       = this._off.getContext('2d', { willReadFrequently: true });
    this._tracks       = [];
    this._nextId       = 0;
    this._maxHistory   = 30;
    this._objDets      = [];
    this._interactions = {};
    this._lastAlert    = {};
    this._fpsFrames    = 0;
    this._fpsLast      = performance.now();
    this.currentFPS    = 0;
    this._renderLoopId = null;
    this._lastDets     = [];
    this.onDetection   = null;
  }

  // ══════════════════════════════════════════════════════
  //  INIT
  // ══════════════════════════════════════════════════════
  async init() {
    if (!_posePromise) _posePromise = this._loadModel(POSE_MODEL, 'pose');
    if (!_objPromise)  _objPromise  = this._loadModel(OBJ_MODEL, 'obj');
    const [poseR, objR] = await Promise.allSettled([_posePromise, _objPromise]);
    if (poseR.status === 'rejected') throw new Error('No se pudo cargar yolov8n-pose.onnx');
    if (objR.status === 'rejected')
      console.warn('%c⚠ yolov8n.onnx no disponible — detección de objetos desactivada', 'color:#ffaa00;font-weight:bold');
    this._startRenderLoop();
  }

  async _loadModel(path, name) {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime no cargado');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    for (const ep of ['webgl','wasm']) {
      try {
        const s = await ort.InferenceSession.create(path, { executionProviders:[ep], graphOptimizationLevel:'all' });
        if (name === 'pose') _poseSession = s;
        else                 _objSession  = s;
        console.log(`%c✓ YOLOv8n-${name} (${ep.toUpperCase()})`, 'color:#00e676;font-weight:bold');
        return;
      } catch(e) { console.warn(`ONNX [${name}/${ep}]:`, e.message); }
    }
    throw new Error(`No se pudo cargar ${path}`);
  }

  _startRenderLoop() {
    const loop = () => {
      if (!this.active) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.zoneManager.drawZone(false);
        this.zoneManager.drawPreview();
        if (this._lastDets.length) this._drawDetections(this._lastDets, this._objDets);
      }
      this._renderLoopId = requestAnimationFrame(loop);
    };
    this._renderLoopId = requestAnimationFrame(loop);
  }

  // ══════════════════════════════════════════════════════
  //  PIPELINE
  // ══════════════════════════════════════════════════════
  async processFrame(video) {
    if (!this.active || !_poseSession) return;
    this._fpsFrames++;
    const now = performance.now();
    if (now - this._fpsLast >= 1000) {
      this.currentFPS = this._fpsFrames; this._fpsFrames = 0; this._fpsLast = now;
    }
    let tensor, meta;
    try { [tensor, meta] = this._preprocess(video); } catch { return; }

    let poseDets = [];
    try {
      const out = await _poseSession.run({ images: tensor });
      poseDets = this._postprocessPose(out.output0 || out[Object.keys(out)[0]], meta);
    } catch(e) { console.warn('Pose:', e.message); }

    let objDets = [];
    if (_objSession) {
      try {
        const out = await _objSession.run({ images: tensor });
        objDets = this._postprocessObj(out.output0 || out[Object.keys(out)[0]], meta);
      } catch(e) { console.warn('Obj:', e.message); }
    }

    if (typeof tensor?.dispose === 'function') tensor.dispose();

    this._objDets  = objDets;   // ANTES de tracks (fix v4.1)
    this._lastDets = poseDets;
    this._updateTracks(poseDets, Date.now());
    this._render();
  }

  // ══════════════════════════════════════════════════════
  //  PREPROCESO / POSTPROCESO
  // ══════════════════════════════════════════════════════
  _preprocess(video) {
    const vw = video.videoWidth || video.width || 640;
    const vh = video.videoHeight || video.height || 480;
    const scale = Math.min(INPUT_W/vw, INPUT_H/vh);
    const nw = Math.round(vw*scale), nh = Math.round(vh*scale);
    const dx = (INPUT_W-nw)/2, dy = (INPUT_H-nh)/2;
    this._offCtx.fillStyle = '#808080';
    this._offCtx.fillRect(0,0,INPUT_W,INPUT_H);
    this._offCtx.drawImage(video, dx, dy, nw, nh);
    const px = this._offCtx.getImageData(0,0,INPUT_W,INPUT_H).data;
    const N  = INPUT_W*INPUT_H;
    const f32 = new Float32Array(3*N);
    for (let i=0; i<N; i++) {
      f32[i]     = px[i*4]   / 255;
      f32[N+i]   = px[i*4+1] / 255;
      f32[2*N+i] = px[i*4+2] / 255;
    }
    return [new ort.Tensor('float32', f32, [1,3,INPUT_H,INPUT_W]), {dx,dy,scale,vw,vh}];
  }

  _postprocessPose(output, {dx,dy,scale,vw,vh}) {
    const data=output.data, S=output.dims[2], dets=[];
    for (let i=0; i<S; i++) {
      const conf=data[4*S+i]; if (conf<CONF_POSE) continue;
      const cx=data[0*S+i], cy=data[1*S+i], bw=data[2*S+i], bh=data[3*S+i];
      const n = v => Math.max(0, Math.min(1, v));
      const nx1=n((cx-bw/2-dx)/(vw*scale)), ny1=n((cy-bh/2-dy)/(vh*scale));
      const nx2=n((cx+bw/2-dx)/(vw*scale)), ny2=n((cy+bh/2-dy)/(vh*scale));
      const kps=[];
      for (let k=0; k<17; k++) kps.push({
        x: n((data[(5+k*3)*S+i]   - dx)/(vw*scale)),
        y: n((data[(5+k*3+1)*S+i] - dy)/(vh*scale)),
        c: data[(5+k*3+2)*S+i],
      });
      dets.push({conf,kps,nx1,ny1,nx2,ny2});
    }
    return this._nms(dets).slice(0,8);
  }

  _postprocessObj(output, {dx,dy,scale,vw,vh}) {
    const data=output.data, S=output.dims[2], dets=[];
    for (let i=0; i<S; i++) {
      let bestCls=-1, bestConf=CONF_OBJ;
      for (let c=0; c<80; c++) {
        const sc=data[(4+c)*S+i];
        if (sc>bestConf) { bestConf=sc; bestCls=c; }
      }
      if (bestCls<0 || !OBJ_CLASSES[bestCls]) continue;
      const cx=data[0*S+i], cy=data[1*S+i], bw=data[2*S+i], bh=data[3*S+i];
      const n = v => Math.max(0, Math.min(1, v));
      dets.push({
        cls:bestCls, conf:bestConf, label:OBJ_CLASSES[bestCls],
        nx1:n((cx-bw/2-dx)/(vw*scale)), ny1:n((cy-bh/2-dy)/(vh*scale)),
        nx2:n((cx+bw/2-dx)/(vw*scale)), ny2:n((cy+bh/2-dy)/(vh*scale)),
      });
    }
    return this._nms(dets).slice(0,20);
  }

  _nms(dets) {
    if (!dets.length) return [];
    dets.sort((a,b) => b.conf-a.conf);
    const keep=[], drop=new Set();
    for (let i=0; i<dets.length; i++) {
      if (drop.has(i)) continue;
      keep.push(dets[i]);
      for (let j=i+1; j<dets.length; j++)
        if (!drop.has(j) && this._iou(dets[i],dets[j])>IOU_THRESH) drop.add(j);
    }
    return keep;
  }

  _iou(a,b) {
    const ix1=Math.max(a.nx1,b.nx1), iy1=Math.max(a.ny1,b.ny1);
    const ix2=Math.min(a.nx2,b.nx2), iy2=Math.min(a.ny2,b.ny2);
    const I = Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1);
    return I/((a.nx2-a.nx1)*(a.ny2-a.ny1)+(b.nx2-b.nx1)*(b.ny2-b.ny1)-I+1e-6);
  }

  // ══════════════════════════════════════════════════════
  //  TRACKING
  // ══════════════════════════════════════════════════════
  _makeTrack(d, now) {
    return {
      id: this._nextId++,
      kps: d.kps, nx1:d.nx1, ny1:d.ny1, nx2:d.nx2, ny2:d.ny2,
      missed: 0,
      history: [{kps:d.kps, t:now}],
      // Detectores base
      inZoneWrist:{}, dwellStart:{},
      pocketL:0, pocketR:0, crossedArms:0,
      // [A] Salida de zona de pago
      cajaExit:{},
      // [B][C][D][G][J] Estado post-contacto
      postContact: null,
      // [E] Merodeo
      zoneVisits:{}, visitedPay:false,
      // [H] Escaneo previo (historial de nariz X con timestamp)
      noseXHist:[],
      // [I] Cuerpo como pantalla (contador)
      bodyScreen:0,
      // [J] Agacharse
      crouchHide:0,
      // Badges para render
      badges:[],
    };
  }

  _updateTracks(dets, now) {
    const matched = new Set();
    for (const t of this._tracks) {
      let best=-1, bestIou=0.10;
      for (let i=0; i<dets.length; i++) {
        if (matched.has(i)) continue;
        const iou = this._iou(t, dets[i]);
        if (iou > bestIou) { best=i; bestIou=iou; }
      }
      if (best >= 0) {
        const d = dets[best];
        Object.assign(t, {kps:d.kps, nx1:d.nx1, ny1:d.ny1, nx2:d.nx2, ny2:d.ny2, missed:0});
        t.history.push({kps:d.kps, t:now});
        if (t.history.length > this._maxHistory) t.history.shift();
        matched.add(best);
      } else {
        t.missed = (t.missed||0) + 1;
      }
    }
    this._tracks = this._tracks.filter(t => (t.missed||0) < 10);
    for (let i=0; i<dets.length; i++)
      if (!matched.has(i)) this._tracks.push(this._makeTrack(dets[i], now));
    for (const t of this._tracks)
      if (!t.missed) this._analyze(t, now);
  }

  // ══════════════════════════════════════════════════════
  //  ANÁLISIS CENTRAL
  // ══════════════════════════════════════════════════════
  _analyze(t, now) {
    const k = t.kps;
    const lw=k[KP.L_WRIST], rw=k[KP.R_WRIST];
    const lh=k[KP.L_HIP],   rh=k[KP.R_HIP];
    const le=k[KP.L_ELBOW], re=k[KP.R_ELBOW];
    const ls=k[KP.L_SHOULDER], rs=k[KP.R_SHOULDER];
    const nose=k[KP.NOSE];

    t.badges = [];

    // ── Detectores base ──────────────────────────────────
    this._detectZone(t, lw, rw, lh, rh, now);          // Z1 Z2 Z3
    this._detectPocket(t, lw, lh, ls, 'L');             // P1
    this._detectPocket(t, rw, rh, rs, 'R');             // P1
    this._detectCrossedArms(t, le, re, lw, rw, ls, rs, lh, rh); // P2
    this._detectHandObj(t, lw, rw, now);                // O1 O2 G F

    // ── Escenarios nuevos ────────────────────────────────
    this._checkCajaHeist(t, lw, rw, lh, rh, le, re, now);              // A
    this._checkPostContact(t, lw, rw, le, re, ls, rs, lh, rh, now);    // B C D
    this._checkProwling(t, now);                                         // E
    this._checkScanBehavior(t, nose, now);                               // H
    this._checkBodyScreen(t, nose, lw, rw);                              // I
    this._checkCrouchHide(t, nose, ls, rs, lh, rh, now);                // J
  }

  // ══════════════════════════════════════════════════════
  //  Z1 Z2 Z3 — ZONA: entrada / permanencia / escape
  // ══════════════════════════════════════════════════════
  _detectZone(t, lw, rw, lh, rh, now) {
    for (const [w, side] of [[lw,'L'],[rw,'R']]) {
      if (!_ok(w)) {
        for (const key of Object.keys(t.inZoneWrist))
          if (key.startsWith(side+'_')) { t.inZoneWrist[key]=false; t.dwellStart[key]=null; }
        continue;
      }
      const zones = this.zoneManager.getZonesForPoint(w.x, w.y);
      for (const zone of zones) {
        const key = `${side}_${zone.id}`;
        if (!t.inZoneWrist[key]) {
          t.inZoneWrist[key] = true;
          t.dwellStart[key]  = now;
          zone.alert = true;
          setTimeout(() => { if(zone) zone.alert=false; }, 2000);
          this._fire(`ze_${key}`, `MANO EN ${zone.name.toUpperCase()}`, 'low', 1500);
          // [E] Registrar visita
          this._recordVisit(t, zone, now);
          // [A] Marcar paso por caja
          if (zone.type === 'pago') t.visitedPay = true;
        }
        // Permanencia
        const elapsed = (now - (t.dwellStart[key]||now)) / 1000;
        if (elapsed >= this.config.dwellTime) {
          t.dwellStart[key] = now + this.config.dwellTime * 1000;
          this._fire(`dw_${key}`, `PERMANENCIA — ${zone.name.toUpperCase()}`, 'high', this.config.cooldown*1000);
        }
        // Escape al torso (Z3)
        if (t.history.length >= 6) this._detectEscape(t, side, zone, lh, rh);
        t.badges.push('⚠ EN ZONA');
      }
      // Salida de zona → registrar cajaExit si era zona de pago
      if (zones.length === 0) {
        for (const key of Object.keys(t.inZoneWrist)) {
          if (!key.startsWith(side+'_') || !t.inZoneWrist[key]) continue;
          t.inZoneWrist[key] = false;
          t.dwellStart[key]  = null;
          const zoneId = key.slice(2);
          const zone   = this.zoneManager.zones.find(z => z.id === zoneId);
          if (zone?.type === 'pago' && _ok(w))
            t.cajaExit[`${side}_${zoneId}`] = { t:now, wristY:w.y };
        }
      }
    }
  }

  _detectEscape(t, side, zone, lh, rh) {
    if (!lh||!rh) return;
    const mid = _mid(lh, rh);
    const hLen = t.history.length;
    const old  = t.history[Math.max(0, hLen-6)];
    const cur  = t.history[hLen-1];
    if (!old||!cur) return;
    const idx = side==='L' ? KP.L_WRIST : KP.R_WRIST;
    const pw=old.kps[idx], cw=cur.kps[idx];
    if (!_ok(pw)||!_ok(cw)) return;
    if (!this.zoneManager.getZonesForPoint(pw.x,pw.y).some(z=>z.id===zone.id)) return;
    const pd = _d(pw.x,pw.y, mid.x,mid.y);
    const cd = _d(cw.x,cw.y, mid.x,mid.y);
    if (cd < pd*0.65 && pd > 0.08)
      this._fire(`esc_${zone.id}_${side}`, `OBJETO OCULTADO — ${zone.name.toUpperCase()}`, 'high', this.config.cooldown*1000);
  }

  // ══════════════════════════════════════════════════════
  //  P1 — BOLSILLOS
  // ══════════════════════════════════════════════════════
  _detectPocket(t, wrist, hip, shoulder, side) {
    if (!hip||!shoulder) return;
    const hx=hip.x, hy=hip.y;
    let pocket = false;
    if (!wrist||wrist.c<0.20)       pocket = true;
    else if (wrist.c<0.50)          pocket = Math.abs(wrist.x-hx)<0.18 && Math.abs(wrist.y-hy)<0.22;
    else pocket = wrist.y>hy-0.05 && Math.abs(wrist.x-hx)<0.15 && Math.abs(wrist.y-hy)<0.19;
    const sk = side==='L' ? 'pocketL' : 'pocketR';
    if (pocket) {
      t[sk]++;
      if (t[sk]>=12) {
        t[sk]=0;
        this._fire(`pkt_${side}_${t.id}`, `MANO ${side==='L'?'IZQ.':'DER.'} EN BOLSILLO`, 'high', this.config.cooldown*1000);
      }
      if (t[sk]>6) t.badges.push('⚠ BOLSILLO');
    } else {
      t[sk] = Math.max(0, t[sk]-2);
    }
  }

  // ══════════════════════════════════════════════════════
  //  P2 — BRAZOS CRUZADOS
  // ══════════════════════════════════════════════════════
  _detectCrossedArms(t, le, re, lw, rw, ls, rs, lh, rh) {
    if (!le||!re||!ls||!rs||!lh||!rh) return;
    const mx=(ls.x+rs.x)/2, my=(ls.y+rs.y)/2, hy=(lh.y+rh.y)/2;
    const ok = Math.abs(le.x-mx)<0.20 && Math.abs(re.x-mx)<0.20
            && le.x>mx && re.x<mx
            && le.y>my && le.y<hy+0.08
            && re.y>my && re.y<hy+0.08
            && ((!lw||lw.c<0.40) || (!rw||rw.c<0.40));
    if (ok) {
      t.crossedArms++;
      if (t.crossedArms>=15) {
        t.crossedArms=0;
        this._fire(`cross_${t.id}`, 'BRAZOS CRUZADOS — POSIBLE OCULTAMIENTO', 'high', this.config.cooldown*1000);
      }
      if (t.crossedArms>8) t.badges.push('⚠ CRUZADO');
    } else {
      t.crossedArms = Math.max(0, t.crossedArms-2);
    }
  }

  // ══════════════════════════════════════════════════════
  //  O1 O2 + G (arrebato) + F (traspaso)
  // ══════════════════════════════════════════════════════
  _detectHandObj(t, lw, rw, now) {
    if (!this._objDets.length) return;
    for (const [w, side] of [[lw,'L'],[rw,'R']]) {
      if (!_ok(w)) continue;
      for (const obj of this._objDets) {
        if (!ALERT_CLASSES.has(obj.cls)) continue;
        const m = 0.06;
        const touching = w.x>=obj.nx1-m && w.x<=obj.nx2+m && w.y>=obj.ny1-m && w.y<=obj.ny2+m;
        const intKey   = `${t.id}_${obj.cls}_${side}`;

        if (touching) {
          if (!this._interactions[intKey])
            this._interactions[intKey] = { startT:now, objBox:{nx1:obj.nx1,ny1:obj.ny1,nx2:obj.nx2,ny2:obj.ny2}, label:obj.label, cls:obj.cls };
          // O1: objeto en zona
          const zones = this.zoneManager.getZonesForPoint((obj.nx1+obj.nx2)/2, (obj.ny1+obj.ny2)/2);
          if (zones.length>0)
            this._fire(`oz_${intKey}`, `${obj.label.toUpperCase()} EN ${zones[0].name.toUpperCase()}`, 'high', this.config.cooldown*1000);

        } else if (this._interactions[intKey]) {
          const d   = this._interactions[intKey];
          delete this._interactions[intKey];
          const dur = now - d.startT;
          if (dur < GRAB_MIN_MS) continue; // demasiado rápido = ruido

          const stillThere = this._objDets.some(o => o.cls===obj.cls && this._iou(o,d.objBox)>0.25);
          if (stillThere) continue; // objeto sigue visible → no fue tomado

          // [F] ¿Traspaso? Otra persona muy cerca
          const nearby = this._countNearby(t, 0.22);
          if (nearby > 0) {
            this._fire(`hof_${t.id}_${obj.cls}`, `TRASPASO: ${d.label.toUpperCase()} (${nearby} persona${nearby>1?'s':''} cerca)`, 'high', this.config.cooldown*1000);
          }
          // [G] Arrebato rápido (300-700ms)
          else if (dur <= GRAB_MAX_MS) {
            this._fire(`grab_${t.id}_${obj.cls}_${side}`, `ARREBATO: ${d.label.toUpperCase()}`, 'high', this.config.cooldown*1000);
          }
          // O2: objeto tomado normal (>700ms)
          else {
            this._fire(`og_${t.id}_${obj.cls}_${side}`, `OBJETO TOMADO: ${d.label.toUpperCase()}`, 'high', this.config.cooldown*1000);
          }

          // Iniciar post-contact para analizar destino [B][C][D][J]
          if (_ok(w)) {
            const elbow = side==='L' ? t.kps[KP.L_ELBOW] : t.kps[KP.R_ELBOW];
            t.postContact = {
              disappearT: now,
              label:      d.label,
              cls:        d.cls,
              side,
              wristY0:    w.y,
              elbowY0:    _ok(elbow) ? elbow.y : null,
              fired:      false,
            };
          }
        }
      }
      // Limpiar interacciones expiradas (>8s sin soltar)
      for (const k of Object.keys(this._interactions))
        if (k.startsWith(`${t.id}_`) && now-this._interactions[k].startT > 8000)
          delete this._interactions[k];
    }
  }

  // Helper: cuántas personas a distancia < maxDist del centro de este track
  _countNearby(t, maxDist) {
    const cx=(t.nx1+t.nx2)/2, cy=(t.ny1+t.ny2)/2;
    let n=0;
    for (const other of this._tracks)
      if (other.id!==t.id && !other.missed && _d(cx,cy,(other.nx1+other.nx2)/2,(other.ny1+other.ny2)/2) < maxDist) n++;
    return n;
  }

  // ══════════════════════════════════════════════════════
  //  [A] CAJA HEIST — mano sale de zona pago, analizar destino 2s
  // ══════════════════════════════════════════════════════
  _checkCajaHeist(t, lw, rw, lh, rh, le, re, now) {
    for (const [w, elbow, hip, side] of [[lw,le,lh,'L'],[rw,re,rh,'R']]) {
      for (const [key, state] of Object.entries(t.cajaExit)) {
        if (!key.startsWith(side+'_')) continue;
        if (now - state.t > CAJA_HEIST_MS) { delete t.cajaExit[key]; continue; }
        if (!_ok(w)) continue;
        const zoneName = key.slice(2);

        // Destino bolsillo: muñeca bajó y está cerca de la cadera
        if (_ok(hip)) {
          const bajó    = w.y > state.wristY + 0.06;
          const enCadera = Math.abs(w.x-hip.x)<0.15 && Math.abs(w.y-hip.y)<0.18;
          if (bajó && enCadera) {
            this._fire(`cj_pkt_${key}`, `CAJA → BOLSILLO: POSIBLE EXTRACCIÓN`, 'high', this.config.cooldown*1000);
            delete t.cajaExit[key]; continue;
          }
        }
        // Destino manga: muñeca sube por encima del codo
        if (_ok(elbow)) {
          const subió     = w.y < state.wristY - 0.07;
          const sobreCodo = w.y < elbow.y - 0.04;
          if (subió && sobreCodo) {
            this._fire(`cj_slv_${key}`, `CAJA → MANGA: POSIBLE EXTRACCIÓN`, 'high', this.config.cooldown*1000);
            delete t.cajaExit[key]; continue;
          }
        }
      }
    }
  }

  // ══════════════════════════════════════════════════════
  //  [B][C][D] POST-CONTACT — destino de mano después de perder objeto
  // ══════════════════════════════════════════════════════
  _checkPostContact(t, lw, rw, le, re, ls, rs, lh, rh, now) {
    const pc = t.postContact;
    if (!pc || pc.fired) return;
    if (now - pc.disappearT > POST_CONTACT_MS) { t.postContact=null; return; }

    const w     = pc.side==='L' ? lw : rw;
    const elbow = pc.side==='L' ? le : re;
    if (!_ok(w)) return;

    // [B] MANGA: muñeca sube >7% por encima del codo
    if (pc.elbowY0!==null && _ok(elbow)) {
      if (w.y < pc.wristY0 - 0.07 && w.y < elbow.y - 0.04) {
        this._fire(`slv_${t.id}_${pc.cls}`, `MANGA — ${pc.label.toUpperCase()} BAJO MANGA`, 'high', this.config.cooldown*1000);
        pc.fired=true; t.postContact=null; return;
      }
    }

    // [C] BAG STUFFING: muñeca <14% de un bolso/mochila en frame
    const nearBag = this._objDets.find(o =>
      BAG_CLASSES.has(o.cls) && _d(w.x,w.y,(o.nx1+o.nx2)/2,(o.ny1+o.ny2)/2) < 0.14
    );
    if (nearBag) {
      this._fire(`bag_${t.id}_${pc.cls}`, `BOLSO — ${pc.label.toUpperCase()} INTRODUCIDO EN ${nearBag.label.toUpperCase()}`, 'high', this.config.cooldown*1000);
      pc.fired=true; t.postContact=null; return;
    }

    // [D] BAJO ROPA: muñeca en torso central + conf baja (oculta)
    if (_ok(ls) && _ok(rs) && _ok(lh) && _ok(rh)) {
      const tx1 = Math.min(ls.x,rs.x)+0.03, tx2 = Math.max(ls.x,rs.x)-0.03;
      const ty1 = Math.min(ls.y,rs.y),      ty2 = Math.max(lh.y,rh.y);
      if (w.x>tx1 && w.x<tx2 && w.y>ty1 && w.y<ty2 && w.c<0.40) {
        this._fire(`trso_${t.id}_${pc.cls}`, `ROPA — ${pc.label.toUpperCase()} OCULTADO BAJO LA ROPA`, 'high', this.config.cooldown*1000);
        pc.fired=true; t.postContact=null; return;
      }
    }
  }

  // ══════════════════════════════════════════════════════
  //  [E] MERODEO — entradas repetidas sin comprar
  // ══════════════════════════════════════════════════════
  _recordVisit(t, zone, now) {
    if (zone.type==='pago') return;
    if (!t.zoneVisits[zone.id]) t.zoneVisits[zone.id]=[];
    t.zoneVisits[zone.id].push(now);
    // Trim a ventana
    t.zoneVisits[zone.id] = t.zoneVisits[zone.id].filter(ts => now-ts < PROWL_WINDOW_MS);
  }

  _checkProwling(t, now) {
    for (const [zoneId, tss] of Object.entries(t.zoneVisits)) {
      if (tss.length < PROWL_THRESH || t.visitedPay) continue;
      const zone = this.zoneManager.zones.find(z=>z.id===zoneId);
      this._fire(
        `prl_${t.id}_${zoneId}`,
        `MERODEO — ${tss.length} ENTRADAS EN ${zone?.name?.toUpperCase()||'ZONA'}`,
        'medium',
        this.config.cooldown * 1500
      );
      t.badges.push('⚠ MERODEO');
    }
  }

  // ══════════════════════════════════════════════════════
  //  [H] ESCANEO PREVIO — cabeza gira izq-der antes de actuar
  //  Gate doble: nariz se mueve mucho + persona cerca de objeto
  //  de alerta O en zona. Evita false positives al caminar.
  // ══════════════════════════════════════════════════════
  _checkScanBehavior(t, nose, now) {
    if (!_ok(nose)) return;

    // Agregar punto de historial
    t.noseXHist.push({ x:nose.x, t:now });
    t.noseXHist = t.noseXHist.filter(p => now-p.t < SCAN_WINDOW_MS);
    if (t.noseXHist.length < 6) return;

    // Calcular varianza X en ventana
    const xs   = t.noseXHist.map(p=>p.x);
    const mean = xs.reduce((a,b)=>a+b,0)/xs.length;
    const variance = xs.reduce((a,x)=>a+(x-mean)**2,0)/xs.length;
    const stddev   = Math.sqrt(variance);

    // Umbral: movimiento de cabeza pronunciado (stddev > 0.06)
    if (stddev < 0.06) return;

    // GATE: solo disparar si persona está en zona O cerca de objeto de alerta
    const inZone = Object.values(t.inZoneWrist).some(v=>v);
    const cx = (t.nx1+t.nx2)/2, cy = (t.ny1+t.ny2)/2;
    const nearAlertObj = this._objDets.some(o =>
      ALERT_CLASSES.has(o.cls) && _d(cx,cy,(o.nx1+o.nx2)/2,(o.ny1+o.ny2)/2) < 0.30
    );
    if (!inZone && !nearAlertObj) return;

    this._fire(`scan_${t.id}`, 'ESCANEO — COMPORTAMIENTO PREVIO A HURTO', 'medium', this.config.cooldown*1000);
    t.badges.push('⚠ ESCANEO');
    t.noseXHist = []; // reset para no spamear
  }

  // ══════════════════════════════════════════════════════
  //  [I] CUERPO COMO PANTALLA — espalda a cámara mientras en zona
  //  Gate: muñeca en zona + nariz no visible
  // ══════════════════════════════════════════════════════
  _checkBodyScreen(t, nose, lw, rw) {
    const noseHidden = !nose || nose.c < KP_THRESH;
    const wristInZone = Object.values(t.inZoneWrist).some(v=>v);
    if (noseHidden && wristInZone) {
      t.bodyScreen++;
      if (t.bodyScreen >= 10) {
        t.bodyScreen = 0;
        this._fire(`bsc_${t.id}`, 'CUERPO COMO PANTALLA — DE ESPALDAS EN ZONA', 'high', this.config.cooldown*1000);
      }
      if (t.bodyScreen > 5) t.badges.push('⚠ PANTALLA');
    } else {
      t.bodyScreen = Math.max(0, t.bodyScreen-2);
    }
  }

  // ══════════════════════════════════════════════════════
  //  [J] AGACHARSE Y OCULTAR — nariz baja durante postContact
  //  Gate: postContact activo + hombros visibles como referencia
  //  Cuando alguien se agacha para meter algo en bolso del piso
  //  o en la media/calcetín
  // ══════════════════════════════════════════════════════
  _checkCrouchHide(t, nose, ls, rs, lh, rh, now) {
    if (!t.postContact || t.postContact.fired) return;
    if (!_ok(nose) || !_ok(ls) || !_ok(rs)) return;

    // Punto de referencia: mitad entre hombros y cadera
    const shoulderY = (ls.y + rs.y) / 2;
    const hipY      = _ok(lh) && _ok(rh) ? (lh.y+rh.y)/2 : shoulderY + 0.3;
    const midY      = (shoulderY + hipY) / 2;

    // Si la nariz está por debajo del punto medio del torso → agachado
    // (Y mayor = más abajo en imagen)
    const crouching = nose.y > midY + 0.08;

    if (crouching) {
      t.crouchHide++;
      if (t.crouchHide >= 8) {
        t.crouchHide = 0;
        this._fire(
          `crch_${t.id}_${t.postContact.cls}`,
          `AGACHADO — ${t.postContact.label.toUpperCase()} OCULTADO EN ZONA BAJA`,
          'high', this.config.cooldown*1000
        );
        t.postContact.fired = true;
        t.badges.push('⚠ AGACHADO');
      }
    } else {
      t.crouchHide = Math.max(0, t.crouchHide-2);
    }
  }

  // ══════════════════════════════════════════════════════
  //  FIRE — dedup + cooldown por clave
  // ══════════════════════════════════════════════════════
  _fire(key, type, severity, coolMs) {
    const now = Date.now();
    if (now - (this._lastAlert[key]||0) < coolMs) return;
    this._lastAlert[key] = now;
    if (this.onDetection)  this.onDetection(type, severity);
    if (this.alertManager) this.alertManager.trigger(type, severity);
  }

  // ══════════════════════════════════════════════════════
  //  RENDER
  // ══════════════════════════════════════════════════════
  _render() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.zoneManager.drawZone(this.zoneManager.zones.some(z=>z.alert));
    this.zoneManager.drawPreview();
    this._drawDetections(this._lastDets, this._objDets);
  }

  _drawDetections(poseDets, objDets) {
    const ctx=this.ctx, cw=this.canvas.width, ch=this.canvas.height;

    // Objetos
    for (const obj of (objDets||[])) {
      const x1=obj.nx1*cw, y1=obj.ny1*ch, x2=obj.nx2*cw, y2=obj.ny2*ch;
      const isAlert = ALERT_CLASSES.has(obj.cls);
      const isBag   = BAG_CLASSES.has(obj.cls);
      const col     = isAlert
        ? (isBag ? 'rgba(191,90,242,0.9)' : 'rgba(255,170,0,0.85)')
        : 'rgba(160,160,160,0.5)';
      ctx.save();
      ctx.strokeStyle=col; ctx.lineWidth=isAlert?1.8:1;
      ctx.setLineDash([4,3]); ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
      const lbl = `${obj.label} ${Math.round(obj.conf*100)}%`;
      const lw2 = ctx.measureText(lbl).width + 6;
      ctx.font='9px "Share Tech Mono",monospace';
      ctx.fillStyle = isAlert ? (isBag?'rgba(191,90,242,0.15)':'rgba(255,170,0,0.15)') : 'rgba(40,40,40,0.4)';
      ctx.fillRect(x1,y1-14,lw2,13);
      ctx.fillStyle=col; ctx.fillText(lbl,x1+3,y1-4);
      ctx.restore();
    }

    // Personas
    for (const det of poseDets) {
      const k  = det.kps;
      const x1 = det.nx1*cw, y1=det.ny1*ch, x2=det.nx2*cw;
      const track    = this._tracks.find(t=>!t.missed&&this._iou(t,det)>0.3);
      const inZone   = track && Object.values(track.inZoneWrist||{}).some(v=>v);
      const hasPost  = track?.postContact && !track.postContact.fired;
      const scanning = track?.badges?.includes('⚠ ESCANEO');

      // Color del bounding box según estado más severo
      const boxCol = inZone   ? '#ff3d3d'
                   : hasPost  ? '#ffaa00'
                   : scanning ? '#bf5af2'
                   : 'rgba(0,200,255,0.45)';
      ctx.save();
      ctx.strokeStyle=boxCol; ctx.lineWidth=(inZone||hasPost)?2:1.5;
      ctx.strokeRect(x1,y1,x2-x1,(det.ny2-det.ny1)*ch);
      ctx.fillStyle=boxCol; ctx.font='10px "Share Tech Mono",monospace';
      ctx.fillText(`${Math.round(det.conf*100)}%`, x1+3, y1-3);
      ctx.restore();

      // Esqueleto
      ctx.save(); ctx.lineWidth=1.8;
      for (const [a,b] of BONES) {
        const pa=k[a], pb=k[b];
        if (!_ok(pa)||!_ok(pb)) continue;
        ctx.beginPath(); ctx.moveTo(pa.x*cw,pa.y*ch); ctx.lineTo(pb.x*cw,pb.y*ch);
        ctx.strokeStyle='rgba(0,200,255,0.5)'; ctx.globalAlpha=0.75; ctx.stroke();
      }
      ctx.globalAlpha=1;

      // Keypoints
      for (let i=0; i<17; i++) {
        const p=k[i]; if (!_ok(p)) continue;
        const isWrist = i===KP.L_WRIST||i===KP.R_WRIST;
        const isHip   = i===KP.L_HIP||i===KP.R_HIP;
        const inZ     = isWrist && this.zoneManager.getZonesForPoint(p.x,p.y).length>0;
        const onObj   = isWrist && (objDets||[]).some(o=>{
          const m=0.06;
          return ALERT_CLASSES.has(o.cls)&&p.x>=o.nx1-m&&p.x<=o.nx2+m&&p.y>=o.ny1-m&&p.y<=o.ny2+m;
        });
        ctx.beginPath();
        ctx.arc(p.x*cw, p.y*ch, isWrist?6:isHip?4:3, 0, Math.PI*2);
        ctx.fillStyle = inZ?'#ff3d3d':isWrist?'#ffb800':isHip?'#bf5af2':'rgba(255,255,255,0.7)';
        ctx.fill();
        if (inZ||onObj) {
          ctx.beginPath(); ctx.arc(p.x*cw,p.y*ch,11,0,Math.PI*2);
          ctx.strokeStyle = inZ?'#ff3d3d':'#ffb800'; ctx.lineWidth=1.5;
          ctx.globalAlpha = 0.5+0.5*Math.sin(Date.now()/200);
          ctx.stroke(); ctx.globalAlpha=1;
        }
      }
      ctx.restore();

      // Badges del track
      if (track?.badges?.length) {
        ctx.save(); ctx.font='bold 9px "Share Tech Mono",monospace';
        let bx=det.nx1*cw; const by=det.ny2*ch+13;
        for (const badge of track.badges) {
          const col = badge.includes('ZONA')    ? '#ff3d3d'
                    : badge.includes('MERODEO') ? '#ffaa00'
                    : badge.includes('ESCANEO') ? '#bf5af2'
                    : 'rgba(255,58,58,0.9)';
          ctx.fillStyle=col; ctx.fillText(badge, bx, by);
          bx += ctx.measureText(badge).width + 8;
        }
        ctx.restore();
      }

      // Indicador naranja pulsante durante post-contact ("SEGUIMIENTO")
      if (hasPost) {
        const w = track.postContact.side==='L' ? k[KP.L_WRIST] : k[KP.R_WRIST];
        if (_ok(w)) {
          ctx.save();
          ctx.beginPath(); ctx.arc(w.x*cw, w.y*ch, 14, 0, Math.PI*2);
          ctx.strokeStyle='#ffaa00'; ctx.lineWidth=2;
          ctx.globalAlpha = 0.4+0.4*Math.sin(Date.now()/150);
          ctx.stroke(); ctx.globalAlpha=1;
          ctx.font='bold 8px "Share Tech Mono",monospace';
          ctx.fillStyle='#ffaa00';
          ctx.fillText('SEGUIMIENTO', w.x*cw-28, w.y*ch-17);
          ctx.restore();
        }
      }
    }
  }

  // ══════════════════════════════════════════════════════
  //  CONTROL
  // ══════════════════════════════════════════════════════
  start() {
    this.active=true; this._lastAlert={}; this._interactions={};
    for (const t of this._tracks)
      Object.assign(t, {
        inZoneWrist:{}, dwellStart:{},
        pocketL:0, pocketR:0, crossedArms:0,
        cajaExit:{}, postContact:null,
        zoneVisits:{}, visitedPay:false,
        noseXHist:[], bodyScreen:0, crouchHide:0,
        badges:[],
      });
  }
  stop()          { this.active=false; }
  updateConfig(c) { Object.assign(this.config, c); }
  destroy()       { if (this._renderLoopId) cancelAnimationFrame(this._renderLoopId); }
}