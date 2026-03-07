/**
 * store-profiles.js — SSIP v6.1
 * ─────────────────────────────────────────────────────────────────────
 * Cambios v6.0 sobre v5.0:
 *
 * [P1] FAMILIAS YOLOE — con YOLOE 1200+ clases, expandimos los IDs
 *      a rangos que YOLOE detecta y COCO no tiene. Las familias ahora
 *      incluyen IDs extendidos (200-999) para categorías LVIS comunes
 *      en retail: cosméticos, medicamentos, snacks, herramientas, etc.
 *      Con YOLO 80 clases estos IDs nunca aparecen → sin efecto.
 *      Con YOLOE sí aparecen → detección real de objetos no-COCO.
 *
 * [P2] SCORE_BONUS_DEFAULTS actualizados para v6.0:
 *      · pinchGrip      — nuevo (MediaPipe confirma toma real)
 *      · secuencia      — nuevo (multiplicador ya aplicado en detection.js,
 *                         este bonus es para el score base de evidencia)
 *      · coordinacion   — nuevo (escaneo sincrónico grupal)
 *      · formacionV     — nuevo (bloqueo de cámara con 3+ personas)
 *      · wristOculta    — nuevo (muñeca de baja confianza post-contacto)
 *
 * [P3] PERFILES RECALIBRADOS para v6.0:
 *      · scoreThreshold ajustados al nuevo sistema de score acumulativo
 *        (con secuencia multiplier, los scores llegan más alto más rápido)
 *      · postContactMs ampliados en perfiles clave (más tiempo para detectar
 *        el destino del objeto)
 *      · families actualizadas con familias YOLOE (FOOD, COSMETIC, PHARMA)
 *      · behaviors: agregados pinchGrip, coordinacion (nuevos en v6.0)
 *      · Tienda de Ropa: families ampliadas — con YOLOE detecta prendas reales
 *
 * [P4] NUEVOS PERFILES:
 *      · electronica  — celulares, laptops, accesorios tech
 *      · estacion     — estación de servicio / minimarket 24h
 *      · libreria     — librería / papelería
 */

// ─────────────────────────────────────────────────────────────────────
//  FAMILIAS DE OBJETOS
//  IDs 0-79   = COCO (yolo26n.onnx)
//  IDs 80-999 = LVIS extendido (yoloe26n.onnx) — sin efecto con COCO
// ─────────────────────────────────────────────────────────────────────
export const OBJ_FAMILIES = {

  SMALL: {
    // COCO: botella(39), taza(41), cuchara(44), tazón(45), reloj(75), jarrón(76), tijera(78)
    // LVIS: spray(312), frasco(201), tubo(445), perfume(602), encendedor(389)
    ids:     new Set([39, 41, 44, 45, 75, 76, 78, 201, 312, 389, 445, 602]),
    label:   'OBJETO PEQUEÑO',
    minConf: 0.38,  // [CAL] ligeramente más permisivo para objetos parcialmente visibles
  },

  MEDIUM: {
    // COCO: copa(40), tenedor(42), cuchillo(43), naranja(49), manzana(47), celular(67), libro(73)
    // LVIS: billetera(156), cartera(203), sobre(488), caja_pequeña(198)
    ids:     new Set([40, 42, 43, 46, 47, 49, 67, 73, 156, 198, 203, 488]),
    label:   'OBJETO',
    minConf: 0.36,
  },

  BAG: {
    // COCO: mochila(24), bolso(26), valija(28)
    // LVIS: bolsa_tela(163), riñonera(677), bolsa_compras(164)
    ids:     new Set([24, 26, 28, 163, 164, 677]),
    label:   'BOLSO/MOCHILA',
    minConf: 0.50,  // [CAL] subido de 0.33 — evita falso-positivo bolso en objetos oscuros
  },

  TECH: {
    // COCO: laptop(63), mouse(64), control(65), teclado(66)
    // LVIS: tablet(805), auricular(118), cargador(210), cámara(199)
    ids:     new Set([63, 64, 65, 66, 118, 199, 210, 805]),
    label:   'DISPOSITIVO',
    minConf: 0.32,
  },

  JEWELRY: {
    // COCO: reloj(74)
    // LVIS: anillo(93), collar(248), pulsera(182), aretes(111), broche(185)
    ids:     new Set([74, 93, 111, 182, 185, 248]),
    label:   'JOYA/RELOJ',
    minConf: 0.45,
  },

  // ── Nuevas familias YOLOE ────────────────────────────────────────
  FOOD: {
    // LVIS: snack(721), chocolate(224), bebida_lata(140), yogurt(876)
    // En supermercados y kioscos es lo más hurtado que COCO no detecta
    ids:     new Set([140, 224, 721, 876, 133, 512, 634]),
    label:   'ALIMENTO/BEBIDA',
    minConf: 0.38,
  },

  COSMETIC: {
    // LVIS: labial(478), base_maquillaje(136), crema(276), desodorante(306)
    ids:     new Set([136, 276, 306, 478, 519, 623]),
    label:   'COSMÉTICO',
    minConf: 0.40,
  },

  PHARMA: {
    // LVIS: caja_medicamento(197), blister(158), frasco_medicamento(361)
    // Crítico para farmacias
    ids:     new Set([158, 197, 361, 402, 558]),
    label:   'MEDICAMENTO',
    minConf: 0.42,
  },

  CLOTHING: {
    // LVIS: remera(736), pantalón(598), campera(207), gorra(369)
    // Con YOLOE por fin detectamos ropa — antes era imposible con COCO
    ids:     new Set([207, 369, 488, 598, 736, 783, 812, 834]),
    label:   'PRENDA',
    minConf: 0.38,
  },

  TOOL: {
    // LVIS: cuchillo_herramienta(459), destornillador(308), llave(473)
    // Para bazares y ferreterías
    ids:     new Set([308, 459, 473, 531, 612]),
    label:   'HERRAMIENTA',
    minConf: 0.40,
  },
};

/** Dado un cls, retorna la familia o null */
export function getFamily(cls) {
  for (const [key, fam] of Object.entries(OBJ_FAMILIES)) {
    if (fam.ids.has(cls)) return { key, ...fam };
  }
  return null;
}

export const BAG_IDS = new Set([
  ...OBJ_FAMILIES.BAG.ids,
]);

export const ALERT_IDS = new Set([
  ...OBJ_FAMILIES.SMALL.ids,
  ...OBJ_FAMILIES.MEDIUM.ids,
  ...OBJ_FAMILIES.BAG.ids,
  ...OBJ_FAMILIES.TECH.ids,
  ...OBJ_FAMILIES.JEWELRY.ids,
  ...OBJ_FAMILIES.FOOD.ids,
  ...OBJ_FAMILIES.COSMETIC.ids,
  ...OBJ_FAMILIES.PHARMA.ids,
  ...OBJ_FAMILIES.CLOTHING.ids,
  ...OBJ_FAMILIES.TOOL.ids,
]);

// ─────────────────────────────────────────────────────────────────────
//  SCORE BONUS DEFAULTS — base para todos los perfiles
//  Recalibrados para v6.0 (con multiplicador de secuencia activo,
//  los scores individuales son ligeramente más conservadores)
// ─────────────────────────────────────────────────────────────────────
const SCORE_BONUS_DEFAULTS = {
  // Comportamientos originales — ligeramente reducidos porque el
  // multiplicador de secuencia (×1.5) los amplifica cuando hay secuencia
  contacto:          13,
  objetoTomado:      38,
  arrebato:          52,
  traspaso:          48,
  bajoropa:          33,
  cadera:            33,
  manga:             33,
  agachado:          28,
  bagStuffing:       43,
  pantalla:          23,
  escaneo:            9,
  merodeo:           18,
  brazoscruzados:    13,
  distractor:        28,
  trayectoria:        7,

  // Nuevos en v6.0
  pinchGrip:         18,   // MediaPipe confirma toma real — evidencia fuerte
  wristOculta:       18,   // muñeca de baja confianza post-contacto
  coordinacion:      20,   // escaneo sincrónico 2+ personas
  formacionV:        15,   // formación en V bloqueando cámara
  secuencia:         10,   // bonus base por secuencia (además del ×1.5)
};

// ─────────────────────────────────────────────────────────────────────
//  PERFILES
// ─────────────────────────────────────────────────────────────────────
const PROFILES = {

  // ── Genérico (fallback) ──────────────────────────────────────────
  generico: {
    name:            'Genérico',
    icon:            '🏪',
    dwellTime:       4,
    contactMinMs:    400,
    grabMaxMs:       700,
    scoreThreshold:  68,       // bajado de 72: el sistema v6 es más preciso
    postContactMs:   5000,     // ampliado de 4000 (más tiempo para detectar destino)
    zoneEntryFrames: 3,
    hipConcealConf:  0.55,
    families:        ['SMALL','MEDIUM','BAG','TECH','FOOD'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    true,
      trayectoria:   true,
      pinchGrip:     true,    // nuevo v6.0
      coordinacion:  true,    // nuevo v6.0
    },
  },

  // ── Supermercado / Almacén ────────────────────────────────────────
  supermercado: {
    name:            'Supermercado / Almacén',
    icon:            '🛒',
    dwellTime:       5,
    contactMinMs:    450,      // reducido un poco: con YOLOE detectamos más objetos reales
    grabMaxMs:       700,
    scoreThreshold:  70,       // bajado de 75
    postContactMs:   5000,
    zoneEntryFrames: 4,
    hipConcealConf:  0.55,
    families:        ['SMALL','MEDIUM','BAG','FOOD','COSMETIC'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    true,
      trayectoria:   false,   // mucho tráfico → falsos positivos
      pinchGrip:     true,
      coordinacion:  true,
    },
    scoreBonus: {
      contacto:      9,        // normal revisar productos
      objetoTomado:  33,
      bagStuffing:   43,
      pinchGrip:     20,       // en super confirma toma real
      coordinacion:  25,       // grupos que hurtan coordinados son comunes
    },
  },

  // ── Farmacia ─────────────────────────────────────────────────────
  farmacia: {
    name:            'Farmacia',
    icon:            '💊',
    dwellTime:       3,
    contactMinMs:    300,      // reducido: con PHARMA+COSMETIC hay más objetos reales
    grabMaxMs:       550,
    scoreThreshold:  62,       // bajado de 68: objetos de farmacia son pequeños y caros
    postContactMs:   5000,
    zoneEntryFrames: 3,
    hipConcealConf:  0.50,     // más sensible
    families:        ['SMALL','MEDIUM','PHARMA','COSMETIC'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    true,
      trayectoria:   true,
      pinchGrip:     true,
      coordinacion:  true,
    },
    scoreBonus: {
      contacto:      16,
      objetoTomado:  42,
      pinchGrip:     22,       // en farmacia el pinch es muy relevante
      manga:         38,
      cadera:        38,
    },
  },

  // ── Kiosco / Cafetería ───────────────────────────────────────────
  kiosco: {
    name:            'Kiosco / Cafetería',
    icon:            '☕',
    dwellTime:       2,
    contactMinMs:    280,
    grabMaxMs:       550,
    scoreThreshold:  60,       // bajado de 65
    postContactMs:   3500,
    zoneEntryFrames: 2,
    hipConcealConf:  0.52,
    families:        ['SMALL','MEDIUM','BAG','FOOD'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    false,   // local chico
      trayectoria:   true,
      pinchGrip:     true,
      coordinacion:  false,   // muy poco espacio para coordinar
    },
    scoreBonus: {
      contacto:      20,
      pinchGrip:     22,
      trayectoria:   12,      // en kiosco la trayectoria directa es muy sospechosa
    },
  },

  // ── Joyería ──────────────────────────────────────────────────────
  joyeria: {
    name:            'Joyería',
    icon:            '💎',
    dwellTime:       2,
    contactMinMs:    180,      // cualquier contacto es relevante
    grabMaxMs:       450,
    scoreThreshold:  50,       // bajado de 55: objetos de altísimo valor
    postContactMs:   6000,     // ventana larga: pueden pasar el objeto a cómplice
    zoneEntryFrames: 2,
    hipConcealConf:  0.58,
    families:        ['JEWELRY','SMALL','MEDIUM'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      false,
      bagStuffing:   false,
      traspaso:      true,
      distractor:    true,
      trayectoria:   true,
      pinchGrip:     true,    // en joyería el pinch es evidencia muy fuerte
      coordinacion:  true,
    },
    scoreBonus: {
      contacto:      28,
      objetoTomado:  58,
      escaneo:       22,
      pinchGrip:     35,      // pinch en joyería = casi certeza
      traspaso:      55,
      distractor:    35,
      coordinacion:  30,
    },
  },

  // ── Tienda de Ropa ───────────────────────────────────────────────
  ropa: {
    name:            'Tienda de Ropa',
    icon:            '👕',
    dwellTime:       6,
    contactMinMs:    550,
    grabMaxMs:       800,
    scoreThreshold:  72,       // bajado de 78
    postContactMs:   6000,     // layering tarda más
    zoneEntryFrames: 4,
    hipConcealConf:  0.50,
    // Con YOLOE ahora detectamos CLOTHING directamente — gran mejora
    families:        ['BAG','CLOTHING','SMALL'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    true,
      trayectoria:   false,
      pinchGrip:     true,
      coordinacion:  true,
    },
    scoreBonus: {
      bagStuffing:   52,
      manga:         42,
      cadera:        40,      // layering bajo ropa
      pinchGrip:     15,      // menos relevante (se toca ropa normalmente)
      coordinacion:  28,
      bajoropa:      45,      // layering es el robo más común en ropa
    },
  },

  // ── Bazar / Tienda variada ───────────────────────────────────────
  bazar: {
    name:            'Bazar / Tienda variada',
    icon:            '🏬',
    dwellTime:       4,
    contactMinMs:    380,
    grabMaxMs:       680,
    scoreThreshold:  66,       // bajado de 70
    postContactMs:   5000,
    zoneEntryFrames: 3,
    hipConcealConf:  0.53,
    families:        ['SMALL','MEDIUM','BAG','TECH','TOOL'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    true,
      trayectoria:   true,
      pinchGrip:     true,
      coordinacion:  true,
    },
  },

  // ── Depósito / Bodega ────────────────────────────────────────────
  deposito: {
    name:            'Depósito / Bodega',
    icon:            '📦',
    dwellTime:       2,
    contactMinMs:    280,
    grabMaxMs:       600,
    scoreThreshold:  55,       // bajado de 60: acceso no autorizado ya es sospechoso
    postContactMs:   5000,
    zoneEntryFrames: 2,
    hipConcealConf:  0.48,     // más sensible
    families:        ['SMALL','MEDIUM','BAG','TECH','TOOL'],
    behaviors: {
      merodeo:       false,    // la sola presencia ya es sospechosa
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    false,
      trayectoria:   false,
      pinchGrip:     true,
      coordinacion:  true,
    },
    scoreBonus: {
      contacto:      22,
      pinchGrip:     25,
      bagStuffing:   50,
      traspaso:      52,
    },
  },

  // ── Cocina / Área de preparación ────────────────────────────────
  cocina: {
    name:            'Cocina / Área de preparación',
    icon:            '🍳',
    dwellTime:       8,
    contactMinMs:    800,
    grabMaxMs:       1000,
    scoreThreshold:  78,       // bajado de 82
    postContactMs:   4000,
    zoneEntryFrames: 5,
    hipConcealConf:  0.58,
    families:        ['SMALL','MEDIUM','TECH','FOOD'],
    behaviors: {
      merodeo:       false,
      escaneo:       false,
      pantalla:      false,
      cadera:        true,
      manga:         true,
      agachado:      false,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    false,
      trayectoria:   false,
      pinchGrip:     false,   // manipular en cocina es normal
      coordinacion:  false,
    },
    scoreBonus: {
      contacto:      4,        // manipular objetos es normal
      cadera:        48,
      manga:         48,
      bagStuffing:   58,
      traspaso:      48,
    },
  },

  // ── Electrónica ── NUEVO v6.0 ────────────────────────────────────
  electronica: {
    name:            'Tienda de Electrónica',
    icon:            '📱',
    dwellTime:       3,
    contactMinMs:    300,
    grabMaxMs:       500,
    scoreThreshold:  58,       // objetos de alto valor, threshold bajo
    postContactMs:   6000,
    zoneEntryFrames: 3,
    hipConcealConf:  0.55,
    families:        ['TECH','SMALL','MEDIUM'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      false,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    true,
      trayectoria:   true,
      pinchGrip:     true,
      coordinacion:  true,
    },
    scoreBonus: {
      contacto:      20,
      objetoTomado:  55,
      pinchGrip:     30,
      traspaso:      55,
      distractor:    35,
      coordinacion:  30,
    },
  },

  // ── Estación de Servicio / Minimarket 24h ── NUEVO v6.0 ──────────
  estacion: {
    name:            'Estación / Minimarket 24h',
    icon:            '⛽',
    dwellTime:       2,        // local pequeño, 24h, alta rotación nocturna
    contactMinMs:    280,
    grabMaxMs:       550,
    scoreThreshold:  58,       // más sensible: turno nocturno, poco personal
    postContactMs:   4000,
    zoneEntryFrames: 2,
    hipConcealConf:  0.50,
    families:        ['SMALL','MEDIUM','BAG','FOOD','TECH'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    true,
      trayectoria:   true,
      pinchGrip:     true,
      coordinacion:  true,
    },
    scoreBonus: {
      contacto:      16,
      trayectoria:   12,
      pinchGrip:     22,
      merodeo:       25,       // merodeo nocturno en estación es muy sospechoso
    },
  },

  // ── Librería / Papelería ── NUEVO v6.0 ───────────────────────────
  libreria: {
    name:            'Librería / Papelería',
    icon:            '📚',
    dwellTime:       5,        // los clientes navegan y leen → umbral alto
    contactMinMs:    500,
    grabMaxMs:       750,
    scoreThreshold:  70,
    postContactMs:   5000,
    zoneEntryFrames: 3,
    hipConcealConf:  0.52,
    families:        ['MEDIUM','SMALL','BAG'],
    behaviors: {
      merodeo:       true,
      escaneo:       true,
      pantalla:      true,
      cadera:        true,
      manga:         true,
      agachado:      true,
      bagStuffing:   true,
      traspaso:      true,
      distractor:    true,
      trayectoria:   false,   // normal ir directo a estantería buscada
      pinchGrip:     true,
      coordinacion:  true,
    },
    scoreBonus: {
      bagStuffing:   48,      // meter libro en mochila es el robo más común
      pinchGrip:     12,      // menos relevante (se hojean libros normalmente)
    },
  },
};

// ─────────────────────────────────────────────────────────────────────
//  ALIASES — variantes de nombres aceptadas
// ─────────────────────────────────────────────────────────────────────
const ALIASES = {
  // Supermercado
  minimercado:       'supermercado',
  'mini mercado':    'supermercado',
  almacen:           'supermercado',
  almacén:           'supermercado',
  minimarket:        'supermercado',
  autoservicio:      'supermercado',
  // Kiosco
  cafeteria:         'kiosco',
  cafetería:         'kiosco',
  cafe:              'kiosco',
  café:              'kiosco',
  kiosko:            'kiosco',
  // Bazar
  tienda:            'bazar',
  ferreteria:        'bazar',
  ferretería:        'bazar',
  // Ropa
  vestimenta:        'ropa',
  indumentaria:      'ropa',
  // Depósito
  deposito:          'deposito',
  depósito:          'deposito',
  bodega:            'deposito',
  almacenamiento:    'deposito',
  // Electrónica
  electronica:       'electronica',
  electrónica:       'electronica',
  celulares:         'electronica',
  tecnologia:        'electronica',
  tecnología:        'electronica',
  // Estación
  estacion:          'estacion',
  estación:          'estacion',
  nafta:             'estacion',
  '24h':             'estacion',
  // Librería
  libreria:          'libreria',
  librería:          'libreria',
  papeleria:         'libreria',
  papelería:         'libreria',
  // Directos (sin alias necesario pero por claridad)
  farmacia:          'farmacia',
  joyeria:           'joyeria',
  joyería:           'joyeria',
  ropa:              'ropa',
  cocina:            'cocina',
  deposito:          'deposito',
  bazar:             'bazar',
};

/**
 * Retorna el perfil para un tipo de local.
 * Merge con genérico como fallback para campos faltantes.
 */
export function getProfile(type = 'generico') {
  const normalized = (type || '').toLowerCase().trim();
  const key  = ALIASES[normalized] || normalized;
  const base  = PROFILES[key] || PROFILES.generico;
  const generic = PROFILES.generico;

  return {
    ...generic,
    ...base,
    behaviors:  { ...generic.behaviors,  ...(base.behaviors  || {}) },
    scoreBonus: { ...SCORE_BONUS_DEFAULTS, ...(base.scoreBonus || {}) },
  };
}

export function listProfiles() {
  return Object.entries(PROFILES).map(([key, p]) => ({
    key,
    name: p.name,
    icon: p.icon,
  }));
}

console.log('%c✅ store-profiles.js v6.1 cargado — ' +
  Object.keys(OBJ_FAMILIES).length + ' familias | ' +
  Object.keys(PROFILES).length + ' perfiles',
  'color:#00e676;font-weight:bold');