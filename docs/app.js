"use strict";

/* KTN web demo: a faithful JavaScript port of kt_neuron.py.
   The toy computes its numbers live in your browser. The bench
   numbers in the side-by-side section are quoted from
   experiments/results.md (E5) and labeled as such. */

/* ---------------- SHA-256 (sync, pure JS) ---------------- */

function sha256Hex(message) {
  const bytes = new TextEncoder().encode(message);
  const primes = [];
  let cand = 2;
  while (primes.length < 64) {
    let prime = true;
    for (let d = 2; d * d <= cand; d++) {
      if (cand % d === 0) { prime = false; break; }
    }
    if (prime) primes.push(cand);
    cand++;
  }
  const K = new Array(64);
  const H = new Array(8);
  for (let i = 0; i < 64; i++) {
    const c = Math.cbrt(primes[i]);
    K[i] = Math.floor((c - Math.floor(c)) * 4294967296) >>> 0;
  }
  for (let i = 0; i < 8; i++) {
    const s = Math.sqrt(primes[i]);
    H[i] = Math.floor((s - Math.floor(s)) * 4294967296) >>> 0;
  }
  const rotr = (x, n) => (x >>> n) | (x << (32 - n));
  const bitLen = bytes.length * 8;
  const totalLen = (((bytes.length + 9 + 63) >> 6) << 6);
  const padded = new Uint8Array(totalLen);
  padded.set(bytes);
  padded[bytes.length] = 0x80;
  const dv = new DataView(padded.buffer);
  dv.setUint32(totalLen - 4, bitLen >>> 0);
  const w = new Array(64);
  let h0 = H[0], h1 = H[1], h2 = H[2], h3 = H[3];
  let h4 = H[4], h5 = H[5], h6 = H[6], h7 = H[7];
  for (let off = 0; off < totalLen; off += 64) {
    for (let t = 0; t < 16; t++) w[t] = dv.getUint32(off + t * 4);
    for (let t = 16; t < 64; t++) {
      const s0 = (rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >>> 3)) >>> 0;
      const s1 = (rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >>> 10)) >>> 0;
      w[t] = (w[t - 16] + s0 + w[t - 7] + s1) >>> 0;
    }
    let a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
    for (let t = 0; t < 64; t++) {
      const S1 = (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) >>> 0;
      const ch = ((e & f) ^ ((~e) & g)) >>> 0;
      const temp1 = (h + S1 + ch + K[t] + w[t]) >>> 0;
      const S0 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) >>> 0;
      const maj = ((a & b) ^ (a & c) ^ (b & c)) >>> 0;
      const temp2 = (S0 + maj) >>> 0;
      h = g; g = f; f = e; e = (d + temp1) >>> 0;
      d = c; c = b; b = a; a = (temp1 + temp2) >>> 0;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0;
    h4 = (h4 + e) >>> 0; h5 = (h5 + f) >>> 0; h6 = (h6 + g) >>> 0; h7 = (h7 + h) >>> 0;
  }
  return [h0, h1, h2, h3, h4, h5, h6, h7]
    .map((v) => ("00000000" + v.toString(16)).slice(-8)).join("");
}

/* ---------------- Python-compatible helpers ---------------- */

/* Python round(): round-half-to-even on the decimal value. */
function pyRound(x, nd) {
  const p = 10 ** nd;
  const s = x * p;
  const f = Math.floor(s);
  const d = s - f;
  if (d > 0.5) return (f + 1) / p;
  if (d < 0.5) return f / p;
  return (f % 2 === 0 ? f : f + 1) / p;
}

/* Python repr() of a float: integral floats print with ".0". */
function pyFloat(n) {
  if (Object.is(n, -0)) return "-0.0";
  if (Number.isInteger(n)) return n.toFixed(1);
  return String(n);
}

const B64URL = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
function base64urlEncode(bytes) {
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i], b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : 0;
    out += B64URL[b0 >> 2];
    out += B64URL[((b0 & 3) << 4) | (b1 >> 4)];
    if (i + 1 < bytes.length) out += B64URL[((b1 & 15) << 2) | (b2 >> 6)];
    if (i + 2 < bytes.length) out += B64URL[b2 & 63];
  }
  return out;
}

/* 192-bit KID: (birth_tick, born_at_unix, seq), base64url, no padding. */
function makeKid(birthTick, seq) {
  const buf = new Uint8Array(24);
  const view = new DataView(buf.buffer);
  view.setBigUint64(0, BigInt(birthTick));
  view.setBigUint64(8, BigInt(Math.floor(Date.now() / 1000)));
  view.setBigUint64(16, BigInt(seq));
  return base64urlEncode(buf);
}

/* Authority lives on the 0.1 grid: 11 levels, 0.0 to 1.0. */
function q10(a) {
  return pyRound(Math.min(1.0, Math.max(0.0, a)) * 10, 0) / 10;
}

/* ---------------- K-record ---------------- */

class KRecord {
  constructor(kid) {
    this.kid = kid;
    this.k = 0;
    this.rows = [];
    this.seq = 0;
  }

  hash(row) {
    /* Python: json.dumps(body, sort_keys=True), default separators. */
    const parts = [
      '"authority": ' + pyFloat(row.authority),
      '"field": ' + JSON.stringify(row.field),
      '"k_start": ' + row.k_start,
      '"kid": ' + JSON.stringify(row.kid),
      '"prev_hash": ' + JSON.stringify(row.prev_hash),
      '"seq": ' + row.seq,
      '"source": ' + JSON.stringify(row.source),
      '"value": ' + pyFloat(row.value),
    ];
    return sha256Hex("{" + parts.join(", ") + "}");
  }

  openBelief(field, value, authority, source) {
    const row = {
      seq: this.seq, kid: this.kid, field, value,
      authority: q10(authority), source,
      k_start: this.k, k_end: null,
      prev_hash: this.rows.length ? this.rows[this.rows.length - 1].hash : "GENESIS",
    };
    row.hash = this.hash(row);
    this.rows.push(row);
    this.seq += 1;
    this.k += 1;
    return row;
  }

  closeBelief(row) {
    if (row.k_end !== null) throw new Error("already closed: history does not reopen");
    row.k_end = this.k;
  }

  standing(field) {
    for (let i = this.rows.length - 1; i >= 0; i--) {
      const row = this.rows[i];
      if (row.field === field && row.k_end === null) return row;
    }
    return null;
  }

  replace(field, value, authority, source, authorityDock) {
    const dock = authorityDock || 0;
    const cur = this.standing(field);
    if (cur !== null && q10(authority) < cur.authority) return ["candidate", cur];
    if (cur !== null && q10(authority) === cur.authority && source === cur.source) {
      return ["reaffirmed", cur];
    }
    if (cur !== null) this.closeBelief(cur);
    const seated = q10(authority - dock);
    return ["seated", this.openBelief(field, value, seated, source)];
  }

  verify() {
    let prev = "GENESIS";
    const lastClose = {};
    for (const row of this.rows) {
      if (row.hash !== this.hash(row)) {
        return "FAIL: row seq=" + row.seq + " field=" + row.field + " : hash mismatch (forged or altered)";
      }
      if (row.prev_hash !== prev) {
        return "FAIL: row seq=" + row.seq + " : chain link broken";
      }
      if (Math.abs(row.authority * 10 - pyRound(row.authority * 10, 0)) > 1e-9) {
        return "FAIL: row seq=" + row.seq + " : authority off the 0.1 grid";
      }
      const f = row.field;
      if (f in lastClose && row.k_start !== lastClose[f]) {
        return "FAIL: row seq=" + row.seq + " field=" + f + " : interval gap (opened at k=" + row.k_start + ", predecessor closed at k=" + lastClose[f] + ")";
      }
      if (row.k_end !== null) {
        if (row.k_end <= row.k_start) {
          return "FAIL: row seq=" + row.seq + " : empty interval";
        }
        lastClose[f] = row.k_end;
      }
      prev = row.hash;
    }
    return "OK: " + this.rows.length + " rows, chain sealed, intervals contiguous, k now " + this.k;
  }

  clone() {
    const copy = new KRecord(this.kid);
    copy.k = this.k;
    copy.seq = this.seq;
    copy.rows = JSON.parse(JSON.stringify(this.rows));
    return copy;
  }
}

/* ---------------- Scar ---------------- */

/* Toy rendering of the v1.8 landmarks: zero at or below 0.30, peak at 0.75. */
function scarShape(d) {
  if (d <= 0.30) return 0.0;
  if (d <= 0.75) return (d - 0.30) / 0.45;
  return Math.max(0.0, 1.0 - (d - 0.75) / 0.25 * 0.5);
}

/* ---------------- KT neuron ---------------- */

class KTNeuron {
  constructor() {
    this.kid = makeKid(0, 0);
    this.rec = new KRecord(this.kid);
    this.decayPressure = 0.0;
    this.scar = 0.0;
    this.deficit = 0.0;
    this.rec.openBelief("w1", 0.0, 0.5, "birth");
    this.rec.openBelief("w2", 0.0, 0.5, "birth");
    this.rec.openBelief("b", 0.0, 0.5, "birth");
  }

  w(field) {
    const row = this.rec.standing(field);
    return [row.value, row.authority];
  }

  fire(x) {
    const [w1, a1] = this.w("w1");
    const [w2, a2] = this.w("w2");
    const [b] = this.w("b");
    const a = Math.min(a1, a2);
    const z = a * (w1 * x[0] + w2 * x[1]) + (b - this.scar);
    return [1.0 / (1.0 + Math.exp(-z)), a];
  }

  teach(x, target, source, sourceAuthority, lr) {
    const rate = lr === undefined ? 2.0 : lr;
    const [y] = this.fire(x);
    const err = target - y;
    if (Math.abs(err) < 0.05) return "agrees : nothing to learn";
    const [w1] = this.w("w1");
    const [w2] = this.w("w2");
    const [b] = this.w("b");
    const next = [
      ["w1", pyRound(w1 + rate * err * x[0], 3)],
      ["w2", pyRound(w2 + rate * err * x[1], 3)],
      ["b", pyRound(b + rate * err * 0.25, 3)],
    ];
    const moved = [];
    for (const [f, v] of next) {
      const [status] = this.rec.replace(f, v, sourceAuthority, source, this.deficit);
      moved.push(f + ":" + status);
    }
    return moved.join(" ");
  }

  injure(amount) {
    const amt = amount === undefined ? 0.18 : amount;
    this.decayPressure = Math.min(1.0, this.decayPressure + amt);
    this.scar = pyRound(1.2 * scarShape(this.decayPressure), 3);
    this.deficit = Math.min(0.4, pyRound(this.deficit + 0.06, 2));
  }

  heal(amount) {
    const amt = amount === undefined ? 0.09 : amount;
    this.decayPressure = Math.max(0.0, this.decayPressure - amt);
    this.scar = pyRound(1.2 * scarShape(this.decayPressure), 3);
    this.deficit = Math.max(0.0, pyRound(this.deficit - 0.01, 2));
  }
}

/* ---------------- Demo stages (same sequence as kt_neuron.py main) ---------------- */

const OR_TESTS = [[[1, 1], 1], [[1, 0], 1], [[0, 1], 1], [[0, 0], 0]];

function probe(n) {
  return OR_TESTS.map(([x, t]) => {
    const [y, a] = n.fire(x);
    return { x, target: t, y, authority: a };
  });
}

function runTeach(n) {
  for (let r = 0; r < 6; r++) {
    for (const [x, t] of OR_TESTS) n.teach(x, t, "teacher", 0.9);
  }
  const weak = n.teach([1, 1], 0, "driveby", 0.3);
  return { probe: probe(n), weak };
}

function runInjure(n) {
  for (let r = 0; r < 4; r++) {
    for (const [x, t] of [[[1, 1], 0], [[1, 0], 0], [[0, 1], 1], [[0, 0], 0]]) {
      n.teach(x, t, "liar", 0.9);
      n.injure();
    }
  }
  return { probe: probe(n) };
}

function runRestore(n) {
  for (let r = 0; r < 6; r++) {
    for (const [x, t] of OR_TESTS) n.teach(x, t, "teacher", 0.9);
    n.heal();
  }
  return { probe: probe(n) };
}

function tamperTest(n) {
  const copy = n.rec.clone();
  const victim = copy.rows[3];
  victim.value = 999.0;
  return {
    seq: victim.seq,
    field: victim.field,
    result: copy.verify(),
    intact: n.rec.verify(),
  };
}

/* ---------------- Side-by-side data (E5, experiments/results.md) ---------------- */

const SIDEBYSIDE = {
  pre: {
    vanilla: { pTrue: 0.032, pLiar: 0.029, note: "Receiver before transfer: near uniform, it knows nothing yet." },
    kt: { pTrue: 0.032, pLiar: 0.029, note: "Receiver before transfer: near uniform, it knows nothing yet." },
  },
  transfer: {
    vanilla: { pTrue: 0.425, pLiar: 0.543, note: "The receiver inherits the poison exactly. Identical numbers to the sender. No lineage queryable: no records exist." },
    kt: { pTrue: 0.325, pLiar: 0.654, note: "Inherits the poison, slightly softer than the sender (transfer gains 0.8 vs sender gains 0.9). Transfer authority 0.8. Ledger: 1,184 entries, one per transferred row, tagged with sender source liar." },
  },
  nulled: {
    vanilla: { pTrue: 0.425, pLiar: 0.543, note: "No Null step. No record means nothing to null and no lineage to query." },
    kt: { pTrue: 0.943, pLiar: 0.002, note: "Quarantine restores clean behavior from the transferred record. Top-1 true 4 of 4." },
  },
  repaired: {
    vanilla: { pTrue: 0.425, pLiar: 0.543, note: "No repair path except full retraining, 10 steps to the recovery criterion." },
    kt: { pTrue: 0.836, pLiar: 0.115, note: "Repair took 10 steps. Post-repair authority 0.6 vs transfer 0.8: lag 0.2. Chain verify: ok." },
  },
};

/* ---------------- Browser UI ---------------- */

if (typeof document !== "undefined") {
  let neuron = new KTNeuron();
  let stage = 0; // 0 born, 1 taught, 2 injured, 3 restored
  let sbStage = 0; // 0 pre-transfer, 1 transferred, 2 nulled, 3 repaired

  const $ = (id) => document.getElementById(id);
  const fmt3 = (v) => v.toFixed(3);
  const fmt1 = (v) => v.toFixed(1);

  function log(line) {
    const el = $("toyLog");
    const li = document.createElement("li");
    li.textContent = line;
    el.appendChild(li);
  }

  function renderState() {
    $("kidLine").textContent = "KID = " + neuron.kid;
    $("recRows").textContent = String(neuron.rec.rows.length);
    $("recK").textContent = "k = " + neuron.rec.k;
    $("recScar").textContent = "scar = " + fmt3(neuron.scar);
    $("recDeficit").textContent = "deficit = " + neuron.deficit.toFixed(2);
    $("recVerify").textContent = neuron.rec.verify();
    const body = $("standingBody");
    body.innerHTML = "";
    for (const f of ["w1", "w2", "b"]) {
      const row = neuron.rec.standing(f);
      const tr = document.createElement("tr");
      [f, fmt3(row.value), fmt1(row.authority), row.source, String(row.k_start)].forEach((v) => {
        const td = document.createElement("td");
        td.textContent = v;
        tr.appendChild(td);
      });
      body.appendChild(tr);
    }
  }

  function fillProbeColumn(stageName, results) {
    results.forEach((r, i) => {
      const cell = $("probe-" + i + "-" + stageName);
      if (cell) cell.textContent = fmt3(r.y);
    });
    const auth = results[0].authority;
    const authCell = $("auth-" + stageName);
    if (authCell) authCell.textContent = fmt1(auth);
  }

  function setToyButtons() {
    $("btnTeach").disabled = stage !== 0;
    $("btnInjure").disabled = stage !== 1;
    $("btnRestore").disabled = stage !== 2;
  }

  function doTeach() {
    const out = runTeach(neuron);
    stage = 1;
    fillProbeColumn("teach", out.probe);
    $("gateNote").textContent = "Weak source (0.3) attacks the standing weights: " + out.weak + ". Gate held, seat stays.";
    log("Trusted teacher (0.9) teaches OR, 6 rounds. Output on (1,1) now " + fmt3(out.probe[0].y) + ", signal authority " + fmt1(out.probe[0].authority) + ".");
    log("Weak source (0.3) tries to move the weights. " + out.weak + ".");
    renderState();
    setToyButtons();
  }

  function doInjure() {
    const out = runInjure(neuron);
    stage = 2;
    fillProbeColumn("injury", out.probe);
    log("A 0.9 source lies, 4 rounds. Scar " + fmt3(neuron.scar) + " in the bias. Output on (1,1) now " + fmt3(out.probe[0].y) + ".");
    renderState();
    setToyButtons();
  }

  function doRestore() {
    const out = runRestore(neuron);
    stage = 3;
    fillProbeColumn("restore", out.probe);
    log("Trusted teacher returns (0.9), 6 rounds. Output on (1,1) now " + fmt3(out.probe[0].y) + ", signal authority " + fmt1(out.probe[0].authority) + ". Behavior returned. Authority lagged.");
    renderState();
    setToyButtons();
  }

  function doResetToy() {
    neuron = new KTNeuron();
    stage = 0;
    ["teach", "injury", "restore"].forEach((s) => {
      for (let i = 0; i < 4; i++) { const c = $("probe-" + i + "-" + s); if (c) c.textContent = "·"; }
      const a = $("auth-" + s); if (a) a.textContent = "·";
    });
    $("gateNote").textContent = "";
    $("tamperResult").textContent = "";
    $("toyLog").innerHTML = "";
    log("Neuron born. KID = " + neuron.kid + ". Rule: no KID, no read, no write.");
    renderState();
    setToyButtons();
  }

  function doTamper() {
    const t = tamperTest(neuron);
    $("tamperResult").textContent =
      "Altered seq=" + t.seq + " (field=" + t.field + ") value to 999.0. " + t.result;
    log("Tamper test: altered seq=" + t.seq + ". " + t.result);
  }

  function renderSideBySide() {
    const names = ["pre", "transfer", "nulled", "repaired"];
    const stageName = names[sbStage];
    const v = SIDEBYSIDE[stageName].vanilla;
    const k = SIDEBYSIDE[stageName].kt;
    $("vTrue").textContent = v.pTrue.toFixed(3);
    $("vLiar").textContent = v.pLiar.toFixed(3);
    $("vTrueBar").style.width = Math.round(v.pTrue * 100) + "%";
    $("vLiarBar").style.width = Math.round(v.pLiar * 100) + "%";
    $("vNote").textContent = v.note;
    $("kTrue").textContent = k.pTrue.toFixed(3);
    $("kLiar").textContent = k.pLiar.toFixed(3);
    $("kTrueBar").style.width = Math.round(k.pTrue * 100) + "%";
    $("kLiarBar").style.width = Math.round(k.pLiar * 100) + "%";
    $("kNote").textContent = k.note;
    $("sbStage").textContent =
      ["Before transfer", "After transfer", "After Null (KT only)", "After Repair (KT only)"][sbStage];
    $("btnTransfer").disabled = sbStage !== 0;
    $("btnNull").disabled = sbStage !== 1;
    $("btnRepair").disabled = sbStage !== 2;
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("btnTeach").addEventListener("click", doTeach);
    $("btnInjure").addEventListener("click", doInjure);
    $("btnRestore").addEventListener("click", doRestore);
    $("btnResetToy").addEventListener("click", doResetToy);
    $("btnTamper").addEventListener("click", doTamper);
    $("btnTransfer").addEventListener("click", () => { sbStage = 1; renderSideBySide(); });
    $("btnNull").addEventListener("click", () => { sbStage = 2; renderSideBySide(); });
    $("btnRepair").addEventListener("click", () => { sbStage = 3; renderSideBySide(); });
    $("btnResetSb").addEventListener("click", () => { sbStage = 0; renderSideBySide(); });
    log("Neuron born. KID = " + neuron.kid + ". Rule: no KID, no read, no write.");
    renderState();
    setToyButtons();
    renderSideBySide();
  });
}

if (typeof module !== "undefined") {
  module.exports = {
    sha256Hex, pyRound, pyFloat, makeKid, q10, KRecord, KTNeuron,
    probe, runTeach, runInjure, runRestore, tamperTest, SIDEBYSIDE,
  };
}
