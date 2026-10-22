// Álgebra lineal mínima sobre arreglos de filas (Array<Float64Array | number[]>).

export const zeros = (r, c) => Array.from({ length: r }, () => new Float64Array(c));
export const eye = n => { const m = zeros(n, n); for (let i = 0; i < n; i++) m[i][i] = 1; return m; };
export const T = A => { const r = A.length, c = A[0].length, o = zeros(c, r); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) o[j][i] = A[i][j]; return o; };
export function mul(A, B) {
  const r = A.length, k = B.length, c = B[0].length, o = zeros(r, c);
  for (let i = 0; i < r; i++) { const oi = o[i], ai = A[i]; for (let l = 0; l < k; l++) { const a = ai[l]; if (a === 0) continue; const bl = B[l]; for (let j = 0; j < c; j++) oi[j] += a * bl[j]; } }
  return o;
}
export const mv = (A, v) => A.map(row => { let s = 0; for (let j = 0; j < v.length; j++) s += row[j] * v[j]; return s; });
export const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
// X'WX y X'Wz con pesos por fila
export function crossW(X, w) {
  const p = X[0].length, o = zeros(p, p);
  for (let i = 0; i < X.length; i++) { const x = X[i], wi = w ? w[i] : 1; if (wi === 0) continue; for (let a = 0; a < p; a++) { const v = wi * x[a]; if (v === 0) continue; const oa = o[a]; for (let b = a; b < p; b++) oa[b] += v * x[b]; } }
  for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) o[a][b] = o[b][a];
  return o;
}
export function crossWv(X, w, z) {
  const p = X[0].length, o = new Float64Array(p);
  for (let i = 0; i < X.length; i++) { const v = (w ? w[i] : 1) * z[i]; if (v === 0) continue; const x = X[i]; for (let a = 0; a < p; a++) o[a] += v * x[a]; }
  return o;
}
// Cholesky (devuelve L o null si no es definida positiva)
export function chol(A) {
  const n = A.length, L = zeros(n, n);
  for (let j = 0; j < n; j++) {
    let s = A[j][j]; for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    if (!(s > 1e-12 * Math.max(1, Math.abs(A[j][j])))) return null;
    L[j][j] = Math.sqrt(s);
    for (let i = j + 1; i < n; i++) { let t = A[i][j]; for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k]; L[i][j] = t / L[j][j]; }
  }
  return L;
}
// inversa por Gauss-Jordan con pivoteo parcial; null si es singular
export function inv(A) {
  const n = A.length, M = A.map((r, i) => { const o = new Float64Array(2 * n); o.set(r); o[n + i] = 1; return o; });
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    const scale = Math.max(...A.map(r => Math.abs(r[c]))) || 1;
    if (Math.abs(M[piv][c]) < 1e-11 * scale) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    const d = M[c][c]; for (let j = 0; j < 2 * n; j++) M[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c]; if (f !== 0) for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[c][j]; }
  }
  return M.map(r => r.slice(n));
}
export function symInv(A) {
  const L = chol(A); if (!L) return inv(A);
  const n = A.length, Li = zeros(n, n);
  for (let i = 0; i < n; i++) { Li[i][i] = 1 / L[i][i]; for (let j = 0; j < i; j++) { let s = 0; for (let k = j; k < i; k++) s -= L[i][k] * Li[k][j]; Li[i][j] = s / L[i][i]; } }
  const o = zeros(n, n);
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { let s = 0; for (let k = i; k < n; k++) s += Li[k][i] * Li[k][j]; o[i][j] = o[j][i] = s; }
  return o;
}
export function logdet(A) {
  const L = chol(A); if (L) { let s = 0; for (let i = 0; i < A.length; i++) s += 2 * Math.log(L[i][i]); return s; }
  // LU para matrices no definidas positivas
  const n = A.length, M = A.map(r => Float64Array.from(r)); let s = 0;
  for (let c = 0; c < n; c++) {
    let piv = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (M[piv][c] === 0) return -Infinity; [M[c], M[piv]] = [M[piv], M[c]];
    s += Math.log(Math.abs(M[c][c]));
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let j = c; j < n; j++) M[r][j] -= f * M[c][j]; }
  }
  return s;
}
export const sub = (A, idx) => idx.map(i => Float64Array.from(idx.map(j => A[i][j])));
export const quadForm = (v, A) => { let s = 0; for (let i = 0; i < v.length; i++) for (let j = 0; j < v.length; j++) s += v[i] * A[i][j] * v[j]; return s; };
