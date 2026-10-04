/* Odds display format, shared by the predictor and the past-predictions page.
   Everything is stored and computed in decimal odds; this only changes how prices
   are shown and typed. The choice is remembered in this browser. */
const OddsFmt = (() => {
  const KEY = "oddsFormat";
  let fmt = "decimal";
  try { if (localStorage.getItem(KEY) === "american") fmt = "american"; } catch (e) { /* storage blocked */ }
  const listeners = [];

  // decimal → American: +150 for underdogs (2.50), −125 for favourites (1.80)
  function toAmerican(d){
    if (!(d > 1)) return null;
    return d >= 2 ? Math.round((d - 1) * 100) : -Math.round(100 / (d - 1));
  }
  function fromAmerican(a){
    if (!isFinite(a) || Math.abs(a) < 100) return NaN;
    return a > 0 ? 1 + a / 100 : 1 + 100 / -a;
  }
  // a decimal price as text in the chosen format
  function show(d, f = fmt){
    if (!(d > 1)) return "—";
    if (f === "decimal") return (+d).toFixed(2);
    const a = toAmerican(d);
    return a > 0 ? "+" + a : "−" + Math.abs(a);
  }
  // what someone typed → decimal odds (NaN if it isn't a price)
  function parse(text, f = fmt){
    const s = String(text ?? "").trim().replace(/[−–]/g, "-");
    if (!s) return NaN;
    const v = parseFloat(s);
    if (!isFinite(v)) return NaN;
    return f === "decimal" ? (v > 1 ? v : NaN) : fromAmerican(v);
  }
  function set(f){
    if (f !== "decimal" && f !== "american" || f === fmt) return;
    const old = fmt; fmt = f;
    try { localStorage.setItem(KEY, f); } catch (e) { /* storage blocked */ }
    document.querySelectorAll("[data-odds-toggle] button").forEach(b =>
      b.setAttribute("aria-pressed", b.dataset.fmt === f ? "true" : "false"));
    listeners.forEach(fn => fn(f, old));
  }
  // wire every <div class="seg" data-odds-toggle> on the page
  function bind(){
    document.querySelectorAll("[data-odds-toggle]").forEach(box => {
      box.innerHTML = `<button type="button" data-fmt="decimal">Decimal</button><button type="button" data-fmt="american">American</button>`;
      box.querySelectorAll("button").forEach(b => {
        b.setAttribute("aria-pressed", b.dataset.fmt === fmt ? "true" : "false");
        b.addEventListener("click", () => set(b.dataset.fmt));
      });
    });
  }
  return { get: () => fmt, set, show, parse, toAmerican, fromAmerican, bind, onChange: fn => listeners.push(fn) };
})();
