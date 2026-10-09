/* FinPulse motion engine — GSAP driven, installed once per browser session.
   Streamlit re-renders the DOM on every interaction, so instead of animating once
   we watch the DOM and animate any new node that carries a data-* motion hook:
     data-split            headline split into words/chars and revealed
     data-reveal           fade / rise / unblur when scrolled into view
     data-count            number counts up (from its previous value when data-key is set)
     data-bar              bar grows to its width
     data-gauge-arc        SVG gauge arc sweeps to its value
     data-gauge-needle     gauge needle rotates to its angle
     data-ring             SVG ring fills to its value
     data-marquee          infinite ticker
     [class*=st-key-card]  Streamlit card containers reveal
*/
(function () {
  if (window.__fpMotion) { window.__fpMotion.scan(); return; }

  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  // Hide motion targets until GSAP has set their start state (never longer than 3.5s)
  const root = document.documentElement;
  if (!reduce) root.classList.add('fp-wait');
  const unwait = () => root.classList.remove('fp-wait');
  setTimeout(unwait, 3500);
  const memory = new Map();         // data-key -> last value (so updates tween from the old value)
  const seen = new WeakSet();
  let gsapReady = false;
  const queue = [];

  const CDN = 'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.13.0/';
  function load(src) {
    return new Promise((res, rej) => {
      if (document.querySelector(`script[src="${src}"]`)) { res(); return; }
      const s = document.createElement('script');
      s.src = src; s.async = true; s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  const fmt = (v, d, p, s) =>
    p + Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) + s;

  // ---------------------------------------------------------------- reveal on view
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      io.unobserve(e.target);
      const el = e.target;
      const delay = parseFloat(el.dataset.delay || 0);
      gsap.to(el, { opacity: 1, y: 0, filter: 'blur(0px)', duration: 1.1, delay, ease: 'expo.out', clearProps: 'filter,transform' });
      el.querySelectorAll('[data-count]').forEach((n) => countUp(n, delay));
      el.querySelectorAll('[data-bar]').forEach((n) => growBar(n, delay + 0.15));
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

  const revealed = new Set();      // data-rid of live blocks that already played their entrance
  function reveal(el) {
    if (reduce) return;
    // Live blocks are re-rendered on every input change: reveal them once, then only tween numbers
    const rid = el.dataset.rid;
    if (rid) {
      if (revealed.has(rid)) { el.querySelectorAll('[data-count]').forEach((n) => countUp(n)); return; }
      revealed.add(rid);
    }
    gsap.set(el, { opacity: 0, y: 26, filter: 'blur(8px)' });
    io.observe(el);
  }

  // ---------------------------------------------------------------- numbers / bars
  function countUp(el, delay = 0) {
    if (el.__fpCounted) return;
    el.__fpCounted = true;
    const to = parseFloat(el.dataset.count);
    const d = parseInt(el.dataset.decimals || 0, 10);
    const p = el.dataset.prefix || '', s = el.dataset.suffix || '';
    const key = el.dataset.key;
    const from = key && memory.has(key) ? memory.get(key) : 0;
    if (key) memory.set(key, to);
    if (reduce || from === to) { el.textContent = fmt(to, d, p, s); return; }
    const o = { v: from };
    el.textContent = fmt(from, d, p, s);
    gsap.to(o, { v: to, duration: key && from !== 0 ? 0.9 : 1.8, delay, ease: 'power3.out',
      onUpdate: () => { el.textContent = fmt(o.v, d, p, s); } });
  }

  function growBar(el, delay = 0) {
    if (el.__fpGrown || reduce) return;
    el.__fpGrown = true;
    gsap.fromTo(el, { width: '0%' }, { width: el.dataset.bar + '%', duration: 1.6, delay, ease: 'expo.out' });
  }

  // ---------------------------------------------------------------- gauge + ring
  function gaugeArc(el) {
    const key = el.dataset.key, to = parseFloat(el.dataset.gaugeArc);
    const from = memory.has(key) ? memory.get(key) : 0;
    memory.set(key, to);
    if (reduce) return;
    gsap.fromTo(el, { strokeDashoffset: 100 - from }, { strokeDashoffset: 100 - to, duration: from ? 1 : 2, ease: from ? 'power3.out' : 'expo.out', delay: from ? 0 : 0.2 });
  }
  function gaugeNeedle(el) {
    const key = el.dataset.key, to = parseFloat(el.dataset.gaugeNeedle);
    const from = memory.has(key) ? memory.get(key) : -90;
    memory.set(key, to);
    if (reduce) return;
    const o = { a: from };
    const draw = () => el.setAttribute('transform', `rotate(${o.a} 120 116)`);
    draw();
    gsap.to(o, { a: to, duration: from === -90 ? 2.2 : 1.1, ease: 'elastic.out(1, 0.55)', delay: from === -90 ? 0.2 : 0, onUpdate: draw });
  }
  function ring(el) {
    const to = parseFloat(el.dataset.ring);
    if (reduce) return;
    gsap.fromTo(el, { strokeDashoffset: 100 }, { strokeDashoffset: 100 - to, duration: 2.4, delay: 0.5, ease: 'expo.out' });
  }

  // ---------------------------------------------------------------- headline split
  function split(el) {
    if (reduce || !window.SplitText) { reveal(el); return; }
    gsap.set(el, { opacity: 1 });
    // Gradient-text accents (.fp-serif) are kept whole: splitting them would break background-clip
    const st = SplitText.create(el, { type: 'words,chars', mask: 'words', ignore: '.fp-serif', wordsClass: 'fp-w', charsClass: 'fp-c' });
    const tl = gsap.timeline({ delay: 0.05 });
    tl.from(st.chars, { yPercent: 115, opacity: 0, rotate: 6, duration: 1.15, ease: 'expo.out', stagger: 0.018 });
    const accents = el.querySelectorAll('.fp-serif');
    if (accents.length) {
      tl.from(accents, { yPercent: 45, opacity: 0, filter: 'blur(14px)', duration: 1.5, ease: 'expo.out', clearProps: 'filter' }, '-=0.95');
      gsap.to(accents, { backgroundPosition: '100% 50%', duration: 6, ease: 'sine.inOut', repeat: -1, yoyo: true });
    }
  }

  // ---------------------------------------------------------------- marquee
  function marquee(el) {
    if (reduce) return;
    el.classList.add('is-gsap');
    const tween = gsap.to(el, { xPercent: -50, duration: 46, ease: 'none', repeat: -1 });
    const host = el.parentElement;
    host.addEventListener('mouseenter', () => gsap.to(tween, { timeScale: 0.15, duration: 0.8 }));
    host.addEventListener('mouseleave', () => gsap.to(tween, { timeScale: 1, duration: 0.8 }));
  }

  // ---------------------------------------------------------------- scan
  function handle(el, fn) {
    if (seen.has(el)) return;
    seen.add(el);
    fn(el);
  }

  // Inline SVG is passed as a data attribute (Streamlit's sanitiser strips svg tags) and mounted here
  function hydrate(root = document) {
    root.querySelectorAll('[data-svg]').forEach((el) => {
      if (el.__fpMounted) return;
      el.__fpMounted = true;
      el.innerHTML = el.getAttribute('data-svg');
    });
  }

  function scan(root = document) {
    hydrate(root);
    if (!gsapReady) { queue.push(root); return; }
    root.querySelectorAll('[data-split]').forEach((el) => handle(el, split));
    root.querySelectorAll('[data-reveal]').forEach((el) => handle(el, reveal));
    root.querySelectorAll('[class*="st-key-card"]').forEach((el) => handle(el, reveal));
    root.querySelectorAll('[data-gauge-arc]').forEach((el) => handle(el, gaugeArc));
    root.querySelectorAll('[data-gauge-needle]').forEach((el) => handle(el, gaugeNeedle));
    root.querySelectorAll('[data-ring]').forEach((el) => handle(el, ring));
    root.querySelectorAll('[data-marquee]').forEach((el) => handle(el, marquee));
    // Counters / bars outside a reveal wrapper animate immediately
    root.querySelectorAll('[data-count]').forEach((el) => { if (!el.closest('[data-reveal]')) countUp(el); });
    root.querySelectorAll('[data-bar]').forEach((el) => { if (!el.closest('[data-reveal]')) growBar(el); });
  }

  let raf = 0;
  const mo = new MutationObserver(() => {
    hydrate();
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => scan());
  });
  hydrate();
  mo.observe(document.body, { childList: true, subtree: true });

  // ---------------------------------------------------------------- pointer effects
  function pointerFX() {
    // Spotlight that follows the cursor across cards
    document.addEventListener('pointermove', (e) => {
      const card = e.target.closest && e.target.closest('.fp-spot, [class*="st-key-card"]');
      if (!card) return;
      const r = card.getBoundingClientRect();
      card.style.setProperty('--mx', `${e.clientX - r.left}px`);
      card.style.setProperty('--my', `${e.clientY - r.top}px`);
    }, { passive: true });

    if (reduce || window.matchMedia('(pointer: coarse)').matches) return;

    // Magnetic primary buttons + CTA links
    const magnets = new WeakMap();
    document.addEventListener('pointermove', (e) => {
      const btn = e.target.closest && e.target.closest('[data-testid="stBaseButton-primary"], .st-key-cta_primary a, .st-key-cta_secondary a');
      if (!btn) return;
      let m = magnets.get(btn);
      if (!m) {
        m = { x: gsap.quickTo(btn, 'x', { duration: 0.6, ease: 'power3.out' }), y: gsap.quickTo(btn, 'y', { duration: 0.6, ease: 'power3.out' }) };
        magnets.set(btn, m);
        btn.addEventListener('pointerleave', () => { m.x(0); m.y(0); });
      }
      const r = btn.getBoundingClientRect();
      m.x((e.clientX - (r.left + r.width / 2)) * 0.22);
      m.y((e.clientY - (r.top + r.height / 2)) * 0.3);
    }, { passive: true });

    // Ambient orbs drift + gentle parallax with the pointer
    const orbs = document.querySelectorAll('.fp-orb');
    orbs.forEach((o, i) => {
      gsap.to(o, { x: `random(-120, 120)`, y: `random(-80, 80)`, scale: `random(0.85, 1.2)`, duration: 14 + i * 4,
        ease: 'sine.inOut', repeat: -1, repeatRefresh: true, yoyo: true });
    });
    const amb = document.querySelector('.fp-ambient');
    if (amb) {
      const qx = gsap.quickTo(amb, 'x', { duration: 2, ease: 'power3.out' });
      const qy = gsap.quickTo(amb, 'y', { duration: 2, ease: 'power3.out' });
      window.addEventListener('pointermove', (e) => {
        qx((e.clientX / window.innerWidth - 0.5) * -30);
        qy((e.clientY / window.innerHeight - 0.5) * -30);
      }, { passive: true });
    }

    // Satellite orbiting the hero ring
    const orbit = () => document.querySelectorAll('.fp-satellite').forEach((s) => {
      if (s.__fpOrbit) return; s.__fpOrbit = true;
      gsap.set(s, { xPercent: 0, yPercent: 0 });
      const o = { a: 0 };
      gsap.to(o, { a: Math.PI * 2, duration: 9, ease: 'none', repeat: -1, onUpdate: () => {
        gsap.set(s, { x: Math.cos(o.a) * 103, y: Math.sin(o.a) * 103 });
      } });
    });
    orbit();
    setInterval(orbit, 1200);
  }

  window.__fpMotion = { scan: () => scan() };

  // ---------------------------------------------------------------- boot
  load(CDN + 'gsap.min.js')
    .then(() => load(CDN + 'SplitText.min.js').catch(() => null))
    .then(() => {
      if (window.SplitText) gsap.registerPlugin(SplitText);
      gsapReady = true;
      pointerFX();
      scan();
      queue.length = 0;
      requestAnimationFrame(unwait);
    })
    .catch(() => { unwait(); /* No network: the page still renders final values without motion. */ });
})();
