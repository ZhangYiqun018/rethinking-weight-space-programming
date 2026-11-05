// Run from any directory: node render.cjs [http://localhost:8000/].
// PLAYWRIGHT_MODULE and CHROME_EXECUTABLE can override the local installation paths.
// Checks the floating table of contents (js/nav.js) at every width x height x language x scroll position: the sidebar
// panel or the contents button must never intersect anything the page draws (text, list markers, images, SVGs,
// painted boxes), nor the language switch or the back-to-top button. "Content" here is found independently of
// nav.js: every text node's line boxes, every replaced element and every box with a background, border or shadow.
// Writes screenshots/*.png and measurements_screenshots.json next to this file.
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const root = __dirname;
const shots = path.join(root, 'screenshots');
fs.mkdirSync(shots, {recursive: true});
const widths = (process.env.WIDTHS || '768,1024,1280,1366,1440,1680,1920,2560').split(',').map(Number);
const heights = (process.env.HEIGHTS || '900,1100').split(',').map(Number);
const langs = ['en', 'zh'];
const positions = ['top', 'tldr', 'q1', 'middle', 'q3', 'article-end', 'bottom'];
const base = process.argv[2] || 'http://127.0.0.1:8000/';
const results = [];
const configs = [];
const failures = [];
const fail = (msg) => { failures.push(msg); console.error('FAIL ' + msg); };

const settle = page => page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 30)))));

// Runs in the page: the nav's state, and every intersection of its floating parts with content.
function inspect() {
  const nav = document.getElementById('toc');
  const toggle = document.getElementById('toc-toggle');
  const panel = document.getElementById('toc-panel');
  const chrome = [document.querySelector('.lang-switch'), document.getElementById('back-to-top')].filter(Boolean);
  const rect = el => { const r = el.getBoundingClientRect(); return {left: r.left, top: r.top, right: r.right, bottom: r.bottom, width: r.width, height: r.height}; };
  const shown = el => { if (!el || el.hidden) return false; const cs = getComputedStyle(el); const r = el.getBoundingClientRect();
    return cs.display !== 'none' && cs.visibility !== 'hidden' && r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth; };
  const mode = nav.classList.contains('is-sidebar') ? 'sidebar' : nav.classList.contains('is-button') ? 'button' : nav.classList.contains('is-inline') ? 'inline' : 'none';
  const floating = [];
  if (mode === 'sidebar' && shown(panel)) floating.push({name: 'sidebar', r: rect(panel)});
  if ((mode === 'button') && shown(toggle)) floating.push({name: 'button', r: rect(toggle)});
  const skip = el => nav.contains(el) || chrome.some(c => c.contains(el));
  const hit = (a, b, pad = 0) => Math.min(a.right + pad, b.right) - Math.max(a.left - pad, b.left) > 0.5 &&
    Math.min(a.bottom + pad, b.bottom) - Math.max(a.top - pad, b.top) > 0.5;
  const inView = b => b.bottom > 0 && b.top < innerHeight && b.right > 0 && b.left < innerWidth;
  const ink = [];
  if (floating.length) {
    const reach = floating.map(f => f.r);
    const near = b => reach.some(r => hit(r, b, 64));
    // What a box itself draws: its whole area for a background or shadow, else only its border strips.
    const transparent = c => c === 'transparent' || /rgba\(.*,\s*0\)$/.test(c);
    const paint = (cs, r) => {
      if (!transparent(cs.backgroundColor) || cs.backgroundImage !== 'none' || cs.boxShadow !== 'none') return [r];
      const out = [];
      const w = s => parseFloat(cs['border' + s + 'Width']) > 0 && cs['border' + s + 'Style'] !== 'none' && !transparent(cs['border' + s + 'Color']) ? parseFloat(cs['border' + s + 'Width']) : 0;
      if (w('Top')) out.push({left: r.left, right: r.right, top: r.top, bottom: r.top + w('Top')});
      if (w('Bottom')) out.push({left: r.left, right: r.right, top: r.bottom - w('Bottom'), bottom: r.bottom});
      if (w('Left')) out.push({left: r.left, right: r.left + w('Left'), top: r.top, bottom: r.bottom});
      if (w('Right')) out.push({left: r.right - w('Right'), right: r.right, top: r.top, bottom: r.bottom});
      return out;
    };
    // A list marker sits left of its item: about 1.2em for a bullet, one digit-width per character plus 0.5em for a number.
    const markerWidth = (el, cs) => {
      const fs = parseFloat(cs.fontSize);
      if (/disc|circle|square/.test(cs.listStyleType)) return 1.2 * fs;
      const n = [...el.parentElement.children].indexOf(el) + (el.parentElement.start || 1);
      return (String(n).length + 1) * 0.6 * fs + 0.5 * fs;
    };
    for (const el of document.body.querySelectorAll('*')) {
      if (skip(el)) continue;
      const r = el.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0 || !inView(r) || !near(r)) continue;
      const cs = getComputedStyle(el);
      if (cs.visibility === 'hidden' || cs.display === 'none') continue;
      const tag = el.tagName;
      const cls = String(el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className).slice(0, 60);
      const boxes = /^(IMG|VIDEO|CANVAS|IFRAME|INPUT|BUTTON|SELECT|TEXTAREA)$/.test(tag) || el instanceof SVGSVGElement ? [rect(el)] : paint(cs, rect(el));
      for (const b of boxes) ink.push({what: 'box', tag, id: el.id, cls, r: b});
      if (cs.display === 'list-item' && cs.listStyleType !== 'none' && cs.listStylePosition === 'outside') {
        const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.5;
        ink.push({what: 'marker', tag, id: el.id, cls, r: {left: r.left - markerWidth(el, cs), right: r.left, top: r.top, bottom: r.top + lh}});
      }
    }
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const range = document.createRange();
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (!n.nodeValue.trim()) continue;
      const p = n.parentElement;
      if (!p || skip(p) || p.closest('script, style')) continue;
      const pr = p.getBoundingClientRect();
      if (!inView(pr) || !near(pr)) continue;
      if (getComputedStyle(p).visibility === 'hidden') continue;
      range.selectNodeContents(n);
      for (const r of range.getClientRects()) if (r.width > 0 && inView(r)) ink.push({what: 'text', tag: p.tagName, id: p.id, cls: n.nodeValue.trim().slice(0, 40), r: {left: r.left, top: r.top, right: r.right, bottom: r.bottom}});
    }
    for (const c of chrome) if (shown(c)) ink.push({what: 'chrome', tag: c.tagName, id: c.id, cls: c.className, r: rect(c)});
  }
  const hits = [];
  const clearance = {};
  for (const f of floating) {
    let best = Infinity;
    for (const b of ink) {
      if (hit(f.r, b.r)) hits.push({floating: f.name, ...b});
      // Horizontal air between the floating part and content on the same rows.
      if (Math.min(f.r.bottom, b.r.bottom) > Math.max(f.r.top, b.r.top) && b.r.left >= f.r.right - 0.5) best = Math.min(best, b.r.left - f.r.right);
    }
    clearance[f.name] = best === Infinity ? null : Math.round(best * 10) / 10;
  }
  const tldr = document.querySelector('.tldr').getBoundingClientRect();
  return {mode, form: toggle.dataset.form || null, obstructed: toggle.classList.contains('is-obstructed'),
    open: nav.classList.contains('is-open'), scrollY, floating, tldrTop: tldr.top, tldrLeft: tldr.left,
    hits: hits.slice(0, 12), hitCount: hits.length, inkChecked: ink.length, clearance,
    panelScrolls: panel.scrollHeight > panel.clientHeight + 1};
}

async function scrollToPosition(page, pos) {
  await page.evaluate(pos => {
    const max = document.documentElement.scrollHeight - innerHeight;
    const art = document.querySelector('d-article').getBoundingClientRect().bottom + scrollY;
    const top = pos === 'top' ? 0 : pos === 'tldr' ? document.querySelector('.tldr').getBoundingClientRect().top + scrollY - 24 :
      pos === 'q1' ? max * 0.25 : pos === 'middle' ? max * 0.5 : pos === 'q3' ? max * 0.75 :
      pos === 'article-end' ? art - innerHeight * 0.6 : max;
    scrollTo({top, behavior: 'instant'});
  }, pos);
  await settle(page);
}

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true, args: ['--no-sandbox'], timeout: 15000
  });
  try {
    for (const height of heights) for (const width of widths) for (const lang of langs) {
      const tag = `${width}x${height}-${lang}`;
      const context = await browser.newContext({viewport: {width, height}, reducedMotion: 'reduce'});
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', e => errors.push(String(e)));
      await page.goto(`${base}?lang=${lang}`, {waitUntil: 'networkidle'});
      await page.evaluate(() => document.fonts.ready);
      await page.waitForTimeout(150);
      await settle(page);
      // Reject a missing Distill template instead of testing a different layout silently.
      if (await page.locator('d-article').evaluate(el => getComputedStyle(el).display) !== 'grid') {
        throw new Error('Distill layout did not load; do not count this as a valid render');
      }
      const config = {width, height, lang, errors};
      for (const position of positions) {
        await scrollToPosition(page, position);
        const m = await page.evaluate(inspect);
        const name = `${tag}-${position}.png`;
        await page.screenshot({path: path.join(shots, name)});
        results.push({width, height, lang, position, screenshot: 'screenshots/' + name, ...m});
        if (m.hitCount) fail(`${tag} ${position}: ${m.mode}/${m.form} intersects ${m.hitCount} content boxes, e.g. ${JSON.stringify(m.hits[0])}`);
        if (m.mode === 'sidebar' && position === 'top') {
          const p = m.floating[0];
          config.sidebar = {left: p.r.left, width: p.r.width, top: p.r.top, tldrTop: m.tldrTop, tldrLeft: m.tldrLeft, clearance: m.clearance.sidebar, scrolls: m.panelScrolls};
          if (Math.abs(p.r.top - m.tldrTop) > 1.5) fail(`${tag}: sidebar top ${p.r.top} not aligned with TL;DR ${m.tldrTop}`);
        }
        if (m.mode === 'sidebar' && position === 'middle' && Math.abs(m.floating[0].r.top - 24) > 1.5) fail(`${tag}: sidebar not held at 24px (${m.floating[0].r.top})`);
        if (m.mode === 'button') {
          config.button = config.button || {form: m.form, positions: {}};
          config.button.positions[position] = m.obstructed ? 'hidden (obstructed)' : m.floating[0] ? `${Math.round(m.floating[0].r.left)},${Math.round(m.floating[0].r.top)} ${Math.round(m.floating[0].r.width)}x${Math.round(m.floating[0].r.height)} clear ${m.clearance.button}px` : 'not shown';
        }
      }
      config.mode = results[results.length - 1].mode;
      const nav = page.locator('#toc');

      if (config.mode === 'button') {
        // Overlay: opens, traps focus, Esc closes and returns focus, choosing an entry closes it.
        await scrollToPosition(page, 'middle');
        await page.locator('#toc-toggle').click();
        await settle(page);
        const o = await page.evaluate(inspect);
        await page.screenshot({path: path.join(shots, `${tag}-overlay.png`)});
        results.push({width, height, lang, position: 'middle+overlay', screenshot: `screenshots/${tag}-overlay.png`, ...o});
        if (!o.open || await page.locator('#toc-panel').isHidden()) fail(`${tag}: overlay did not open`);
        if (o.hitCount) fail(`${tag}: button intersects content with overlay open`);
        if (await page.locator('#toc-toggle').getAttribute('aria-expanded') !== 'true') fail(`${tag}: aria-expanded not true`);
        let trapped = true;
        for (let i = 0; i < 45; i++) {
          await page.keyboard.press(i % 9 === 8 ? 'Shift+Tab' : 'Tab');
          if (!await page.evaluate(() => document.getElementById('toc-panel').contains(document.activeElement))) { trapped = false; break; }
        }
        if (!trapped) fail(`${tag}: focus escaped the overlay`);
        await page.keyboard.press('Escape');
        await settle(page);
        if (await nav.evaluate(n => n.classList.contains('is-open'))) fail(`${tag}: Esc did not close`);
        if (!await page.evaluate(() => document.activeElement === document.getElementById('toc-toggle'))) fail(`${tag}: focus not returned to the button`);
        await page.locator('#toc-toggle').click();
        await page.locator('#toc-backdrop').click({position: {x: width - 10, y: height / 2}});
        await settle(page);
        if (await nav.evaluate(n => n.classList.contains('is-open'))) fail(`${tag}: backdrop click did not close`);
        await page.locator('#toc-toggle').click();
        await page.locator('#toc a[href="#exp5"]').click();
        await page.waitForTimeout(750);
        if (await nav.evaluate(n => n.classList.contains('is-open'))) fail(`${tag}: choosing an entry did not close`);
        config.overlay = {opens: o.open, trapped, esc: true};
      } else if (config.mode === 'sidebar') {
        await scrollToPosition(page, 'middle');
        await page.locator('#toc a[href="#exp5"]').click();
        await page.waitForTimeout(750);
      } else {
        await page.locator('#toc-toggle').click();
        await page.locator('#toc a[href="#exp5"]').click();
        await page.waitForTimeout(750);
      }
      if (!page.url().endsWith('#exp5')) fail(`${tag}: hash not updated (${page.url()})`);
      if (await page.locator('#toc a[href="#exp5"]').getAttribute('aria-current') !== 'location') fail(`${tag}: active-section highlight missing`);
      const headTop = await page.locator('#exp5').evaluate(el => el.getBoundingClientRect().top);
      if (Math.abs(headTop - 24) > 4) fail(`${tag}: scrolled to #exp5 at ${headTop}px, expected 24`);

      // A heading is active once it has passed 30% of the window height: put Experiment 5's heading at 25%.
      await page.evaluate(() => scrollTo({top: document.getElementById('exp5').getBoundingClientRect().top + scrollY - innerHeight * 0.25, behavior: 'instant'}));
      await page.waitForTimeout(250);
      await settle(page);
      if (await page.locator('#toc a[href="#exp5"]').getAttribute('aria-current') !== 'location') fail(`${tag}: Exp 5 not active with its heading at 25% of the window`);
      await page.evaluate(() => scrollTo({top: document.getElementById('exp5').getBoundingClientRect().top + scrollY - innerHeight * 0.4, behavior: 'instant'}));
      await page.waitForTimeout(250);
      await settle(page);
      if (await page.locator('#toc a[href="#exp4"]').getAttribute('aria-current') !== 'location') fail(`${tag}: Exp 4 should still be active with Exp 5's heading at 40%`);

      if (config.mode === 'sidebar') {
        // The active entry must be visible inside the sidebar, also deep in the page.
        await page.evaluate(() => scrollTo({top: document.getElementById('exp8-run-inventory').getBoundingClientRect().top + scrollY - 40, behavior: 'instant'}));
        await page.waitForTimeout(400);
        await settle(page);
        const v = await page.evaluate(() => {
          const a = document.querySelector('#toc a[aria-current="location"]');
          const p = document.getElementById('toc-panel').getBoundingClientRect();
          if (!a) return {id: null};
          const r = a.getBoundingClientRect();
          return {id: a.getAttribute('href'), inView: r.top >= p.top - 1 && r.bottom <= p.bottom + 1};
        });
        if (!v.inView) fail(`${tag}: active entry ${v.id} not kept in view in the sidebar`);
        await page.screenshot({path: path.join(shots, `${tag}-active-deep.png`)});
        config.activeKeptInView = v;
      }

      await page.locator('#exp5').evaluate(el => {
        const next = [...document.querySelectorAll('.back-to-summary a')].find(a => a.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_PRECEDING);
        next.click();
      });
      await page.waitForTimeout(750);
      if (!page.url().endsWith('#summary-table')) fail(`${tag}: back-to-summary link failed`);
      await page.evaluate(() => scrollTo({top: document.documentElement.scrollHeight, behavior: 'instant'}));
      await page.waitForTimeout(100);
      await page.locator('#back-to-top').click();
      await page.waitForTimeout(750);
      if (await page.evaluate(() => scrollY) > 2) fail(`${tag}: back-to-top failed`);

      // Switching language re-measures: record the mode in the other language and check again.
      const other = lang === 'en' ? 'zh' : 'en';
      await page.locator(`.lang-switch button[data-set-lang="${other}"]`).click();
      await page.waitForTimeout(150);
      await settle(page);
      await scrollToPosition(page, 'q1');
      const s = await page.evaluate(inspect);
      config.afterSwitch = {lang: other, mode: s.mode, form: s.form};
      if (s.hitCount) fail(`${tag}: after switching to ${other}, intersects content`);

      await page.emulateMedia({media: 'print'});
      if (await nav.evaluate(n => getComputedStyle(n).display) !== 'none') fail(`${tag}: navigation visible in print`);
      await page.emulateMedia({media: 'screen'});
      if (errors.length) fail(`${tag}: page errors ${errors.join(' | ')}`);
      configs.push(config);
      console.log(`${tag}: ${config.mode}${config.button ? '/' + config.button.form : ''}; after switch ${other}: ${s.mode}${s.form ? '/' + s.form : ''}`);
      await context.close();
    }
  } finally {
    fs.writeFileSync(path.join(root, 'measurements_screenshots.json'), JSON.stringify({configs, results, failures}, null, 1));
    await browser.close();
  }
  if (failures.length) throw new Error(`${failures.length} failures; inspect measurements_screenshots.json`);
  console.log(`PASS: ${results.length} viewport/scroll cases, ${configs.length} configurations`);
})().catch(error => {
  fs.writeFileSync(path.join(root, 'render-error.txt'), String(error));
  console.error(String(error)); process.exitCode = 1;
});
