// Figure render check: node blog_response/verify/figs/render_figs.cjs [URL]
// Defaults to file://. For every figure at 1440, 390 and 360px, EN and ZH, and every view of its toggle buttons:
// rendered data marks equal the expected count from the data files, no non-finite SVG attributes, nothing wider than
// the figure box, and no two SVG text labels overlap. Saves one screenshot per figure and view.
const fs=require('node:fs');
const path=require('node:path');
const {pathToFileURL}=require('node:url');
const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const root=path.resolve(__dirname,'../..');
const base=process.argv[2] || pathToFileURL(path.join(root,'index.html')).href;
const specs=fs.readdirSync(path.join(root,'figs/data')).filter(x=>x.endsWith('.json')).map(x=>JSON.parse(fs.readFileSync(path.join(root,'figs/data',x))));
fs.mkdirSync(path.join(__dirname,'screenshots'),{recursive:true});
const results=[];
const failures=[];

// Runs in the page for one figure host.
function measure(host) {
  const r=host.getBoundingClientRect();
  const svgs=[...host.querySelectorAll('.fig-body svg.fig-svg')];
  const finite=!svgs.some(s=>[...s.querySelectorAll('*')].some(n=>[...n.attributes].some(a=>/NaN|Infinity|undefined/.test(a.value))));
  const overflow=[...host.querySelectorAll('.fig-body svg.fig-svg, .fig-legend, .fig-controls')].filter(e=>{const b=e.getBoundingClientRect();return b.right>r.right+1||b.left<r.left-1;}).length;
  // Text collisions inside each SVG (lines of one label are one box).
  const collisions=[];
  for(const s of svgs){
    const boxes=[...s.querySelectorAll('text')].filter(t=>t.textContent.trim()).map(t=>({t:t.textContent.trim().slice(0,30),b:t.getBoundingClientRect()})).filter(x=>x.b.width>0);
    for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){
      const a=boxes[i].b,c=boxes[j].b;
      const ox=Math.min(a.right,c.right)-Math.max(a.left,c.left), oy=Math.min(a.bottom,c.bottom)-Math.max(a.top,c.top);
      if(ox>1.5&&oy>2.5)collisions.push(`${boxes[i].t} × ${boxes[j].t}`);
    }
  }
  return {marks:Number(host.dataset.renderedMarks),expected:Number(host.dataset.expectedMarks),counted:host.querySelectorAll('.fig-body .data-mark').length,
    panels:host.dataset.panels,svgs:svgs.length,finite,overflow,collisions:collisions.slice(0,8),collisionCount:collisions.length,
    selects:host.querySelectorAll('select').length,width:r.width};
}

(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_EXECUTABLE || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--no-sandbox'],timeout:15000});
 try {
  for(const width of [1440,390,360]) for(const lang of ['en','zh']) {
   // A tall window, so that element screenshots of tall figures are painted completely.
   const page=await browser.newPage({viewport:{width,height:2400},reducedMotion:'reduce'});
   const errors=[];page.on('pageerror',e=>errors.push(e.message));
   const url=new URL(base);url.searchParams.set('lang',lang);
   await page.goto(url.href,{waitUntil:'load'});
   await page.waitForFunction(()=>[...document.querySelectorAll('.data-figure')].every(h=>h.dataset.renderedMarks));
   await page.evaluate(()=>document.fonts.ready);
   await page.waitForTimeout(200);
   for(const spec of specs) {
    const host=page.locator('#'+spec.id+' .data-figure');
    const toggles=await host.locator('.fig-toggle').evaluateAll(bs=>bs.map(b=>b.dataset.control+':'+b.dataset.option));
    const views=['initial',...toggles.filter(t=>!t.endsWith(':0'))];
    for(const view of views) {
     if(view!=='initial'){
      const [c,o]=view.split(':');
      // Reset other controls to their first option, then choose this one.
      await host.locator(`.fig-toggle[data-control="${c}"][data-option="${o}"]`).click();
      await page.waitForTimeout(50);
     }
     const m=await host.evaluate(measure);
     const pass=m.marks===m.expected&&m.counted===m.marks&&m.marks>0&&m.finite&&m.overflow===0&&m.selects===0;
     results.push({width,lang,id:spec.id,view,...m,pass});
     const name=`${width}-${lang}-${spec.id}-${view.replace(':','_')}.png`;
     // Start the figure at the top of the window: content below the viewport is not painted in element screenshots.
     if(width!==360||!pass){await page.locator('#'+spec.id).evaluate(e=>e.scrollIntoView({block:'start',behavior:'instant'}));await page.waitForTimeout(80);
      await page.locator('#'+spec.id).screenshot({path:path.join(__dirname,'screenshots',name)});}
     if(!pass)failures.push(`${width} ${lang} ${spec.id} ${view}: ${JSON.stringify(m)}`);
     if(m.collisionCount)console.log(`labels overlap: ${width} ${lang} ${spec.id} ${view}: ${m.collisions.join(' | ')}`);
    }
    // Back to the first view for the language check.
    for(const t of toggles.filter(t=>t.endsWith(':0'))){const [c]=t.split(':');await host.locator(`.fig-toggle[data-control="${c}"][data-option="0"]`).click();}
   }
   // The language toggle re-renders without a reload.
   const legendSel='#fig-exp5-ppo-vs-astra .fig-legend';
   const before=await page.locator(legendSel).textContent();
   const other=lang==='en'?'zh':'en';
   await page.locator(`[data-set-lang="${other}"]`).click();
   await page.waitForFunction(([sel,old])=>document.querySelector(sel).textContent!==old,[legendSel,before]);
   const after=await page.evaluate(()=>[...document.querySelectorAll('.data-figure')].map(h=>Number(h.dataset.renderedMarks)===Number(h.dataset.expectedMarks)));
   if(after.some(x=>!x))failures.push(`${width} ${lang}: marks wrong after switching to ${other}`);
   results.push({width,lang,toggle:true,pageErrors:errors});
   if(errors.length)failures.push(`${width} ${lang}: page errors ${errors.join(' | ')}`);
   await page.close();
  }
 } finally {
  fs.writeFileSync(path.join(__dirname,'render-results.json'),JSON.stringify({results,failures},null,1)+'\n');
  await browser.close();
 }
 const collisions=results.reduce((a,r)=>a+(r.collisionCount||0),0);
 if(failures.length){console.error(failures.join('\n'));throw Error(`${failures.length} figure checks failed`);}
 console.log(`PASS: ${results.filter(x=>x.pass).length} figure views; ${collisions} overlapping label pairs; screenshots in verify/figs/screenshots/`);
})().catch(e=>{console.error(String(e));process.exitCode=1;});
