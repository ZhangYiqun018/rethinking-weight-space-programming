// DOM-only smoke test. It executes the real renderer, but does not verify layout or fonts.
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
class Node {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.dataset={};this.style={setProperty(){}};this.events={};this.clientWidth=320;this._value=undefined;}
 setAttribute(k,v){this.attrs[k]=String(v);}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 addEventListener(k,f){this.events[k]=f;}
 get options(){return this.children;}
 get value(){return this._value===undefined?(this.tag==='select'?'0':''):this._value;}
 set value(v){this._value=v;}
 querySelectorAll(selector){const match=n=>selector==='*'||(selector[0]==='.'?(n.className||n.attrs.class||'').split(' ').includes(selector.slice(1)):n.tag===selector);return this.children.flatMap(n=>[...(match(n)?[n]:[]),...n.querySelectorAll(selector)]);}
 querySelector(s){return this.querySelectorAll(s)[0]||null;}
}
let total=0;
for(const width of [300,358,704]) for(const lang of ['en','zh']) {
 const figures={};const html=new Node('html');html.dataset.lang=lang;
 const context={window:{},document:{documentElement:html,readyState:'complete',createElement:t=>new Node(t),createElementNS:(ns,t)=>new Node(t),getElementById:id=>figures[id]},ResizeObserver:class{observe(){}},MutationObserver:class{observe(){}}};
 vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(root,'figs/data/bundle.js'),'utf8'),context);
 for(const spec of Object.values(context.window.WSP_FIGURES)){const f=new Node('figure'),h=new Node('div');h.className='data-figure';h.clientWidth=width;f.append(h);figures[spec.id]=f;}
 vm.runInContext(fs.readFileSync(path.join(root,'js/figs.js'),'utf8'),context);
 for(const spec of Object.values(context.window.WSP_FIGURES)){
  const f=figures[spec.id],select=f.querySelector('select');
  spec.panels.forEach((panel,i)=>{
   select.value=String(i);select.events.change();
   const svg=f.querySelector('svg');assert(svg);
   assert.equal(svg.querySelectorAll('.data-mark').length,panel.expected_marks,`${spec.id} panel ${i}`);
   for(const node of svg.querySelectorAll('*'))for(const value of Object.values(node.attrs))assert(!/NaN|Infinity/.test(value));
   total++;
  });
 }
}
console.log(`PASS: renderer executed ${total} panels (DOM-only; no visual-layout claim)`);
