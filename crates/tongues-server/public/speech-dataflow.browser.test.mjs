import {test,expect} from "@playwright/test";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const publicRoot=path.dirname(fileURLToPath(import.meta.url));
let server,baseUrl,savedGraph,runSequence=0,holdRuns=false,meterStorm=false;
const activeRuns=new Map();
test.use({hasTouch:true});

const replacement=family=>({
  family,configuration_schema_id:`fixture.${family}.config`,configuration_schema_version:1,
  port_aliases:{},configuration_aliases:{},disconnect_ports:[],
});
const schema={type:"object",properties:{language:{type:"string",enum:["en","fr"]},timestamps:{type:"boolean"}},required:["language"]};
const ports=[
  {id:"audio",label:"audio",direction:"input",value_type:"audio_stream",cardinality:"one"},
  {id:"committed",label:"committed",direction:"output",value_type:"transcript_committed",cardinality:"many"},
];
const components=Object.fromEntries([
  ["base",{id:"base",node_kind:"asr",provider:"Fixture",model:"Base",readiness:"ready",capabilities:["asr"],configuration_schema:schema,default_config:{language:"en",timestamps:true},detail:"Installed base model",replacement:replacement("asr")}],
  ["alternate",{id:"alternate",node_kind:"asr",provider:"Alternate Labs",model:"Ready Model",readiness:"ready",capabilities:["asr"],configuration_schema:schema,default_config:{language:"fr",timestamps:false},detail:"Installed alternate model",replacement:replacement("asr")}],
  ["unavailable",{id:"unavailable",node_kind:"asr",provider:"Alternate Labs",model:"Missing Model",readiness:"unavailable",capabilities:["asr"],configuration_schema:schema,default_config:{language:"en"},detail:"Model files are absent",replacement:replacement("asr")}],
  ["tts-base",{id:"tts-base",node_kind:"tts",provider:"Fixture",model:"Voice",readiness:"ready",capabilities:["tts"],configuration_schema:{type:"object",properties:{voice:{type:"string",enum:["alto","tenor"],"x-ui-widget":"menu"}},required:["voice"]},default_config:{voice:"alto"},detail:"Installed fixture voice",replacement:replacement("tts")}],
  ...Array.from({length:1200},(_,index)=>[`bulk-${index}`,{id:`bulk-${index}`,node_kind:"asr",provider:`Provider ${index%12}`,model:`Catalog Model ${index}`,readiness:"ready",capabilities:["asr"],configuration_schema:schema,default_config:{language:"en"},detail:`Synthetic discovery fixture ${index}`,replacement:replacement("asr")}]),
]);
const discovery={
  schema_version:2,revision:"fixture-catalog-v1",
  node_kinds:{
    asr:{kind:"asr",label:"ASR",requires_component:true,required_capabilities:["asr"],configuration_schema:schema,default_config:{language:"en"},ports,replacement:replacement("asr")},
    microphone:{kind:"microphone",label:"Microphone",requires_component:false,default_config:{},configuration_schema:{type:"object"},ports:[
      {id:"out",label:"audio",direction:"output",value_type:"audio_stream",cardinality:"many",streaming:true},
    ],replacement:replacement("microphone")},
    transcript_sink:{kind:"transcript_sink",label:"Transcript",requires_component:false,default_config:{},configuration_schema:{type:"object"},ports:[
      {id:"in",label:"transcript",direction:"input",value_type:"transcript_committed",cardinality:"one",streaming:true},
    ],replacement:replacement("transcript_sink")},
    audio_passthrough:{kind:"audio_passthrough",label:"Audio pass-through",requires_component:false,default_config:{},configuration_schema:{type:"object"},ports:[
      {id:"in",label:"audio in",direction:"input",value_type:"audio_stream",cardinality:"one",streaming:true},
      {id:"out",label:"audio out",direction:"output",value_type:"audio_stream",cardinality:"many",streaming:true},
    ],replacement:replacement("audio_passthrough")},
    text_source:{kind:"text_source",label:"Text source",requires_component:false,default_config:{text:"Hello from Graph Studio"},configuration_schema:{type:"object",properties:{text:{type:"string","x-ui-widget":"short_text"}},required:["text"]},ports:[
      {id:"out",label:"text",direction:"output",value_type:"text",cardinality:"many",streaming:true},
    ],replacement:replacement("text_source")},
    tts:{kind:"tts",label:"TTS",requires_component:true,required_capabilities:["tts"],configuration_schema:{type:"object",properties:{voice:{type:"string",enum:["alto","tenor"],"x-ui-widget":"menu"}},required:["voice"]},default_config:{voice:"alto"},ports:[
      {id:"text",label:"text",direction:"input",value_type:"text",cardinality:"one",streaming:true},
      {id:"audio",label:"audio",direction:"output",value_type:"audio_stream",cardinality:"many",streaming:true},
    ],replacement:replacement("tts")},
    synthesis_fixture:{kind:"synthesis_fixture",label:"Long-label synthesis fixture",requires_component:false,configuration_schema:{type:"object",properties:{
      voice:{type:"string",title:"Voice with a deliberately long label",enum:["alto","tenor"],"x-ui-widget":"menu","x-ui-priority":0},
      rate:{type:"number",title:"Speaking rate in syllables per second",minimum:.5,maximum:4,step:.1,"x-ui-widget":"slider","x-ui-priority":1},
      pitch:{type:"number",title:"Pitch adjustment in semitones",minimum:-12,maximum:12,step:1,"x-ui-widget":"number","x-ui-priority":2},
    },required:["voice","rate","pitch"]},default_config:{voice:"alto",rate:1.5,pitch:0},ports:[
      {id:"text",label:"source text with a very long port label",direction:"input",value_type:"text",cardinality:"one",streaming:true},
      {id:"prosody",label:"prosody control",direction:"input",value_type:"control",cardinality:"one",streaming:true},
      {id:"audio",label:"synthesized audio with a very long port label",direction:"output",value_type:"audio_stream",cardinality:"many",streaming:true},
      {id:"events",label:"word boundary events",direction:"output",value_type:"control",cardinality:"many",streaming:true},
    ],replacement:replacement("synthesis_fixture")},
    audio_output:{kind:"audio_output",label:"Audio output",requires_component:false,default_config:{target:"browser",browser_device_id:"default",system_device_id:"default",wav_path:"data/speech-output.wav"},configuration_schema:{type:"object",required:["target"],properties:{
      target:{type:"string",title:"Destination",enum:["browser","system","wav"],"x-enum-labels":["This browser","Server audio device (CPAL)","WAV file"],"x-ui-priority":0},
      browser_device_id:{type:"string",title:"Browser playback device",default:"default",enum:["default"],"x-enum-labels":["Browser default"],"x-ui-visible-when":{target:"browser"},"x-ui-priority":10},
      system_device_id:{type:"string",title:"Server playback device",default:"default",enum:["default","Fixture speakers"],"x-enum-labels":["System default","Fixture speakers"],"x-ui-visible-when":{target:"system"},"x-ui-priority":10},
      wav_path:{type:"string",title:"WAV output path",default:"data/speech-output.wav",format:"path","x-ui-visible-when":{target:"wav"},"x-ui-priority":10},
    }},ports:[
      {id:"in",label:"audio",direction:"input",value_type:"audio_stream",cardinality:"one",streaming:true},
    ],replacement:replacement("audio_output")},
    transcript_merge:{kind:"transcript_merge",label:"Transcript merge",requires_component:false,merge:{strategy:"source_order"},default_config:{},configuration_schema:{type:"object"},ports:[
      {id:"in",label:"transcripts",direction:"input",value_type:"transcript_committed",cardinality:"many",streaming:true},
      {id:"out",label:"merged transcript",direction:"output",value_type:"transcript_committed",cardinality:"many",streaming:true},
    ],replacement:replacement("transcript_merge")},
  },
  components,
};
const graph={
  schema_version:2,graph_id:"pipeline:browser-fixture",revision:7,
  metadata:{name:"Browser fixture",description:"",allow_unsafe_execution:false,labels:{
    "studio.layout.v1":JSON.stringify({
      "node:asr":{x:420,y:200},"node:mic-1":{x:120,y:150},"node:mic-2":{x:120,y:310},
      "node:sink-1":{x:720,y:140},"node:sink-2":{x:720,y:300},
    }),
    "studio.node-faceplate-geometry.v1":JSON.stringify({
      "node:asr":{width:360,height:186,collapsed_height:74},
      "node:mic-1":{width:210,height:130,collapsed_height:60},
    }),
  }},
  nodes:[
    {id:"node:asr",kind:"asr",component_id:"base",config:{language:"en",timestamps:true},disabled:false,bypassed:false},
    {id:"node:mic-1",kind:"microphone",component_id:null,config:{},disabled:false,bypassed:false},
    {id:"node:mic-2",kind:"microphone",component_id:null,config:{},disabled:false,bypassed:false},
    {id:"node:sink-1",kind:"transcript_sink",component_id:null,config:{},disabled:false,bypassed:false},
    {id:"node:sink-2",kind:"transcript_sink",component_id:null,config:{},disabled:false,bypassed:false},
  ],
  edges:[],selected_sinks:[{node_id:"node:asr",port_id:"committed"}],
};

function readRequestJson(request){
  return new Promise((resolve,reject)=>{
    let body="";
    request.setEncoding("utf8");
    request.on("data",chunk=>body+=chunk);
    request.on("end",()=>{try{resolve(body?JSON.parse(body):{});}catch(error){reject(error);}});
    request.on("error",reject);
  });
}

function writeRunEvent(response,event){
  response.write(`${JSON.stringify(event)}\n`);
}

function finishRun(runId,status,kind,detail){
  const run=activeRuns.get(runId);
  if(!run||run.closed)return;
  run.status=status;
  run.closed=true;
  run.timers.forEach(clearTimeout);
  if(status==="completed"&&run.artifact){
    writeRunEvent(run.response,{run_id:runId,status:"monitoring",node_id:run.artifact.node_id,kind:"artifact",elapsed_ms:24,artifact:run.artifact});
  }
  writeRunEvent(run.response,{run_id:runId,status,node_id:run.nodeId,kind,elapsed_ms:25,detail});
  run.response.end();
}

function largeGraph(nodeCount=180){
  const document=structuredClone(graph);
  document.graph_id="pipeline:large-fixture";
  document.revision=11;
  document.metadata.name="Large interaction fixture";
  document.nodes=Array.from({length:nodeCount},(_,index)=>({
    id:`node:large-${index}`,kind:index%2?"transcript_sink":"microphone",component_id:null,config:{},disabled:false,bypassed:false,
  }));
  document.edges=[];
  document.selected_sinks=[];
  document.metadata.labels["studio.layout.v1"]=JSON.stringify(Object.fromEntries(document.nodes.map((node,index)=>[
    node.id,{x:120+(index%18)*240,y:120+Math.floor(index/18)*180},
  ])));
  return document;
}

function renderedBoundaryGraph(){
  const document=structuredClone(graph);
  document.graph_id="pipeline:rendered-boundary-fixture";
  document.revision=19;
  document.metadata.name="Rendered boundary fixture";
  document.metadata.labels={
    "studio.layout.v1":JSON.stringify({
      "node:text":{x:100,y:245},
      "node:synth":{x:390,y:245},
      "node:output":{x:735,y:245},
    }),
    "studio.node-faceplate.v1":JSON.stringify({collapsed:{"node:text":true}}),
    "studio.node-faceplate-geometry.v1":JSON.stringify({
      "node:text":{width:180,height:170,collapsed_height:76},
      "node:synth":{width:300,height:300,collapsed_height:78},
      "node:output":{width:180,height:250,collapsed_height:78},
    }),
  };
  document.nodes=[
    {id:"node:text",kind:"text_source",component_id:null,config:{text:"An intentionally long source label exercises maximum faceplate width."},disabled:false,bypassed:false},
    {id:"node:synth",kind:"synthesis_fixture",component_id:null,config:{voice:"alto",rate:1.5,pitch:0},disabled:false,bypassed:false},
    {id:"node:output",kind:"audio_output",component_id:null,config:{target:"browser",browser_device_id:"default",system_device_id:"default",wav_path:"data/speech-output.wav"},disabled:false,bypassed:false},
  ];
  document.edges=[
    {id:"edge:text-synth",from:{node_id:"node:text",port_id:"out"},to:{node_id:"node:synth",port_id:"text"},capacity:8},
    {id:"edge:synth-output",from:{node_id:"node:synth",port_id:"audio"},to:{node_id:"node:output",port_id:"in"},capacity:8},
  ];
  document.selected_sinks=[];
  return document;
}

function cytoscapeStub(){
  const noop=()=>{};
  return ()=>{
    const elements=new Map();
    const element=id=>{
      const data=elements.get(id);
      if(!data)return{length:0,select:noop,addClass:noop,removeClass:noop};
      return{
        length:1,select:noop,addClass:noop,removeClass:noop,
        data:noop,
        position:()=>data.position??{x:0,y:0},
        renderedPosition:()=>data.position??{x:0,y:0},
      };
    };
    return{
      on:noop,off:noop,fit:noop,panBy:noop,pan:()=>({x:0,y:0}),zoom:()=>1,extent:()=>({x1:0,y1:0,x2:840,y2:560}),
      collection:items=>items,
      add:items=>items.forEach(item=>elements.set(item.data.id,item)),
      elements:()=>({remove:()=>elements.clear(),unselect:noop}),
      nodes:()=>({removeClass:noop}),
      getElementById:element,
    };
  };
}

async function addFromPalette(page,text,{keyboard=false}={}){
  const previousCount=await page.locator(".patch-node-card").count();
  const item=page.locator(".palette-node").filter({hasText:text}).first();
  if(keyboard){await item.focus();await item.press("Enter");}
  else await item.click();
  await expect(page.locator(".patch-node-card")).toHaveCount(previousCount+1);
  const card=page.locator(".patch-node-card").last();
  await expect(card).toBeVisible();
  return card.getAttribute("data-node-id");
}

const jack=(page,nodeId,direction)=>page.locator(`[data-patch-jack][data-node-id="${nodeId}"][data-direction="${direction}"]`);

async function persistGraph(page){
  savedGraph=null;
  await page.getByRole("button",{name:"Save"}).click();
  await expect.poll(()=>savedGraph).not.toBeNull();
  return structuredClone(savedGraph);
}

async function renderedBoundaryMetrics(page){
  return page.evaluate(async()=>{
    const {graphStudioTestHooks:hooks}=await import("/speech-dataflow.js");
    const canvas=document.querySelector("#canvas");
    const canvasBounds=canvas.getBoundingClientRect();
    const relative=element=>{
      const bounds=element.getBoundingClientRect();
      return{
        x:bounds.left-canvasBounds.left,y:bounds.top-canvasBounds.top,
        width:bounds.width,height:bounds.height,
        right:bounds.right-canvasBounds.left,bottom:bounds.bottom-canvasBounds.top,
      };
    };
    const center=element=>{
      const bounds=relative(element);
      return{x:bounds.x+bounds.width/2,y:bounds.y+bounds.height/2};
    };
    const nodes=[...document.querySelectorAll(".patch-node-card")].map(card=>{
      const id=card.dataset.nodeId;
      const offCanvasTabStops=[...card.querySelectorAll("button,input,select,textarea")].filter(control=>{
        const bounds=relative(control);
        return control.tabIndex>=0&&(
          bounds.x<0||bounds.right>canvasBounds.width||bounds.y<0||bounds.bottom>canvasBounds.height
        );
      }).length;
      return{
        id,faceplate:relative(card),hitbox:hooks.renderedNodeBounds(id),
        inert:card.inert,collapsed:card.dataset.state==="collapsed",offCanvasTabStops,
      };
    });
    const jacks=[...document.querySelectorAll("[data-patch-jack]")].map(jack=>({
      nodeId:jack.dataset.nodeId,portId:jack.dataset.portId,direction:jack.dataset.direction,
      center:center(jack),tabIndex:jack.tabIndex,focused:jack===document.activeElement,
      outlineStyle:getComputedStyle(jack).outlineStyle,outlineWidth:getComputedStyle(jack).outlineWidth,
    }));
    const cables=[...document.querySelectorAll(".patch-cable")].map(path=>{
      const length=path.getTotalLength();
      const start=path.getPointAtLength(0),end=path.getPointAtLength(length);
      return{id:path.dataset.edgeId,start:{x:start.x,y:start.y},end:{x:end.x,y:end.y},bounds:relative(path)};
    });
    const host=document.querySelector(".patch-overlay-host");
    const layers=[...document.querySelectorAll(".patch-organization,.patch-cables,.patch-jacks,.patch-node-cards")].map(layer=>({
      className:layer.getAttribute("class"),bounds:relative(layer),overflow:getComputedStyle(layer).overflow,
      parentClass:layer.parentElement?.className??null,
    }));
    const probe=selector=>{
      const element=document.querySelector(selector),bounds=element.getBoundingClientRect();
      const x=bounds.left+bounds.width/2,y=bounds.top+Math.min(bounds.height/2,24);
      return{
        selector,
        top:document.elementFromPoint(x,y)?.closest?.(".patch-cables,.patch-jacks,.patch-node-cards,.patch-organization")?.className??null,
        stack:[...document.elementsFromPoint(x,y)].map(item=>item.id||item.className||item.tagName).slice(0,8),
      };
    };
    return{
      viewport:hooks.viewportBounds(),host:{
        bounds:relative(host),overflow:getComputedStyle(host).overflow,
        childClasses:[...host.children].map(child=>child.className?.baseVal??child.className),
      },nodes,jacks,cables,layers,
      probes:[probe("#inspector-panel"),probe(".toolbar")],
    };
  });
}

function renderedBoundaryViolations(metrics){
  const tolerance=3,violations=[];
  const enclosed=(inner,outer)=>(
    inner.x>=outer.x-tolerance&&inner.y>=outer.y-tolerance
    &&inner.right<=outer.right+tolerance&&inner.bottom<=outer.bottom+tolerance
  );
  for(const node of metrics.nodes){
    if(!enclosed(node.faceplate,node.hitbox))violations.push(`module ${node.id}: faceplate=${JSON.stringify(node.faceplate)} hitbox=${JSON.stringify(node.hitbox)} viewport=${JSON.stringify(metrics.viewport)}`);
    const nodeJacks=metrics.jacks.filter(jack=>jack.nodeId===node.id);
    for(const jack of nodeJacks){
      const edge=jack.direction==="input"?node.faceplate.x:node.faceplate.right;
      if(Math.abs(jack.center.x-edge)>tolerance)violations.push(`module ${node.id} port ${jack.portId}: jack=${JSON.stringify(jack.center)} faceplateEdge=${edge} viewport=${JSON.stringify(metrics.viewport)}`);
    }
  }
  const endpoints={
    "edge:text-synth":{
      start:metrics.jacks.find(jack=>jack.nodeId==="node:text"&&jack.portId==="out"),
      end:metrics.jacks.find(jack=>jack.nodeId==="node:synth"&&jack.portId==="text"),
    },
    "edge:synth-output":{
      start:metrics.jacks.find(jack=>jack.nodeId==="node:synth"&&jack.portId==="audio"),
      end:metrics.jacks.find(jack=>jack.nodeId==="node:output"&&jack.portId==="in"),
    },
  };
  for(const cable of metrics.cables){
    for(const endpoint of ["start","end"]){
      const jack=endpoints[cable.id]?.[endpoint];
      if(!jack||Math.hypot(cable[endpoint].x-jack.center.x,cable[endpoint].y-jack.center.y)>tolerance){
        violations.push(`edge ${cable.id} ${endpoint}: cable=${JSON.stringify(cable[endpoint])} jack=${JSON.stringify(jack?.center)} viewport=${JSON.stringify(metrics.viewport)}`);
      }
    }
  }
  for(const layer of metrics.layers){
    if(!enclosed(layer.bounds,metrics.viewport)||layer.overflow!=="hidden"||layer.parentClass!=="patch-overlay-host")violations.push(`overlay ${layer.className}: bounds=${JSON.stringify(layer.bounds)} overflow=${layer.overflow} parent=${layer.parentClass} viewport=${JSON.stringify(metrics.viewport)}`);
  }
  if(!enclosed(metrics.host.bounds,metrics.viewport)||!["clip","hidden"].includes(metrics.host.overflow))violations.push(`overlay host: bounds=${JSON.stringify(metrics.host.bounds)} overflow=${metrics.host.overflow} viewport=${JSON.stringify(metrics.viewport)}`);
  for(const probe of metrics.probes)if(probe.top)violations.push(`hit-test ${probe.selector}: overlay=${probe.top} stack=${JSON.stringify(probe.stack)} viewport=${JSON.stringify(metrics.viewport)}`);
  return violations;
}

test.beforeAll(async()=>{
  server=http.createServer(async(request,response)=>{
    const pathname=new URL(request.url,"http://fixture").pathname;
    if(request.method==="POST"&&pathname==="/api/pipeline/run"){
      const document=await readRequestJson(request);
      const runId=`fixture-run-${++runSequence}`;
      const nodeId=document.nodes[0]?.id??"graph";
      const wavNode=document.nodes.find(node=>node.kind==="audio_output"&&node.config?.target==="wav");
      const artifact=wavNode?{
        node_id:wavNode.id,path:wavNode.config.wav_path,
        download_url:`/api/files/download/${wavNode.config.wav_path.split("/").map(encodeURIComponent).join("/")}`,
      }:null;
      response.writeHead(200,{"Content-Type":"application/x-ndjson","Cache-Control":"no-store"});
      const run={response,status:"running",nodeId,artifact,timers:[],closed:false};
      activeRuns.set(runId,run);
      writeRunEvent(response,{run_id:runId,status:"running",node_id:nodeId,kind:"started",elapsed_ms:0});
      if(holdRuns)return;
      const outputCount=meterStorm?600:Math.max(1,document.nodes.length);
      for(let index=0;index<outputCount;index++){
        const delay=meterStorm?10:35+index*8;
        run.timers.push(setTimeout(()=>writeRunEvent(response,{
          run_id:runId,status:"running",node_id:document.nodes[index%Math.max(1,document.nodes.length)]?.id??nodeId,
          kind:"output",elapsed_ms:index+1,output:{port_id:index%2?"committed":"out",value:index%2?`phrase ${index}`:[0.82]},
        }),delay));
      }
      run.timers.push(setTimeout(()=>finishRun(runId,"completed","completed"),meterStorm?45:70+outputCount*8));
      return;
    }
    const command=pathname.match(/^\/api\/pipeline\/runs\/([^/]+)\/(stop|panic)$/);
    if(request.method==="POST"&&command){
      const runId=decodeURIComponent(command[1]),action=command[2];
      response.writeHead(200,{"Content-Type":"application/json"});
      response.end(JSON.stringify({run_id:runId,status:"stopping",action}));
      finishRun(runId,"cancelled","cancelled",action==="panic"?"Panic requested by operator.":"Stop requested by operator.");
      return;
    }
    const runLookup=pathname.match(/^\/api\/pipeline\/runs\/([^/]+)$/);
    if(request.method==="GET"&&runLookup){
      const runId=decodeURIComponent(runLookup[1]),run=activeRuns.get(runId);
      response.writeHead(run?200:404,{"Content-Type":"application/json"});
      response.end(JSON.stringify(run?{run_id:runId,status:run.status,started_at_ms:Date.now()-25,artifacts:run.artifact?[run.artifact]:[]}:{error:"run not found"}));
      return;
    }
    const relative=pathname==="/"?"speech-dataflow.html":pathname.replace(/^\/+/,"");
    const target=path.resolve(publicRoot,relative);
    if(!target.startsWith(`${publicRoot}${path.sep}`)||!fs.existsSync(target)){response.writeHead(404);response.end("not found");return;}
    const type=target.endsWith(".html")?"text/html":target.endsWith(".mjs")||target.endsWith(".js")?"text/javascript":"text/plain";
    response.writeHead(200,{"Content-Type":type});fs.createReadStream(target).pipe(response);
  });
  await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
  baseUrl=`http://127.0.0.1:${server.address().port}`;
});
test.afterAll(async()=>new Promise(resolve=>server.close(resolve)));

test.beforeEach(async({page},testInfo)=>{
  savedGraph=null;
  holdRuns=false;
  meterStorm=false;
  const realLayout=testInfo.title.includes("rendered boundary");
  if(!realLayout)await page.addInitScript(stub=>{globalThis.cytoscape=eval(`(${stub})`)();},cytoscapeStub.toString());
  await page.route("https://cdn.jsdelivr.net/**",route=>realLayout
    ?route.fulfill({contentType:"text/javascript",body:fs.readFileSync(path.resolve(publicRoot,"../../../node_modules/cytoscape/dist/cytoscape.min.js"),"utf8")})
    :route.abort());
  await page.route("**/api/pipeline/**",async route=>{
    const request=route.request(),url=new URL(request.url()),pathname=url.pathname;
    if(pathname==="/api/pipeline/run"||pathname.startsWith("/api/pipeline/runs/"))return route.continue();
    if(pathname==="/api/pipeline/catalog")return route.fulfill({json:discovery});
    if(pathname==="/api/pipeline/starters")return route.fulfill({json:{graphs:[graph]}});
    if(pathname==="/api/pipeline/validate"){
      const document=request.postDataJSON();
      const diagnostics=document.metadata?.name==="Incomplete replacement draft"
        ?[{code:"port.required_input_missing",severity:"error",message:"A required input is not connected.",target:{node_id:"node:mic-1",port_id:"out"}}]
        :document.metadata?.name==="Rendered boundary fixture"
        ?[{code:"fixture.output_error",severity:"error",message:"Fixture output is unavailable.",target:{node_id:"node:output",port_id:"in"}}]
        :[];
      return route.fulfill({json:{valid:diagnostics.length===0,diagnostics}});
    }
    if(request.method()==="PUT"&&pathname.startsWith("/api/pipeline/graphs/")){
      savedGraph=request.postDataJSON();return route.fulfill({json:{document:savedGraph}});
    }
    if(pathname==="/api/pipeline/graphs")return route.fulfill({json:{graphs:savedGraph?[{graph_id:savedGraph.graph_id,name:savedGraph.metadata.name,revision:savedGraph.revision}]:[]}});
    if(pathname.startsWith("/api/pipeline/graphs/")&&savedGraph)return route.fulfill({json:{document:savedGraph}});
    return route.fulfill({status:404,json:{error:`unhandled fixture route ${pathname}`}});
  });
  await page.goto(`${baseUrl}/speech-dataflow.html`);
  await expect(page.getByText("Ready to compile and execute")).toBeVisible();
});

test("actual replacement dialog supports large catalogs, keyboard review, cancel, atomic apply, undo, redo, and persistence",async({page})=>{
  const replace=page.getByRole("button",{name:"Replace"});
  await replace.focus();await replace.press("Enter");
  const dialog=page.getByRole("dialog",{name:"Replace node"});
  await expect(dialog).toBeVisible();
  await expect(page.locator("#replacement-search")).toBeFocused();
  await expect(page.locator("#replacement-results")).toContainText("1202 candidates");
  await expect(page.locator(".replacement-candidate")).toHaveCount(100);

  await page.locator("#replacement-search").fill("Missing Model");
  const unavailable=page.getByRole("option",{name:/Missing Model/});
  await unavailable.click();
  await expect(unavailable).toHaveAttribute("data-applicable","false");
  await expect(page.locator("#replacement-continue")).toBeDisabled();
  await expect(unavailable).toContainText("Model files are absent");

  await page.locator("#replacement-cancel").click();
  await expect(dialog).toBeHidden();
  await expect(replace).toBeFocused();
  await page.getByRole("button",{name:"Save"}).click();
  await expect.poll(()=>savedGraph?.nodes[0]?.component_id).toBe("base");

  await replace.click();await page.locator("#replacement-search").fill("Ready Model");
  const ready=page.getByRole("option",{name:/Ready Model/});await ready.focus();await ready.press("Enter");
  await page.locator("#replacement-continue").click();
  await expect(page.getByRole("heading",{name:"Review replacement impact"})).toBeVisible();
  await expect(page.locator("#replacement-wiring-impact")).toContainText("Selected sink: preserved");
  await expect(page.locator("#replacement-config-impact")).toContainText("language: preserved");
  await expect(page.locator("#replacement-apply")).toBeEnabled();
  await page.locator("#replacement-apply").click();
  await expect(page.locator("#node-title")).toContainText("Ready Model");

  await page.getByRole("button",{name:"Undo"}).click();
  await expect(page.locator("#node-title")).toContainText("Base");
  await page.getByRole("button",{name:"Redo"}).click();
  await expect(page.locator("#node-title")).toContainText("Ready Model");
  await page.getByRole("button",{name:"Save"}).click();
  await expect.poll(()=>savedGraph?.nodes[0]?.component_id).toBe("alternate");
  expect(savedGraph.selected_sinks).toEqual([{node_id:"node:asr",port_id:"committed"}]);
});

test("replacement remains applyable when an incomplete draft has only pre-existing diagnostics",async({page})=>{
  await page.locator("#pipeline-name").fill("Incomplete replacement draft");
  await page.locator("#pipeline-name").dispatchEvent("change");
  await expect(page.locator("#validation")).toContainText("graph diagnostic");

  await page.getByRole("button",{name:"Replace",exact:true}).click();
  await page.locator("#replacement-search").fill("Ready Model");
  await page.getByRole("option",{name:/Ready Model/}).click();
  await page.locator("#replacement-continue").click();

  await expect(page.locator("#replacement-diagnostics")).toContainText("existing diagnostic");
  await expect(page.locator("#replacement-apply")).toBeEnabled();
  await page.locator("#replacement-apply").click();
  await expect(page.locator("#node-title")).toContainText("Ready Model");
});

test("dialog is keyboard-contained and becomes a full-width touch sheet on a narrow screen",async({page})=>{
  await page.setViewportSize({width:390,height:720});
  await page.getByRole("button",{name:"Inspector"}).click();
  await page.getByRole("button",{name:"Replace"}).click();
  const dialog=page.locator("#replacement-dialog"),box=await dialog.boundingBox();
  expect(box.width).toBeGreaterThanOrEqual(389);
  await page.locator("#replacement-search").fill("Ready Model");
  await page.getByRole("option",{name:/Ready Model/}).tap();
  await expect(page.locator("#replacement-continue")).toBeEnabled();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(page.getByRole("button",{name:"Replace"})).toBeFocused();
});

test("stored faceplate geometry controls card size and jack anchor position",async({page})=>{
  const cardLocator=page.locator('.patch-node-card[data-node-id="node:asr"]');
  await expect(cardLocator).toBeVisible();
  const metrics=await cardLocator.evaluate(element=>{
    const asrInput=document.querySelector('[data-patch-jack][data-node-id="node:asr"][data-direction="input"]');
    const asrOutput=document.querySelector('[data-patch-jack][data-node-id="node:asr"][data-direction="output"]');
    const cardLeft=parseFloat(getComputedStyle(element).left);
    const width=parseFloat(getComputedStyle(element).getPropertyValue("--patch-node-card-width"));
    const inputLeft=parseFloat(getComputedStyle(asrInput.closest(".patch-jack-wrap")).left);
    const outputLeft=parseFloat(getComputedStyle(asrOutput.closest(".patch-jack-wrap")).left);
    return{
      cardLeft,inputLeft,outputLeft,width,
      inputOffset:Math.round(inputLeft-cardLeft),
      outputOffset:Math.round(outputLeft-cardLeft),
    };
  });
  expect(metrics.width).toBe(360);
  expect(metrics.inputOffset).toBe(-180);
  expect(metrics.outputOffset).toBe(180);
});

test("rendered boundary contract contains faceplates, cables, hit targets, and focus through real layout transitions",async({page})=>{
  test.setTimeout(60_000);
  await page.setViewportSize({width:1440,height:900});
  savedGraph=renderedBoundaryGraph();
  await page.getByRole("button",{name:"Open"}).click();
  await page.locator("#saved-graphs button").click();
  await expect(page.locator(".patch-node-card")).toHaveCount(3);
  await expect(page.locator("#validation")).toContainText("Fixture output is unavailable");
  await page.locator('.patch-cable-hit[data-edge-id="edge:synth-output"]').click();
  await expect(page.locator('.patch-cable[data-edge-id="edge:synth-output"]')).toHaveClass(/selected/);

  const assertGeometry=async context=>{
    await expect.poll(async()=>renderedBoundaryViolations(await renderedBoundaryMetrics(page)),{
      message:`rendered boundary regression after ${context}`,
      timeout:5_000,
    }).toEqual([]);
  };
  await assertGeometry("desktop initial render");
  await expect(page.locator(".canvas-wrap")).toHaveScreenshot("graph-studio-rendered-boundary-desktop.png",{
    animations:"disabled",caret:"hide",maxDiffPixelRatio:.01,
  });

  await page.getByRole("button",{name:"Expand text_source faceplate"}).click();
  await page.getByRole("button",{name:"Collapse synthesis_fixture faceplate"}).click();
  await assertGeometry("collapse and expand");
  await page.getByRole("button",{name:"Expand synthesis_fixture faceplate"}).click();

  await page.evaluate(async()=>{
    const {graphStudioTestHooks:hooks}=await import("/speech-dataflow.js");
    hooks.zoom(.76);
    hooks.panBy({x:-48,y:34});
  });
  await assertGeometry("pan and browser zoom");
  await page.evaluate(async()=>{
    const {graphStudioTestHooks:hooks}=await import("/speech-dataflow.js");
    hooks.zoom(1);
    hooks.panBy({x:48,y:-34});
  });

  await page.setViewportSize({width:390,height:720});
  await assertGeometry("narrow resize");
  let narrowMetrics=await renderedBoundaryMetrics(page);
  expect(narrowMetrics.jacks.filter(jack=>(
    jack.center.x<22||jack.center.x>narrowMetrics.viewport.right-22
    ||jack.center.y<22||jack.center.y>narrowMetrics.viewport.bottom-22
  )).every(jack=>jack.tabIndex===-1),"off-canvas jacks must leave the keyboard order").toBe(true);
  expect(narrowMetrics.nodes.every(node=>node.offCanvasTabStops===0),"clipped faceplates must not expose off-canvas controls to keyboard focus").toBe(true);

  await page.getByRole("button",{name:"Inspector"}).click();
  await expect(page.locator("#inspector-panel")).toHaveClass(/open/);
  await page.getByRole("button",{name:"Inspector"}).click();
  await expect(page.locator("#inspector-panel")).not.toHaveClass(/open/);
  await assertGeometry("mobile inspector drawer transition");
  await expect(page.locator(".canvas-wrap")).toHaveScreenshot("graph-studio-rendered-boundary-narrow.png",{
    animations:"disabled",caret:"hide",maxDiffPixelRatio:.01,
  });

  await page.setViewportSize({width:1440,height:900});
  await page.evaluate(async()=>{
    const {graphStudioTestHooks:hooks}=await import("/speech-dataflow.js");
    hooks.teardownAndReinitialize();
  });
  await expect(page.locator(".patch-overlay-host")).toHaveCount(1);
  await expect(page.locator(".patch-cables")).toHaveCount(1);
  await expect(page.locator(".patch-node-cards")).toHaveCount(1);
  await assertGeometry("teardown and reinitialize");

  const focusableJack=page.locator('[data-patch-jack][data-node-id="node:synth"][data-port-id="text"]');
  await focusableJack.focus();
  await expect(focusableJack).toBeFocused();
  const focused=await renderedBoundaryMetrics(page);
  const focusedJack=focused.jacks.find(jack=>jack.focused);
  expect(focusedJack?.outlineStyle).not.toBe("none");
  expect(Number.parseFloat(focusedJack?.outlineWidth??"0")).toBeGreaterThan(0);

  await page.locator("#pipeline-name").fill("Rendered boundary live status");
  await page.locator("#pipeline-name").dispatchEvent("change");
  await expect(page.locator("#validation")).toContainText("Ready");
  await page.getByRole("button",{name:"Run"}).click();
  await expect(page.locator("#run-state")).toHaveText("Completed");
  await assertGeometry("live status updates");

  const canvasBox=await page.locator("#canvas").boundingBox();
  await page.mouse.dblclick(canvasBox.x+canvasBox.width/2,canvasBox.y+canvasBox.height-30);
  await expect(page.getByRole("dialog",{name:"Add module"})).toBeVisible();
  await page.locator("#quick-add-search").fill("Audio pass-through");
  await page.getByRole("option",{name:/Audio pass-through/}).click();
  const inserted=page.locator(".patch-node-card").last();
  const insertedId=await inserted.getAttribute("data-node-id");
  await page.locator('.patch-cable-hit[data-edge-id="edge:synth-output"]').click();
  await jack(page,"node:output","input").dragTo(jack(page,insertedId,"input"));
  const stored=await persistGraph(page);
  expect(stored.edges.find(edge=>edge.id==="edge:synth-output")?.to).toEqual({node_id:insertedId,port_id:"in"});
});

test("frames and reviewed subpatches persist and drill with browser history",async({page})=>{
  await page.getByRole("button",{name:"Frame"}).click();
  const dialog=page.getByRole("dialog",{name:"Create frame"});
  await expect(dialog).toBeVisible();
  await page.locator("#organization-text").fill("Recognition section");
  await page.locator("#organization-apply").click();
  await expect(page.locator("#organization-list")).toContainText("Frame: Recognition section");

  await page.getByRole("button",{name:"Subpatch"}).click();
  await expect(page.getByRole("dialog",{name:"Create subpatch"})).toBeVisible();
  await expect(page.locator("#organization-port-review input")).toHaveCount(1);
  await page.locator("#organization-text").fill("Recognizer");
  await page.locator("#organization-apply").click();
  const item=page.locator(".organization-item").filter({hasText:"Recognizer"});
  await expect(item).toContainText("1 reviewed ports");
  await item.getByRole("button",{name:"Collapse"}).click();
  await expect(page.locator(".patch-subpatch-summary")).toContainText("1 nodes · 1 ports");
  await page.getByRole("button",{name:"Save"}).click();
  await expect.poll(()=>savedGraph?.presentation?.collapsed_subpatches?.length).toBe(1);
  await item.getByRole("button",{name:"Expand"}).click();
  await item.getByRole("button",{name:"Open"}).click();
  await expect(page).toHaveURL(/subpatch=/);
  await expect(page.locator("#subpatch-breadcrumbs")).toContainText("Recognizer");
  await page.goBack();
  await expect(page).not.toHaveURL(/subpatch=/);

  await page.getByRole("button",{name:"Save"}).click();
  await expect.poll(()=>savedGraph?.schema_version).toBe(3);
  expect(savedGraph.presentation.frames).toHaveLength(1);
  expect(savedGraph.subpatches).toHaveLength(1);
});

test("visible jacks patch, reject, reconnect, fan out, cancel, delete, and persist without implicit graph edits",async({page})=>{
  const jack=(nodeId,direction)=>page.locator(`[data-patch-jack][data-node-id="${nodeId}"][data-direction="${direction}"]`);
  const save=async()=>{
    savedGraph=null;
    await page.locator("#status").evaluate(element=>{element.textContent="";});
    await page.getByRole("button",{name:"Save"}).click();
    await expect.poll(()=>savedGraph).not.toBeNull();
    await expect(page.locator("#status")).toContainText("Saved Browser fixture");
  };

  await expect(jack("node:mic-1","output")).toBeVisible();
  await expect(jack("node:asr","input")).toBeVisible();
  await jack("node:mic-1","output").dragTo(jack("node:asr","input"));
  await expect(page.locator(".patch-cable")).toHaveCount(1);
  await save();
  expect(savedGraph.edges).toHaveLength(1);
  const audioEdge=structuredClone(savedGraph.edges[0]);

  await jack("node:mic-2","output").dragTo(jack("node:asr","input"));
  await expect(page.locator(".patch-cable")).toHaveCount(1);
  await expect(page.locator("#status")).toContainText("already occupied");
  await page.keyboard.press("Escape");
  await save();
  expect(savedGraph.edges).toEqual([audioEdge]);

  await jack("node:asr","output").focus();
  await jack("node:asr","output").press("Enter");
  await jack("node:sink-1","input").focus();
  await jack("node:sink-1","input").press("Enter");
  await expect(page.locator(".patch-cable")).toHaveCount(2);
  await save();
  await page.locator("#graph-outline button").first().dispatchEvent("click");
  await expect(page.locator(".patch-cable.selected")).toHaveCount(0);
  await jack("node:asr","output").focus();
  await jack("node:asr","output").press("Enter");
  await jack("node:sink-2","input").focus();
  await jack("node:sink-2","input").press("Enter");
  await expect(page.locator(".patch-cable")).toHaveCount(3);
  await expect(page.locator(".patch-connection-list").getByRole("button",{name:/connected to Transcript in/})).toHaveCount(2);

  await save();
  const beforeCancel=JSON.stringify(savedGraph);
  await jack("node:asr","output").focus();
  await jack("node:asr","output").press("Enter");
  await page.keyboard.press("Escape");
  await expect(page.locator(".patch-cable-preview")).toHaveCount(0);
  await save();
  expect(JSON.stringify(savedGraph)).toBe(beforeCancel);

  await page.locator(`.patch-cable-hit[data-edge-id="${audioEdge.id}"]`).dispatchEvent("pointerdown");
  await jack("node:mic-1","output").dragTo(jack("node:mic-2","output"));
  await save();
  const reconnected=savedGraph.edges.find(edge=>edge.id===audioEdge.id);
  expect(reconnected.capacity).toBe(audioEdge.capacity);
  expect(reconnected.from).toEqual({node_id:"node:mic-2",port_id:"out"});

  await page.getByRole("button",{name:"Undo"}).click();
  await save();
  expect(savedGraph.edges.find(edge=>edge.id===audioEdge.id).from).toEqual({node_id:"node:mic-1",port_id:"out"});
  await page.getByRole("button",{name:"Redo"}).click();
  await save();
  expect(savedGraph.edges.find(edge=>edge.id===audioEdge.id).from).toEqual({node_id:"node:mic-2",port_id:"out"});

  const audioConnection=page.locator(`.patch-connection-list button[data-edge-id="${audioEdge.id}"]`);
  await audioConnection.focus();
  await expect(audioConnection).toBeFocused();
  await page.keyboard.press("Delete");
  await expect(page.locator(".patch-cable")).toHaveCount(2);
  await page.getByRole("button",{name:"Undo"}).click();
  await expect(page.locator(".patch-cable")).toHaveCount(3);
  await page.getByRole("button",{name:"Redo"}).click();
  await expect(page.locator(".patch-cable")).toHaveCount(2);
  await save();
  expect(savedGraph.edges.some(edge=>edge.id===audioEdge.id)).toBe(false);
  expect(savedGraph.edges).toHaveLength(2);
});

test("multi-selection copy, paste, arrange, delete, undo, redo, and branch clearing preserve internal topology",async({page})=>{
  const jack=(nodeId,direction)=>page.locator(`[data-patch-jack][data-node-id="${nodeId}"][data-direction="${direction}"]`);
  const save=async()=>{
    savedGraph=null;
    await page.locator("#status").evaluate(element=>{element.textContent="";});
    await page.getByRole("button",{name:"Save"}).click();
    await expect.poll(()=>savedGraph).not.toBeNull();
    await expect(page.locator("#status")).toContainText("Saved Browser fixture");
  };
  const originalIds=new Set(graph.nodes.map(node=>node.id));

  await jack("node:mic-1","output").focus();await jack("node:mic-1","output").press("Enter");
  await jack("node:asr","input").focus();await jack("node:asr","input").press("Enter");
  const outline=page.locator("#graph-outline button");
  await outline.nth(0).dispatchEvent("click");
  await outline.nth(1).dispatchEvent("click",{shiftKey:true});
  await expect(outline.nth(0)).toHaveAttribute("aria-pressed","true");
  await expect(outline.nth(1)).toHaveAttribute("aria-pressed","true");

  await page.getByRole("button",{name:"Copy",exact:true}).click();
  await page.getByRole("button",{name:"Paste",exact:true}).click();
  await expect(page.locator(".patch-cable")).toHaveCount(2);
  await save();
  const pastedIds=savedGraph.nodes.map(node=>node.id).filter(id=>!originalIds.has(id));
  expect(pastedIds).toHaveLength(2);expect(new Set(savedGraph.nodes.map(node=>node.id)).size).toBe(7);
  const pastedEdge=savedGraph.edges.find(edge=>pastedIds.includes(edge.from.node_id)||pastedIds.includes(edge.to.node_id));
  expect(pastedIds).toContain(pastedEdge.from.node_id);expect(pastedIds).toContain(pastedEdge.to.node_id);

  await page.getByRole("button",{name:"Undo"}).click();await save();
  expect(savedGraph.nodes).toHaveLength(5);expect(savedGraph.edges).toHaveLength(1);
  await page.getByRole("button",{name:"Redo"}).click();await save();
  expect(savedGraph.nodes).toHaveLength(7);expect(savedGraph.edges).toHaveLength(2);

  await page.getByRole("button",{name:"Snap off"}).click();
  await expect(page.getByRole("button",{name:"Snap on"})).toHaveAttribute("aria-pressed","true");
  await outline.last().focus();await outline.last().press("ArrowRight");await save();
  const snapped=savedGraph.presentation.node_positions;
  expect(pastedIds.every(id=>snapped[id].x%24===0&&snapped[id].y%24===0)).toBe(true);
  await page.getByRole("button",{name:"Undo"}).click();
  await page.getByRole("button",{name:"Fit selection"}).click();

  await page.getByRole("button",{name:"Align top"}).click();await save();
  const layout=savedGraph.presentation.node_positions;
  expect(new Set(pastedIds.map(id=>layout[id].y)).size).toBe(1);
  await page.getByRole("button",{name:"Undo"}).click();
  await page.getByRole("button",{name:"Tidy"}).click();
  await expect(page.getByRole("button",{name:"Redo"})).toBeDisabled();

  await page.locator("#duplicate-selection").click();
  await save();expect(savedGraph.nodes).toHaveLength(9);expect(savedGraph.edges).toHaveLength(3);
  await page.getByRole("button",{name:"Delete selection"}).click();
  await save();expect(savedGraph.nodes).toHaveLength(7);expect(savedGraph.edges).toHaveLength(2);
  await page.getByRole("button",{name:"Undo"}).click();
  await save();expect(savedGraph.nodes).toHaveLength(9);expect(savedGraph.edges).toHaveLength(3);
});

test("quick-add opens at intent, filters cable consumers, and inserts on a cable atomically",async({page})=>{
  const jack=(nodeId,direction)=>page.locator(`[data-patch-jack][data-node-id="${nodeId}"][data-direction="${direction}"]`);
  const save=async()=>{
    savedGraph=null;
    await page.locator("#status").evaluate(element=>{element.textContent="";});
    await page.getByRole("button",{name:"Save"}).click();
    await expect.poll(()=>savedGraph).not.toBeNull();
    await expect(page.locator("#status")).toContainText("Saved Browser fixture");
  };
  const dialog=page.getByRole("dialog",{name:"Add module"});

  await page.locator("#canvas").dispatchEvent("dblclick",{clientX:520,clientY:360});
  await expect(dialog).toBeVisible();await expect(page.locator("#quick-add-search")).toBeFocused();
  await page.locator("#quick-add-search").fill("Audio pass-through");
  await page.getByRole("option",{name:/Audio pass-through/}).click();
  await expect(dialog).toBeHidden();await save();expect(savedGraph.nodes).toHaveLength(6);
  await page.getByRole("button",{name:"Undo"}).click();await save();expect(savedGraph.nodes).toHaveLength(5);

  await jack("node:mic-1","output").dragTo(page.locator("#canvas"),{targetPosition:{x:360,y:300}});
  await expect(dialog).toBeVisible();
  await expect(page.getByRole("option",{name:/Audio pass-through/})).toBeVisible();
  await page.locator("#quick-add-search").fill("Audio pass-through");
  await page.getByRole("option",{name:/Audio pass-through/}).click();
  await save();expect(savedGraph.nodes).toHaveLength(6);expect(savedGraph.edges).toHaveLength(1);
  const originalEdge=structuredClone(savedGraph.edges[0]);

  await page.locator(`.patch-cable-hit[data-edge-id="${originalEdge.id}"]`).dispatchEvent("pointerdown");
  await page.keyboard.press("i");await expect(dialog).toBeVisible();
  await page.locator("#quick-add-search").fill("Audio pass-through");
  await page.getByRole("option",{name:/Audio pass-through/}).click();
  await save();expect(savedGraph.nodes).toHaveLength(7);expect(savedGraph.edges).toHaveLength(2);
  const upstream=savedGraph.edges.find(edge=>edge.id===originalEdge.id);
  expect(upstream.capacity).toBe(originalEdge.capacity);expect(upstream.from).toEqual(originalEdge.from);
  expect(upstream.to).not.toEqual(originalEdge.to);

  await page.getByRole("button",{name:"Undo"}).click();await save();
  expect(savedGraph.nodes).toHaveLength(6);expect(savedGraph.edges).toEqual([originalEdge]);
  await page.getByRole("button",{name:"Redo"}).click();await save();
  expect(savedGraph.nodes).toHaveLength(7);expect(savedGraph.edges).toHaveLength(2);

  await page.locator("#palette-search").fill("Audio pass-through");
  await page.locator(`.patch-cable-hit[data-edge-id="${originalEdge.id}"]`).evaluate(target=>{
    const transfer=new DataTransfer();
    transfer.setData("application/x-tongues-catalog-id","kind:audio_passthrough");
    target.dispatchEvent(new DragEvent("drop",{bubbles:true,cancelable:true,dataTransfer:transfer}));
  });
  await save();expect(savedGraph.nodes).toHaveLength(8);expect(savedGraph.edges).toHaveLength(3);
  expect(savedGraph.edges.some(edge=>edge.id===originalEdge.id)).toBe(true);

  await jack("node:asr","input").evaluate((target)=>{
    const transfer=new DataTransfer();
    transfer.setData("application/x-tongues-catalog-id","kind:audio_passthrough");
    target.dispatchEvent(new DragEvent("drop",{bubbles:true,cancelable:true,dataTransfer:transfer}));
  });
  await save();expect(savedGraph.nodes).toHaveLength(9);expect(savedGraph.edges).toHaveLength(4);
  expect(savedGraph.edges.some(edge=>edge.to.node_id==="node:asr"&&edge.to.port_id==="audio")).toBe(true);
});

test("complete ASR and TTS patches can be built, run, and persisted with pointer and keyboard paths",async({page})=>{
  await page.getByRole("button",{name:"New"}).click();
  await expect(page.locator(".patch-node-card")).toHaveCount(0);
  const microphone=await addFromPalette(page,"Microphone");
  const recognizer=await addFromPalette(page,"Base");
  const transcript=await addFromPalette(page,"transcript_sink");
  for(let index=0;index<8;index++)await page.locator("#graph-outline button").nth(0).press("ArrowLeft");
  for(let index=0;index<8;index++)await page.locator("#graph-outline button").nth(2).press("ArrowRight");
  await jack(page,microphone,"output").dragTo(jack(page,recognizer,"input"));
  await jack(page,recognizer,"output").dragTo(jack(page,transcript,"input"));
  await page.getByRole("button",{name:"Run"}).click();
  await expect(page.locator("#run-state")).toHaveText("Completed");
  await expect(page.locator("#run-events")).toContainText("output");
  let stored=await persistGraph(page);
  expect(stored.nodes.map(node=>node.kind)).toEqual(["microphone","asr","transcript_sink"]);
  expect(stored.edges).toHaveLength(2);

  await page.getByRole("button",{name:"New"}).focus();
  await page.getByRole("button",{name:"New"}).press("Enter");
  await expect(page.locator(".patch-node-card")).toHaveCount(0);
  const source=await addFromPalette(page,"Text source",{keyboard:true});
  const synthesizer=await addFromPalette(page,"Voice",{keyboard:true});
  const speaker=await addFromPalette(page,"Audio output",{keyboard:true});
  await jack(page,source,"output").focus();await jack(page,source,"output").press("Enter");
  await jack(page,synthesizer,"input").focus();await jack(page,synthesizer,"input").press("Enter");
  await jack(page,synthesizer,"output").focus();await jack(page,synthesizer,"output").press("Enter");
  await jack(page,speaker,"input").focus();await jack(page,speaker,"input").press("Enter");

  const destination=page.locator(`[data-node-card="${speaker}"] select[data-config-field="target"]`);
  await destination.selectOption("wav");
  const wavPath=page.locator(`[data-node-card="${speaker}"] input[data-config-field="wav_path"]`);
  await expect(wavPath).toBeVisible();
  await wavPath.evaluate(input=>{input.value="data/rendered-voice.wav";input.onchange();});
  await expect(page.locator(`[data-node-card="${speaker}"] input[data-config-field="wav_path"]`)).toHaveValue("data/rendered-voice.wav");

  const voice=page.locator(`[data-node-card="${synthesizer}"] select[data-config-field="voice"]`);
  await voice.evaluate(select=>{select.value="tenor";select.onchange();});
  await expect(page.locator(`[data-node-card="${synthesizer}"] select[data-config-field="voice"]`)).toHaveValue("tenor");
  await page.getByRole("button",{name:"Undo"}).click();
  await expect(page.locator(`[data-node-card="${synthesizer}"] select[data-config-field="voice"]`)).toHaveValue("alto");
  await page.getByRole("button",{name:"Redo"}).click();
  await expect(page.locator(`[data-node-card="${synthesizer}"] select[data-config-field="voice"]`)).toHaveValue("tenor");

  await page.getByRole("button",{name:"Run"}).focus();
  await page.getByRole("button",{name:"Run"}).press("Enter");
  await expect(page.locator("#run-state")).toHaveText("Completed");
  const download=page.getByRole("link",{name:"Download generated WAV rendered-voice.wav"});
  await expect(download).toBeVisible();
  await expect(download).toHaveAttribute("href","/api/files/download/data/rendered-voice.wav");
  await expect(download).toHaveAttribute("download","rendered-voice.wav");
  stored=await persistGraph(page);
  expect(stored.nodes.map(node=>node.kind)).toEqual(["text_source","tts","audio_output"]);
  expect(stored.nodes.find(node=>node.id===synthesizer).config.voice).toBe("tenor");
  expect(stored.nodes.find(node=>node.id===speaker).config).toMatchObject({target:"wav",wav_path:"data/rendered-voice.wav"});
  expect(stored.edges).toHaveLength(2);
});

test("transport locks staged structural edits and Stop or Panic end activity with named failure state",async({page})=>{
  holdRuns=true;
  await jack(page,"node:mic-1","output").dragTo(jack(page,"node:asr","input"));
  await page.locator("#pipeline-name").fill("Held transport");
  const initialCards=await page.locator(".patch-node-card").count();

  await page.getByRole("button",{name:"Run"}).click();
  await expect(page.locator("#run-state")).toHaveText("Running");
  await expect(page.getByRole("button",{name:"Stop"})).toBeEnabled();
  await page.locator(".palette-node").filter({hasText:"Audio pass-through"}).first().click();
  await expect(page.locator("#status")).toContainText("Stop transport before editing");
  await expect(page.locator(".patch-node-card")).toHaveCount(initialCards);
  await jack(page,"node:mic-2","output").dragTo(jack(page,"node:asr","input"));
  await expect(page.locator("#status")).toContainText("Stop transport before patching");

  await page.getByRole("button",{name:"Stop"}).click();
  await expect(page.locator("#run-state")).toHaveText("Cancelled");
  await expect(page.getByRole("button",{name:"Run"})).toBeEnabled();

  await page.getByRole("button",{name:"Run"}).click();
  await expect(page.locator("#run-state")).toHaveText("Running");
  await page.getByRole("button",{name:"Panic"}).click();
  await expect(page.locator("#run-state")).toHaveText("Cancelled");
  const connection=page.locator(".patch-connection-list").getByRole("button");
  await expect(connection).toContainText("connection state failed");
  await expect(connection).toContainText("Panic requested by operator");
  await expect(page.locator(".patch-cable")).toHaveClass(/edge-state-failed/);
  expect(await page.locator(".patch-cable").evaluate(element=>getComputedStyle(element).strokeDasharray)).not.toBe("none");
});

test("presentation edits and semantic wiring survive save, share, new, and reopen",async({page})=>{
  await jack(page,"node:mic-1","output").dragTo(jack(page,"node:asr","input"));
  await page.locator(".patch-cable-hit").dispatchEvent("pointerdown");
  await page.getByRole("button",{name:"Add reroute at canvas center"}).click();
  await page.getByRole("button",{name:"Note"}).click();
  await page.locator("#organization-text").fill("Operator note");
  await page.locator("#organization-apply").click();
  await page.locator("#cable-opacity").fill("0.6");
  await page.locator("#cable-opacity").dispatchEvent("change");

  await page.getByRole("button",{name:"Share"}).click();
  const share=page.getByRole("dialog",{name:"Share graph"});
  await expect(share).toBeVisible();
  const shared=JSON.parse(await page.locator("#share-json").inputValue());
  expect(shared.edges).toHaveLength(1);
  expect(shared.presentation.notes[0].text).toBe("Operator note");
  expect(shared.presentation.cables[shared.edges[0].id].reroute_points).toHaveLength(1);
  expect(shared.presentation.global_cable_opacity).toBe(0.6);
  await expect(page.locator("#share-url")).toHaveValue(/\/studio\/graphs\/pipeline%3A/);
  await share.getByRole("button",{name:"Close"}).click();

  await page.getByRole("button",{name:"New"}).click();
  await expect(page.locator(".patch-cable")).toHaveCount(0);
  await page.getByRole("button",{name:"Open"}).click();
  await page.locator("#saved-graphs button").click();
  await expect(page.locator(".patch-cable")).toHaveCount(1);
  await expect(page.locator(".patch-note")).toHaveText("Operator note");
  const reopened=await persistGraph(page);
  expect(reopened.edges).toEqual(shared.edges);
  expect(reopened.presentation).toEqual(shared.presentation);
});

test("accessible names, documented shortcuts, non-color cues, and touch targets match the implementation",async({page})=>{
  await page.setViewportSize({width:390,height:720});
  await jack(page,"node:mic-1","output").dragTo(jack(page,"node:asr","input"));
  const sourceJack=jack(page,"node:mic-1","output");
  const targetJack=jack(page,"node:asr","input");
  for(const target of [sourceJack,targetJack]){
    const box=await target.boundingBox();
    expect(box.width).toBeGreaterThanOrEqual(44);
    expect(box.height).toBeGreaterThanOrEqual(44);
  }
  await expect(sourceJack).toHaveAttribute("aria-label",/Microphone.*output audio.*audio stream.*connection.*Base.*connection state ready/i);
  await expect(page.locator(".patch-connection-list").getByRole("button")).toContainText(/audio stream.*connected to.*connection state ready/i);

  const shortcutContract={
    undo:"Control+Z Meta+Z",
    redo:"Control+Shift+Z Meta+Shift+Z Control+Y",
    "copy-selection":"Control+C Meta+C",
    "cut-selection":"Control+X Meta+X",
    "paste-selection":"Control+V Meta+V",
    "duplicate-selection":"Control+D Meta+D",
    "delete-selection":"Delete Backspace",
  };
  for(const [id,value] of Object.entries(shortcutContract))await expect(page.locator(`#${id}`)).toHaveAttribute("aria-keyshortcuts",value);
  const help=fs.readFileSync(path.resolve(publicRoot,"../../../docs/speech-dataflow.md"),"utf8");
  for(const phrase of ["Control+Space","Control+C","Control+X","Control+V","Control+D","Control+Shift+Z","Control+Y","Delete or Backspace","press I","Run, Stop, or Panic"]){
    expect(help).toContain(phrase);
  }
  expect(help).toContain("44-pixel cable-jack targets");
  expect(help).toContain("distinct line patterns as well as colors");
});

test("large graphs meet generous interaction budgets and streamed activity stays bounded",async({page})=>{
  test.setTimeout(60_000);
  savedGraph=largeGraph(180);
  const loadStarted=Date.now();
  await page.getByRole("button",{name:"Open"}).click();
  await page.locator("#saved-graphs button").click();
  await expect(page.locator(".patch-node-card")).toHaveCount(180,{timeout:5_000});
  expect(Date.now()-loadStarted).toBeLessThan(5_000);

  const selectionStarted=Date.now();
  await page.locator("#graph-outline button").nth(90).dispatchEvent("click");
  await expect(page.locator("#graph-outline button").nth(90)).toHaveAttribute("aria-pressed","true");
  expect(Date.now()-selectionStarted).toBeLessThan(750);
  const searchStarted=Date.now();
  await page.locator("#canvas").dispatchEvent("dblclick",{clientX:300,clientY:300});
  await page.locator("#quick-add-search").fill("Audio pass-through");
  await expect(page.getByRole("option",{name:/Audio pass-through/})).toBeVisible();
  expect(Date.now()-searchStarted).toBeLessThan(1_000);
  await page.locator("#quick-add-cancel").click();
  await expect(page.locator("#quick-add-dialog")).toBeHidden();

  await page.getByRole("button",{name:"New"}).click();
  await expect(page.locator(".patch-node-card")).toHaveCount(0);
  await addFromPalette(page,"Microphone");
  meterStorm=true;
  await page.locator("#pipeline-name").fill("Meter storm");
  await page.evaluate(()=>{
    globalThis.__cardMutations=0;
    new MutationObserver(records=>globalThis.__cardMutations+=records.length)
      .observe(document.querySelector(".patch-node-cards"),{childList:true});
  });
  await page.getByRole("button",{name:"Run"}).click();
  await expect(page.locator("#run-state")).toHaveText("Completed");
  expect(await page.locator("#run-events li").count()).toBeLessThanOrEqual(200);
  await expect(page.locator("#run-events li").last()).toContainText("completed");
  expect(await page.evaluate(()=>globalThis.__cardMutations)).toBeLessThanOrEqual(20);
  await expect(page.locator(".patch-node-card")).toHaveCount(1);
});
