import {test,expect} from "@playwright/test";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";

const publicRoot=path.dirname(fileURLToPath(import.meta.url));
let server,baseUrl,savedGraph;
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
  },
  components,
};
const graph={
  schema_version:2,graph_id:"pipeline:browser-fixture",revision:7,
  metadata:{name:"Browser fixture",description:"",allow_unsafe_execution:false,labels:{"studio.layout.v1":JSON.stringify({
    "node:asr":{x:420,y:200},"node:mic-1":{x:120,y:150},"node:mic-2":{x:120,y:310},
    "node:sink-1":{x:720,y:140},"node:sink-2":{x:720,y:300},
  })}},
  nodes:[
    {id:"node:asr",kind:"asr",component_id:"base",config:{language:"en",timestamps:true},disabled:false,bypassed:false},
    {id:"node:mic-1",kind:"microphone",component_id:null,config:{},disabled:false,bypassed:false},
    {id:"node:mic-2",kind:"microphone",component_id:null,config:{},disabled:false,bypassed:false},
    {id:"node:sink-1",kind:"transcript_sink",component_id:null,config:{},disabled:false,bypassed:false},
    {id:"node:sink-2",kind:"transcript_sink",component_id:null,config:{},disabled:false,bypassed:false},
  ],
  edges:[],selected_sinks:[{node_id:"node:asr",port_id:"committed"}],
};

function cytoscapeStub(){
  const noop=()=>{};
  return ()=>{
    const elements=new Map();
    const element=id=>{
      const data=elements.get(id);
      if(!data)return{length:0,select:noop,addClass:noop,removeClass:noop};
      return{
        length:1,select:noop,addClass:noop,removeClass:noop,
        position:()=>data.position??{x:0,y:0},
        renderedPosition:()=>data.position??{x:0,y:0},
      };
    };
    return{
      on:noop,off:noop,fit:noop,panBy:noop,extent:()=>({x1:0,y1:0,x2:840,y2:560}),
      add:items=>items.forEach(item=>elements.set(item.data.id,item)),
      elements:()=>({remove:()=>elements.clear(),unselect:noop}),
      nodes:()=>({removeClass:noop}),
      getElementById:element,
    };
  };
}

test.beforeAll(async()=>{
  server=http.createServer((request,response)=>{
    const pathname=new URL(request.url,"http://fixture").pathname;
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

test.beforeEach(async({page})=>{
  savedGraph=null;
  await page.addInitScript(stub=>{globalThis.cytoscape=eval(`(${stub})`)();},cytoscapeStub.toString());
  await page.route("https://cdn.jsdelivr.net/**",route=>route.abort());
  await page.route("**/api/pipeline/**",async route=>{
    const request=route.request(),url=new URL(request.url()),pathname=url.pathname;
    if(pathname==="/api/pipeline/catalog")return route.fulfill({json:discovery});
    if(pathname==="/api/pipeline/starters")return route.fulfill({json:{graphs:[graph]}});
    if(pathname==="/api/pipeline/validate")return route.fulfill({json:{valid:true,diagnostics:[]}});
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
  await jack("node:asr","output").dragTo(jack("node:sink-2","input"));
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

  const audioConnection=page.locator(`.patch-connection-list button[data-edge-id="${audioEdge.id}"]`);
  await audioConnection.focus();
  await expect(audioConnection).toBeFocused();
  await page.keyboard.press("Delete");
  await expect(page.locator(".patch-cable")).toHaveCount(2);
  await save();
  expect(savedGraph.edges.some(edge=>edge.id===audioEdge.id)).toBe(false);
  expect(savedGraph.edges).toHaveLength(2);
});
