/***********************
 * Server
 ***********************/
const SERVER_URL = 'http://localhost:5000';
let currentServer = SERVER_URL;

// Remember to change the URL IP is you use a new server
// const SERVER_URL = 'http://156.239.253.200:5000';
// let currentServer = SERVER_URL;



/***********************
 * Session (NEW)
 ***********************/
let sessionId = null;

async function ensureSession() {
  // 先用本地缓存
  sessionId = localStorage.getItem('session_id');
  if (sessionId) return sessionId;

  // 没有就向后端申请一个
  const res = await fetch(currentServer + '/new_session', { method: 'POST' });
  if (!res.ok) throw new Error(`new_session failed: ${res.status} ${res.statusText}`);
  const data = await res.json();
  if (!data || !data.success || !data.session_id) throw new Error('new_session: invalid response');
  sessionId = data.session_id;
  localStorage.setItem('session_id', sessionId);
  return sessionId;
}




/***********************
 * Fixed config ids (layout + model)
 ***********************/
const CONFIG_IDS = [
  'layout1_model1',
  'layout1_model2',
  'layout1_model3',
  'layout1_model4',
  'layout2_model1',
  'layout2_model2',
  'layout2_model3',
  'layout2_model4',
  'layout3_model1',
  'layout3_model2',
  'layout3_model3',
  'layout3_model4',
  'layout4_model1',
  'layout4_model2',
  'layout4_model3',
  'layout4_model4',
];



// const MODEL_PERMS = [
//   ["model1","model2","model3"],
//   ["model1","model3","model2"],
//   ["model2","model1","model3"],
//   ["model2","model3","model1"],
//   ["model3","model1","model2"],
//   ["model3","model2","model1"],
// ];
const MODEL_PERMS = [
  ["model1","model2","model3","model4"],
  ["model1","model2","model4","model3"],
  ["model1","model3","model2","model4"],
  ["model1","model3","model4","model2"],
  ["model1","model4","model2","model3"],
  ["model1","model4","model3","model2"],

  ["model2","model1","model3","model4"],
  ["model2","model1","model4","model3"],
  ["model2","model3","model1","model4"],
  ["model2","model3","model4","model1"],
  ["model2","model4","model1","model3"],
  ["model2","model4","model3","model1"],

  ["model3","model1","model2","model4"],
  ["model3","model1","model4","model2"],
  ["model3","model2","model1","model4"],
  ["model3","model2","model4","model1"],
  ["model3","model4","model1","model2"],
  ["model3","model4","model2","model1"],

  ["model4","model1","model2","model3"],
  ["model4","model1","model3","model2"],
  ["model4","model2","model1","model3"],
  ["model4","model2","model3","model1"],
  ["model4","model3","model1","model2"],
  ["model4","model3","model2","model1"],
];



function hashString(s){
  let h = 0;
  for (let i=0; i<s.length; i++) h = (h*31 + s.charCodeAt(i)) >>> 0;
  return h >>> 0;
}

// 用 ProlificID（优先）或 sessionId 做稳定映射；若都无就落到本地时间
function getModelOrderForParticipant(){
  const key = (prolificId?.value?.trim()) || sessionId || String(Date.now());
  const idx = hashString(key) % MODEL_PERMS.length;
  return MODEL_PERMS[idx];
}

function pick4LayoutsDistinct(){
  const layouts = ["layout1","layout2","layout3","layout4"];
  // 洗牌
  for (let i=layouts.length-1; i>0; i--){
    const j = Math.floor(Math.random()*(i+1));
    [layouts[i], layouts[j]] = [layouts[j], layouts[i]];
  }
  return layouts.slice(0,4);
}


function compactLog(log) {
  return {
    prolificId: log?.prolificId ?? "",
    age: log?.age ?? "",
    gender: log?.gender ?? "",
    assignment: log?.assignment ?? null,   // 原样保留
    rounds: (log?.rounds ?? []).map(r => ({
      task: r?.task,
      round: r?.round,
      persona: r?.persona,
      configId: r?.configId,
      keys: r?.keys ?? [],
      stepLogs: (r?.stepLogs ?? []).map(s => ({
        // 只保留 cumulative_reward；若缺失则置为 null
        cumulative_reward: (s && 'cumulative_reward' in s) ? s.cumulative_reward : null
      }))
    })),
    questionnaires: log?.questionnaires ?? null
  };
}


/***********************
 * DOM
 ***********************/
const el = (id) => document.getElementById(id);

const pageIntro       = el('page-intro');
const pageConsent     = el('page-consent');
const pageInstruction = el('page-instruction');

// New instruction pages
const pageInstruction1 = document.getElementById('page-instruction-1');
const pageInstruction2 = document.getElementById('page-instruction-2');
const btnToInstruction2 = document.getElementById('to-instruction-2');

// Old pageInstruction 引用改为 pageInstruction1 的地方：键盘路由用它判断可见性

const pageTask        = el('page-task');
const pageQs          = el('page-qs');
const pageEnd         = el('page-end');

const btnToConsent     = el('to-consent');
const btnToInstruction = el('to-instruction');
const btnStartTask1    = el('start-task-1');

const btnPlay    = el('btnPlay');
const btnRestart = el('btnRestart');
const btnNext    = el('btnNext');
const taskTag    = el('taskTag');
const roundLabel = el('roundLabel');
const roundProgress = el('roundProgress');
const stepsLeftSpan  = el('stepsLeft');

const qsTitle = el('qsTitle');
const qsNext  = el('qsNext');

const canvas = document.getElementById('gameCanvas');
const ctx    = canvas.getContext('2d');


const roundEndPanel = document.getElementById('round-end-panel');
const personaFidelityRadios = document.getElementsByName('personaFidelity');



// Practice canvas & button
const canvasPractice = document.getElementById('gameCanvas_practice');
const ctxPractice    = canvasPractice.getContext('2d');
const btnTryPractice = el('btnTryPractice');

// Persona input & history
const personaInput     = el('personaInput');
const personaHistoryEl = el('personaHistory');

const personaInputWrap  = document.getElementById('personaInputWrap');
const personaSelectWrap = document.getElementById('personaSelectWrap');
const personaSelect     = document.getElementById('personaSelect');

// Task2 可选 persona 列表（来自 Task1 4 个），以及已用集合
let personaOptionsTask2 = [];
let personaUsedTask2 = new Set();
let personaOptionsTask3 = [];
let personaUsedTask3 = new Set();
let personaOptionsTask4 = [];
let personaUsedTask4 = new Set();


/***********************
 * Form validation (intro / consent)
 ***********************/
const prolificId   = el('prolificId');
const age          = el('age');
const gender       = el('gender');
const consentCheck = el('consentCheck');





// New pre-task pages & elements
const pagePretask1 = document.getElementById('page-pretask-1');
const pagePretask2 = document.getElementById('page-pretask-2');

const btnEnterTask1 = document.getElementById('btnEnterTask1');
const btnEnterTask2 = document.getElementById('btnEnterTask2');

const canvasPretask1 = document.getElementById('gameCanvas_pretask1');
const ctxPretask1 = canvasPretask1?.getContext('2d');

const canvasPretask2 = document.getElementById('gameCanvas_pretask2');
const ctxPretask2 = canvasPretask2?.getContext('2d');

const pretask1RobotImg = document.getElementById('pretask1Robot');
const pretask2RobotImg = document.getElementById('pretask2Robot');



const pagePretask3 = document.getElementById('page-pretask-3');
const btnEnterTask3 = document.getElementById('btnEnterTask3');
const canvasPretask3 = document.getElementById('gameCanvas_pretask3');
const ctxPretask3 = canvasPretask3?.getContext('2d');
const pretask3RobotImg = document.getElementById('pretask3Robot');

let isPretask3Playing = false;


const pagePretask4 = document.getElementById('page-pretask-4');
const btnEnterTask4 = document.getElementById('btnEnterTask4');
const canvasPretask4 = document.getElementById('gameCanvas_pretask4');
const ctxPretask4 = canvasPretask4?.getContext('2d');
const pretask4RobotImg = document.getElementById('pretask4Robot');

let isPretask4Playing = false;




function validateIntro(){
  btnToConsent.disabled = !(prolificId.value.trim() && age.value && gender.value);
}
prolificId?.addEventListener('input', validateIntro);
age?.addEventListener('input', validateIntro);
gender?.addEventListener('change', validateIntro);

consentCheck?.addEventListener('change', ()=>{
  btnToInstruction.disabled = !consentCheck.checked;
});

/***********************
 * Navigation
 ***********************/
function show(section){
  [pageIntro,pageConsent,pageInstruction1,pageInstruction2,pagePretask1,pagePretask2,pagePretask3,pagePretask4,pageTask,pageQs,pageEnd]
    .forEach(p => p.classList.add('hidden'));

  section.classList.remove('hidden');

  // 离开练习页就停止 practice 键盘路由
  const onInstruction1 = (section === pageInstruction1);
  if (!onInstruction1) {
    isPracticePlaying = false;
    practiceGameOver = true;
  }

  // 离开 pretask 时也停掉预览键盘路由
  if (section !== pagePretask1) isPretask1Playing = false;
  if (section !== pagePretask2) isPretask2Playing = false;
  if (section !== pagePretask3) isPretask3Playing = false;
  if (section !== pagePretask4) isPretask4Playing = false;

}




/***********************
 * Images & drawing
 ***********************/
const TILE_SIZE = 80;
const tileNameMap = {
  0: "space.png",
  1: "counter.png",
  3: "FreshTomato.png",
  4: "FreshLettuce.png",
  5: "plate.png",
  6: "cutboard.png",
  7: "delivery.png",
  8: "FreshOnion.png",
  9: "dirtyplate.png",
  10: "BadLettuce.png"
};


const robotSkins = ["agent-robot.png", "agent-robot2.png", "agent-robot3.png", "agent-robot4.png"];
let robotSkinTask1 = null;
let robotSkinTask2 = null;
let robotSkinTask3 = null;
let robotSkinTask4 = null;

let currentPersonaFidelity = null; // 1~7，用户自评


function shuffleArray(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// 在开始 Task1 之前调用
function assignRobotSkins() {
  const shuffled = shuffleArray(robotSkins);
  robotSkinTask1 = shuffled[0];
  robotSkinTask2 = shuffled[1];
  robotSkinTask3 = shuffled[2];
  robotSkinTask4 = shuffled[3];
}





const images = {};
function preloadImages(callback) {
  let loadedImages = 0;
  const imageNames = [
    "space.png", "counter.png", "FreshTomato.png", "ChoppedTomato.png",
    "FreshLettuce.png", "ChoppedLettuce.png", "plate.png", "cutboard.png",
    "delivery.png", "FreshOnion.png", "ChoppedOnion.png", "dirtyplate.png",
    "BadLettuce.png", "agent-red.png", "agent-blue.png", "agent-robot.png", "agent-robot2.png", "agent-robot3.png", "agent-robot4.png"
  ];
  imageNames.forEach(name => {
    images[name] = new Image();
    images[name].src = "static/images/" + name;
    images[name].onload = () => { if (++loadedImages === imageNames.length) callback(); };
    images[name].onerror = () => { if (++loadedImages === imageNames.length) callback(); };
  });
}

// // Draw helper for any canvas
// function drawStateToCanvas(state, canvasEl, ctxEl) {
//   if (!state || !state.map || !state.xlen || !state.ylen) return;

//   canvasEl.width = state.ylen * TILE_SIZE;
//   canvasEl.height = state.xlen * TILE_SIZE;

//   // map
//   for (let x = 0; x < state.xlen; x++) {
//     for (let y = 0; y < state.ylen; y++) {
//       const tile = state.map[x][y];
//       const imgName = tileNameMap[tile] || "space.png";
//       const img = images[imgName];
//       if (img) ctxEl.drawImage(img, y*TILE_SIZE, x*TILE_SIZE, TILE_SIZE, TILE_SIZE);
//     }
//   }

//   // items
//   const holdingPositions = new Set((state.agents||[]).map(a => `${a.x},${a.y}`));
//   (state.items||[]).forEach(item => {
//     const posKey = `${item.x},${item.y}`;
//     if (!holdingPositions.has(posKey)) {
//       const counterImg = images["counter.png"];
//       if (counterImg) ctxEl.drawImage(counterImg, item.y*TILE_SIZE, item.x*TILE_SIZE, TILE_SIZE, TILE_SIZE);

//       const baseImg = images[(item.type||'') + ".png"];
//       if (baseImg) ctxEl.drawImage(baseImg, item.y*TILE_SIZE, item.x*TILE_SIZE, TILE_SIZE, TILE_SIZE);

//       if (item.containing) {
//         const containedImg = images[item.containing + ".png"];
//         if (containedImg) ctxEl.drawImage(containedImg, item.y*TILE_SIZE, item.x*TILE_SIZE, TILE_SIZE, TILE_SIZE);
//       }
//       if (item.holding) {
//         const holdingImg = images[item.holding + ".png"];
//         if (holdingImg) ctxEl.drawImage(holdingImg, item.y*TILE_SIZE, item.x*TILE_SIZE, TILE_SIZE, TILE_SIZE);
//       }
//     }
//   });

//   // agents
//   (state.agents||[]).forEach(agent => {

//     let agentImg = null;

//     if (agent.color === "robot") {
//       if (isPracticePlaying) {
//         agentImg = images["agent-robot.png"];   // Practice 固定
//       } else if (currentTask === 1) {
//         agentImg = images[robotSkinTask1];
//       } else if (currentTask === 2) {
//         agentImg = images[robotSkinTask2];
//       } else if (currentTask === 3) {
//         agentImg = images[robotSkinTask3];
//       }
//     } else {
//       agentImg = images[`agent-${agent.color}.png`];
//     }


//     if (agentImg) {
//       ctxEl.drawImage(agentImg, agent.y*TILE_SIZE, agent.x*TILE_SIZE, TILE_SIZE, TILE_SIZE);
//     }




//     if (agent.holding) {
//       const holdImg = images[agent.holding + ".png"];
//       if (holdImg) {
//         ctxEl.drawImage(
//           holdImg,
//           agent.y*TILE_SIZE + TILE_SIZE*0.5,
//           agent.x*TILE_SIZE + TILE_SIZE*0.5,
//           TILE_SIZE*0.5, TILE_SIZE*0.5
//         );
//       }
//       if ((agent.holding === "plate" || agent.holding === "dirtyplate") && agent.holding_containing) {
//         const containedImg = images[agent.holding_containing + ".png"];
//         if (containedImg) {
//           ctxEl.drawImage(
//             containedImg,
//             agent.y*TILE_SIZE + TILE_SIZE*(0.5 + 0.25*0.5),
//             agent.x*TILE_SIZE + TILE_SIZE*(0.5 + 0.25*0.5),
//             TILE_SIZE*0.5*0.7, TILE_SIZE*0.5*0.7
//           );
//         }
//       }
//     }
//   });
// }

// function drawStateToCanvas(state, canvasEl, ctxEl) {
//   if (!state || !state.map || !state.xlen || !state.ylen) return;

//   // 保持标签上设定的像素尺寸，不再改 canvas.width/height
//   const cw = canvasEl.width  || 800;
//   const ch = canvasEl.height || 800;
//   ctxEl.clearRect(0, 0, cw, ch);

//   // 计算每格像素并居中
//   const cell = Math.floor(Math.min(cw / state.ylen, ch / state.xlen));
//   const drawW = cell * state.ylen;
//   const drawH = cell * state.xlen;
//   const offX = Math.floor((cw - drawW) / 2);
//   const offY = Math.floor((ch - drawH) / 2);

//   // map
//   for (let x = 0; x < state.xlen; x++) {
//     for (let y = 0; y < state.ylen; y++) {
//       const img = images[tileNameMap[state.map[x][y]] || "space.png"];
//       if (img) ctxEl.drawImage(img, offX + y*cell, offY + x*cell, cell, cell);
//     }
//   }

//   // items
//   const holding = new Set((state.agents||[]).map(a => `${a.x},${a.y}`));
//   (state.items||[]).forEach(item => {
//     const key = `${item.x},${item.y}`;
//     if (holding.has(key)) return;
//     const draw = (name) => {
//       const img = images[name + ".png"];
//       if (img) ctxEl.drawImage(img, offX + item.y*cell, offY + item.x*cell, cell, cell);
//     };
//     const counterImg = images["counter.png"];
//     if (counterImg) ctxEl.drawImage(counterImg, offX + item.y*cell, offY + item.x*cell, cell, cell);
//     if (item.type) draw(item.type);
//     if (item.containing) draw(item.containing);
//     if (item.holding) draw(item.holding);
//   });

//   // agents
//   (state.agents||[]).forEach(agent => {
//     let agentImg = null;
//     if (agent.color === "robot") {
//       if (isPracticePlaying) agentImg = images["agent-robot.png"];
//       else if (currentTask === 1) agentImg = images[robotSkinTask1];
//       else if (currentTask === 2) agentImg = images[robotSkinTask2];
//       else if (currentTask === 3) agentImg = images[robotSkinTask3];
//     } else {
//       agentImg = images[`agent-${agent.color}.png`];
//     }
//     if (agentImg) ctxEl.drawImage(agentImg, offX + agent.y*cell, offY + agent.x*cell, cell, cell);

//     if (agent.holding) {
//       const holdImg = images[agent.holding + ".png"];
//       if (holdImg) {
//         // 盘子相对 agent 所在格子的偏移与尺寸
//         const PLATE_SCALE = 0.5;       // 盘子占格子的比例（你原来就是 0.5）
//         const plateW = cell * PLATE_SCALE;
//         const plateH = cell * PLATE_SCALE;
//         const plateX = offX + agent.y * cell + (cell - plateW); // 你原来放右下：+ cell*0.5
//         const plateY = offY + agent.x * cell + (cell - plateH);
//         // 如果还是想放在右下角，保持上面两行；想放左下/右上可改偏移。
//         // 若你想完全复刻原来的右下角写法，也可：
//         // const plateX = offX + agent.y*cell + cell*0.5;
//         // const plateY = offY + agent.x*cell + cell*0.5;

//         // 画盘子
//         ctxEl.drawImage(holdImg, plateX, plateY, plateW, plateH);

//         // 若盘子里有东西 → 在盘子矩形内部居中绘制（相对盘子偏移）
//         if ((agent.holding === "plate" || agent.holding === "dirtyplate") && agent.holding_containing) {
//           const contentImg = images[agent.holding_containing + ".png"];
//           if (contentImg) {
//             const CONTENT_SCALE = 0.9;              // 蔬菜相对盘子的缩放（60–70%）
//             const contentW = plateW * CONTENT_SCALE; // 更小一点
//             const contentH = plateH * CONTENT_SCALE;
//             const contentX = plateX + (plateW - contentW) / 2; // 在盘子矩形居中
//             const contentY = plateY + (plateH - contentH) / 2;
//             ctxEl.drawImage(contentImg, contentX, contentY, contentW, contentH);
//           }
//         }
//       }
//     }

//   });
// }


function drawStateToCanvas(state, canvasEl, ctxEl) {
  if (!state || !state.map || !state.xlen || !state.ylen) return;

  // === 工具：在给定矩形内画盘子，并把内容(菜)按盘子缩放后居中 ===
  function drawPlateWithContent(x, y, w, h, plateName, contentName) {
    const plateImg = images[plateName + ".png"];
    if (plateImg) ctxEl.drawImage(plateImg, x, y, w, h);

    if (contentName) {
      const contentImg = images[contentName + ".png"];
      if (contentImg) {
        const CONTENT_SCALE = 0.65; // 菜相对盘子的缩放（可按需微调）
        const cw = w * CONTENT_SCALE;
        const ch = h * CONTENT_SCALE;
        const cx = x + (w - cw) / 2;
        const cy = y + (h - ch) / 2;
        ctxEl.drawImage(contentImg, cx, cy, cw, ch);
      }
    }
  }

  // === 画布清理 & 棋盘几何 ===
  const cw = canvasEl.width  || 800;
  const ch = canvasEl.height || 800;
  ctxEl.clearRect(0, 0, cw, ch);

  const cell  = Math.floor(Math.min(cw / state.ylen, ch / state.xlen));
  const drawW = cell * state.ylen;
  const drawH = cell * state.xlen;
  const offX  = Math.floor((cw - drawW) / 2);
  const offY  = Math.floor((ch - drawH) / 2);

  // === 地图 ===
  for (let x = 0; x < state.xlen; x++) {
    for (let y = 0; y < state.ylen; y++) {
      const img = images[tileNameMap[state.map[x][y]] || "space.png"];
      if (img) ctxEl.drawImage(img, offX + y*cell, offY + x*cell, cell, cell);
    }
  }

  // === 台面上的物品（不包含被拿在手里的格子）===
  const holdingCells = new Set((state.agents||[]).map(a => `${a.x},${a.y}`));
  (state.items||[]).forEach(item => {
    const key = `${item.x},${item.y}`;
    if (holdingCells.has(key)) return;

    const baseX = offX + item.y * cell;
    const baseY = offY + item.x * cell;

    // 有些关卡用到的“台面底图”
    const counterImg = images["counter.png"];
    if (counterImg) ctxEl.drawImage(counterImg, baseX, baseY, cell, cell);

    // 盘子：统一走“盘子+内容缩放居中”的逻辑
    if (item.type === "plate" || item.type === "dirtyplate") {
      drawPlateWithContent(baseX, baseY, cell, cell, item.type, item.containing || null);
    } else {
      // 其他物品：按一格全尺寸画
      const baseImg = images[(item.type||'') + ".png"];
      if (baseImg) ctxEl.drawImage(baseImg, baseX, baseY, cell, cell);

      // 某些物品可能还有 containing/holding 字段，这里保持原有叠加画法
      if (item.containing) {
        const containedImg = images[item.containing + ".png"];
        if (containedImg) ctxEl.drawImage(containedImg, baseX, baseY, cell, cell);
      }
      if (item.holding) {
        const holdingImg = images[item.holding + ".png"];
        if (holdingImg) ctxEl.drawImage(holdingImg, baseX, baseY, cell, cell);
      }
    }
  });

  // === 角色（含机器人皮肤与手持物）===
  (state.agents||[]).forEach(agent => {
    // 角色图
    let agentImg = null;
    if (agent.color === "robot") {
      if (isPracticePlaying) agentImg = images["agent-robot.png"];
      else if (currentTask === 1) agentImg = images[robotSkinTask1];
      else if (currentTask === 2) agentImg = images[robotSkinTask2];
      else if (currentTask === 3) agentImg = images[robotSkinTask3];
      else if (currentTask === 4) agentImg = images[robotSkinTask4];
    } else {
      agentImg = images[`agent-${agent.color}.png`];
    }
    if (agentImg) {
      ctxEl.drawImage(agentImg, offX + agent.y*cell, offY + agent.x*cell, cell, cell);
    }

    // 手持物
    if (agent.holding) {
      const holdName = agent.holding; // 如 'plate' / 'dirtyplate' / 其他
      const holdImg  = images[holdName + ".png"];
      if (!holdImg) return;

      // 统一把“手持物”放在当前格子的右下角（保持你原来风格）
      const SCALE_IN_HAND = 0.5;      // 手持物相对一格大小
      const w = cell * SCALE_IN_HAND;
      const h = cell * SCALE_IN_HAND;
      const x = offX + agent.y*cell + (cell - w);
      const y = offY + agent.x*cell + (cell - h);

      if (holdName === "plate" || holdName === "dirtyplate") {
        // 盘子在手中：菜按盘子缩小并在盘子矩形内居中
        const contentName = agent.holding_containing || null;
        drawPlateWithContent(x, y, w, h, holdName, contentName);
      } else {
        // 其他手持物品：直接画小图在右下角
        ctxEl.drawImage(holdImg, x, y, w, h);
      }
    }
  });
}



// Thin wrapper for the main game canvas
function drawState(state) {
  drawStateToCanvas(state, canvas, ctx);
}

/***********************
 * Practice state & logic
 ***********************/
let isPracticePlaying = false;
let practiceGameOver  = false;
let practiceCongratsShown = false;

// 练习累计奖励 gating（你当前代码用 100，如需 600 改这里常量）
let practiceCumulativeReward = 0;
const PRACTICE_PASS_SCORE = 100;

async function startPracticeRound() {
  isPracticePlaying = false;
  practiceGameOver = false;

  try {
    const data = await postJSON(currentServer + '/reset', { config_id: 'layout_practice' });
    if (data.state) {
      // ✅ 先进入 practice 模式
      isPracticePlaying = true;

      // 再画第一帧
      drawStateToCanvas(data.state, canvasPractice, ctxPractice);

      practiceCumulativeReward = data.cumulative_reward ?? 0;
      updateStartTaskGate(); // 或你的 updatePracticeGateUI / updateStartTaskGate
      // isPracticePlaying = true;  // ← 删除这行（已经提前设置了）
    }
  } catch (err) {
    console.error(err);
    alert('Practice reset failed: ' + err.message);
  }
}


btnTryPractice?.addEventListener('click', startPracticeRound);











/***********************
 * Task 1和2之前的练习轮
 ***********************/



// Pre-task preview flags
let isPretask1Playing = false;
let isPretask2Playing = false;

// 通用：启动一个预览（用真实 config）
async function startPretaskRound(which /* 1/2/3 */) {
  try {
    let configId = currentConfigId;
    if (which === 2) configId = participantAssignment.task2.configId;
    if (which === 3) configId = participantAssignment.task3.configId;
    if (which === 4) configId = participantAssignment.task4.configId;

    const data = await postJSON(currentServer + '/reset', { config_id: configId });
    if (!data.state) return;

    if (which === 1) {
      isPretask1Playing = true;
      drawStateToCanvas(data.state, canvasPretask1, ctxPretask1);
    } else if (which === 2) {
      isPretask2Playing = true;
      drawStateToCanvas(data.state, canvasPretask2, ctxPretask2);
    } else if (which === 3) {
      isPretask3Playing = true;
      drawStateToCanvas(data.state, canvasPretask3, ctxPretask3);
    } else {
      isPretask4Playing = true;
      drawStateToCanvas(data.state, canvasPretask4, ctxPretask4);
    }
  } catch (err) {
    console.error(err);
    alert('Preview reset failed: ' + err.message);
  }
}





/***********************
 * Global runtime state
 ***********************/
let currentTask = 1;      // 1 or 2
let currentRound = 1;     // 1..8
let isPlaying = false;
let gameOver   = false;



let resetWarningShown = false;   // ✅ 本轮是否已经提示过“即将重置”




// Persona
let currentPersona = "";
let personaHistoryTask1 = [];
let personaHistoryTask2 = [];
let personaHistoryTask3 = [];
let personaHistoryTask4 = [];


// Step logs for the current round
let currentRoundSteps = [];  // [{ t, key, state, cumulative_reward, steps_left }]

// Assignment & logging
let currentConfigId = null; // e.g., 'layout1_model2'
let currentLayoutId = null;
let currentModelId  = null;



let participantAssignment = {
  task1: { configId: null },
  task2: { configId: null },
  task3: { configId: null },
  task4: { configId: null },
};

const logData = {
  prolificId:"", age:"", gender:"",
  assignment: participantAssignment,
  rounds: [],
  questionnaires: { task1: null, task2: null, task3: null, task4: null }   // 👈 多一个 task3
};









/***********************
 * Utils
 ***********************/
function sampleOne(list, exclude = new Set()){
  const candidates = list.filter(x => !exclude.has(x));
  if (candidates.length === 0) return list[Math.floor(Math.random()*list.length)];
  return candidates[Math.floor(Math.random()*candidates.length)];
}
function parseConfigId(cfgId){
  const m = cfgId.split('_');
  return { layoutId: m[0] || null, modelId: m[1] || null };
}

// 规范化 persona：去首尾空格、合并多空格、小写
function normalizePersona(s) {
  return (s || "")
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();
}



/***********************
 * Assignment for two tasks
 ***********************/

function assignConfigsForParticipant(){
  // ✅ model 顺序仍然 counterbalanced
  const models = getModelOrderForParticipant();   // e.g., ["model3","model1","model4","model2"]

  // ✅ layout 也参与均衡（基于 participant id 稳定映射）
  const allLayouts = ["layout1","layout2","layout3","layout4"];

  const key = (prolificId?.value?.trim()) || sessionId || String(Date.now());
  const layoutIdx = hashString(key) % allLayouts.length;
  const chosenLayout = allLayouts[layoutIdx];

  participantAssignment = {
    task1: { configId: `${chosenLayout}_${models[0]}` },
    task2: { configId: `${chosenLayout}_${models[1]}` },
    task3: { configId: `${chosenLayout}_${models[2]}` },
    task4: { configId: `${chosenLayout}_${models[3]}` },
  };

  logData.assignment = participantAssignment;

  // 当前任务初始化到 task1
  currentTask = 1;
  currentConfigId = participantAssignment.task1.configId;
  ({layoutId: currentLayoutId, modelId: currentModelId} = parseConfigId(currentConfigId));
}



/***********************
 * Header & persona history
 ***********************/


function getPersonaHistory(){
  if (currentTask === 1) return personaHistoryTask1;
  if (currentTask === 2) return personaHistoryTask2;
  if (currentTask === 3) return personaHistoryTask3;
  return personaHistoryTask4;
}


function updateHeader(){
  // Task 1 / 4
  taskTag.textContent = `Task ${currentTask} / 4`;

  // Round 1 / 2
  roundLabel.textContent = `Round ${currentRound} / 2`;

  // Progress bar: 从 0 → 100%
  roundProgress.style.width = `${((currentRound - 1) / 2) * 100}%`;

  // persona history update
  if (personaHistoryEl) {
    personaHistoryEl.innerHTML = "";
    getPersonaHistory().forEach((txt, idx) => {
      const s = document.createElement('span');
      s.className = 'pill';
      s.textContent = `${idx+1}. ${txt}`;
      personaHistoryEl.appendChild(s);
    });
  }
}



/***********************
 * Requests
 ***********************/

async function postJSON(url, body = {}) {
  // 自动注入 session_id
  try {
    if (!sessionId) await ensureSession();
  } catch (e) {
    console.error(e);
    throw e;
  }
  const payload = { ...body, session_id: sessionId };

  const res = await fetch(url, {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  });
  let data = {};
  try { data = await res.json(); } catch(e){}
  if (!res.ok || data.success === false) {
    const msg = (data && data.error) ? data.error : `${res.status} ${res.statusText}`;
    throw new Error(msg);
  }
  return data;
}

// 如果你仍然想保留 GET 工具函数可以留着，但 get_state 改成 POST 更省心
function fetchInitialState(){
  return postJSON(currentServer + '/get_state', {}); // 带 session_id
}



async function getJSON(url){
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

/***********************
 * Keyboard handling
 ***********************/

document.addEventListener('keydown', async (event) => {
  const keysToPrevent = ['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];
  if (keysToPrevent.includes(event.key)) event.preventDefault();

  // Practice-first routing
  if (isPracticePlaying && !practiceGameOver) {
    try {
      const data = await postJSON(currentServer + '/key_event', {
        key: event.key,
        config_id: 'layout_practice'
      });
      if (data.state) {
        drawStateToCanvas(data.state, canvasPractice, ctxPractice);
      }
      if ('cumulative_reward' in data) {
        practiceCumulativeReward = data.cumulative_reward;
        updateStartTaskGate(); // 替换 updatePracticeGateUI()
        if (practiceCumulativeReward >= PRACTICE_PASS_SCORE && !practiceCongratsShown) {
          practiceCongratsShown = true;
          alert('Congrats! You have mastered the practice.');
        }
      }
      if ('steps_left' in data && data.steps_left <= 0) {
        practiceGameOver = true;
        isPracticePlaying = false;
        alert('Practice Finished!');
      }
    } catch (err) {
      console.error(err);
      alert('Failed to send practice key event: ' + err.message);
    }
    return; // do not fall through to main handler
  }



  const pretask1Visible = pagePretask1 && !pagePretask1.classList.contains('hidden');
  const pretask2Visible = pagePretask2 && !pagePretask2.classList.contains('hidden');
  const pretask3Visible = pagePretask3 && !pagePretask3.classList.contains('hidden');
  const pretask4Visible = pagePretask4 && !pagePretask4.classList.contains('hidden');

  // === PRETASK 1 键盘 ===
  if (isPretask1Playing && pretask1Visible) {
    event.preventDefault();
    try {
      const data = await postJSON(currentServer + '/key_event', {
        key: event.key,
        config_id: currentConfigId  // Task 1 的真实 config
      });
      if (data.state) drawStateToCanvas(data.state, canvasPretask1, ctxPretask1);
      // 不需要弹窗/面板，预览不限步数；如需要也可参考 practice 分支处理 steps_left
    } catch (err) {
      console.error(err);
    }
    return;
  }

  // === PRETASK 2 键盘 ===
  if (isPretask2Playing && pretask2Visible) {
    event.preventDefault();
    try {
      const data = await postJSON(currentServer + '/key_event', {
        key: event.key,
        config_id: participantAssignment.task2.configId  // Task 2 的真实 config
      });
      if (data.state) drawStateToCanvas(data.state, canvasPretask2, ctxPretask2);
    } catch (err) {
      console.error(err);
    }
    return;
  }


  if (isPretask3Playing && pretask3Visible) {
    event.preventDefault();
    try {
      const data = await postJSON(currentServer + '/key_event', {
        key: event.key,
        config_id: participantAssignment.task3.configId
      });
      if (data.state) drawStateToCanvas(data.state, canvasPretask3, ctxPretask3);
    } catch (err) {
      console.error(err);
    }
    return;
  }

  if (isPretask4Playing && pretask4Visible) {
    event.preventDefault();
    try {
      const data = await postJSON(currentServer + '/key_event', {
        key: event.key,
        config_id: participantAssignment.task4.configId
      });
      if (data.state) drawStateToCanvas(data.state, canvasPretask4, ctxPretask4);
    } catch (err) {
      console.error(err);
    }
    return;
  }



  // Main game
  if (!isPlaying || gameOver) return;

  try {
    const data = await postJSON(currentServer + '/key_event', {
      key: event.key,
      config_id: currentConfigId
    });
    if (data.state) drawState(data.state);

    // Step log
    currentRoundSteps.push({
      t: Date.now(),
      key: event.key,
      state: sanitizeState(data.state) || null,  // 清洗 state
      cumulative_reward: data.cumulative_reward ?? null,
      steps_left: ('steps_left' in data) ? data.steps_left : null
    });


    if ('steps_left' in data) {
      stepsLeftSpan.textContent = data.steps_left;

      // ✅ 根据 layout 不同，控制 reset 提醒时机
      const isLayout4 = currentLayoutId === 'layout4';

      // console.log(isLayout4)

      // layout4: 在 140 和 70 提醒
      if (isLayout4) {
        if ((data.steps_left === 130 || data.steps_left === 60) && !resetWarningShown) {
          // resetWarningShown = true;
          alert("⚠️ The item positions will be reset now. Please continue behaving the way you just did.");
        }
      }
      // 其他 layout: 仍然在 100 提醒
      else {
        if (data.steps_left === 100 && !resetWarningShown) {
          // resetWarningShown = true;
          alert("⚠️ The item positions will be reset now. Please continue behaving the way you just did.");
        }
      }

      if (data.steps_left <= 0 && !gameOver) {
        gameOver = true;
        isPlaying = false;
        btnRestart.disabled = true;

        if (roundEndPanel) {
          roundEndPanel.classList.remove('hidden');
          btnNext.disabled = true;
          roundEndPanel.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
      }
    }



  } catch (err) {
    console.error(err);
    alert('Failed to send key event: ' + err.message);
  }
});

/***********************
 * Round control
 ***********************/
async function resetServerState(){
  return postJSON(currentServer + '/reset', { config_id: currentConfigId });
}



// async function startRound(){
//   // Persona required
//   // Persona required
//   if (currentTask === 1) {
//     currentPersona = (personaInput?.value || "").trim();
//     if (!currentPersona) { alert('Please enter your persona for this round'); return; }
//   } else {
//     // Task 2 走下拉
//     const idx = personaSelect?.value || "";
//     if (!idx) { alert('Please choose a persona used in Task 1'); return; }
//     currentPersona = personaOptionsTask2[parseInt(idx, 10)] || "";
//     if (!currentPersona) { alert('Please choose a persona used in Task 1'); return; }
//   }
//   if (!currentConfigId) { alert('Config_id Not Assigned'); return; }

//   isPlaying = false; gameOver = false;
//   btnNext.disabled = true;
//   btnRestart.disabled = true;
//   stepsLeftSpan.textContent = '—';

//   // 清空当前轮步日志
//   currentRoundSteps = [];

//   // NEW: 隐藏回合结束面板 + 重置评分
//   if (roundEndPanel) roundEndPanel.classList.add('hidden');
//   currentPersonaFidelity = null;
//   personaFidelityRadios.forEach(r => r.checked = false);

//   try{
//     const data = await resetServerState();
//     if (data.state) {
//       drawState(data.state);
//       stepsLeftSpan.textContent = data.steps_left ?? '—';
//       isPlaying = true;
//       btnRestart.disabled = false;

//       // initial snapshot
//       currentRoundSteps.push({
//         t: Date.now(),
//         key: 'RESET',
//         state: data.state,
//         cumulative_reward: data.cumulative_reward ?? 0,
//         steps_left: data.steps_left ?? null
//       });
//     }
//   }catch(err){
//     console.error(err);
//     alert('Reset failed: ' + err.message);
//   }
// }


async function startRound(){
  // Persona required
  if (currentTask === 1) {
    currentPersona = (personaInput?.value || "").trim();
    if (!currentPersona) { alert('Please enter your persona for this round'); return; }

    // === 新增：Task 1 禁止重复 persona ===
    const norm = normalizePersona(currentPersona);
    const hasDup = personaHistoryTask1.some(p => normalizePersona(p) === norm);
    if (hasDup) {
      alert('This persona was already used in Task 1. Please describe a different persona for this round.');
      return; // 阻止开始
    }
    // === 新增结束 ===

  } else {
    // Task 2/3：下拉选择，已有“不可重复”逻辑，这里保持原样
    const idx = personaSelect?.value || "";
    if (!idx) { alert('Please choose a persona used in Task 1'); return; }
    // const opts = (currentTask === 2) ? personaOptionsTask2 : personaOptionsTask3;

    const opts =
      (currentTask === 2) ? personaOptionsTask2 :
      (currentTask === 3) ? personaOptionsTask3 :
      personaOptionsTask4;

    currentPersona = opts[parseInt(idx, 10)] || "";
    if (!currentPersona) { alert('Please choose a persona used in Task 1'); return; }
  }

  if (!currentConfigId) { alert('Config_id Not Assigned'); return; }

  isPlaying = false; gameOver = false;
  btnNext.disabled = true;
  btnRestart.disabled = true;
  stepsLeftSpan.textContent = '—';

  // 清空当前轮步日志
  currentRoundSteps = [];

  // ✅ 新增
  resetWarningShown = false;

  // 隐藏回合结束面板 + 重置评分
  if (roundEndPanel) roundEndPanel.classList.add('hidden');
  currentPersonaFidelity = null;
  personaFidelityRadios.forEach(r => r.checked = false);

  try{
    const data = await resetServerState();
    if (data.state) {
      drawState(data.state);
      stepsLeftSpan.textContent = data.steps_left ?? '—';
      isPlaying = true;
      btnRestart.disabled = false;

      // initial snapshot
      // currentRoundSteps.push({
      //   t: Date.now(),
      //   key: 'RESET',
      //   state: data.state,
      //   cumulative_reward: data.cumulative_reward ?? 0,
      //   steps_left: data.steps_left ?? null
      // });

      currentRoundSteps.push({
        t: Date.now(),
        key: 'RESET',
        state: sanitizeState(data.state), // ← 走清洗
        cumulative_reward: data.cumulative_reward ?? 0,
        steps_left: data.steps_left ?? null
      });


    }
  }catch(err){
    console.error(err);
    alert('Reset failed: ' + err.message);
  }
}



function setPersonaHighlight(needsInput) {
  if (!personaInput) return;
  if (needsInput) {
    personaInput.style.border = "2px solid #facc15";                 // 黄色边框
    personaInput.style.boxShadow = "0 0 4px rgba(250,204,21,0.6)";   // 黄色光晕
  } else {
    personaInput.style.border = "1px solid #e5e7eb";                 // 恢复默认
    personaInput.style.boxShadow = "none";
  }
}



function clearMainCanvas() {
  if (!canvas || !ctx) return;
  canvas.width = 800;    // 你的默认尺寸
  canvas.height = 800;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}




function endRoundAndNext(){
  // 必须先回答本轮自评
  if (currentPersonaFidelity == null) {
    alert('Please rate how well you followed your persona for this round.');
    return;
  }

  // 从本轮步骤日志中抽取按键
  const keySequence = currentRoundSteps
    .filter(s => s && s.key && s.key !== 'RESET')
    .map(s => s.key);

  // 记录轮信息（不再存 layout 信息；保留 configId）
  logData.rounds.push({
    task: currentTask,
    round: currentRound,
    persona: currentPersona,            // 你已有的 persona 文本
    configId: currentConfigId,
    keys: keySequence,
    stepLogs: currentRoundSteps.slice(),
    personaFidelity: currentPersonaFidelity   // ✅ 新增：本轮自评分
  });


  // 加入历史 persona
  if (currentTask === 1) {
    personaHistoryTask1.push(currentPersona);
  } else {
    personaHistoryTask2.push(currentPersona);
  }

  // 进入下一轮或问卷
  if (currentRound < 2) {
    currentRound += 1;

    if (currentTask === 2 || currentTask === 3 || currentTask === 4) {
      // const opts = (currentTask === 2) ? personaOptionsTask2 : personaOptionsTask3;
      // const used = (currentTask === 2) ? personaUsedTask2 : personaUsedTask3;

      const opts =
          (currentTask === 2) ? personaOptionsTask2 :
          (currentTask === 3) ? personaOptionsTask3 :
          personaOptionsTask4;

      const used =
          (currentTask === 2) ? personaUsedTask2 :
          (currentTask === 3) ? personaUsedTask3 :
          personaUsedTask4;


      const usedIdx = opts.findIndex(p => p === currentPersona);
      if (usedIdx >= 0) used.add(usedIdx);
      if (personaSelect) personaSelect.value = "";
      renderPersonaSelectOptions();
      if (btnPlay) btnPlay.disabled = true;
    } else {
      if (personaInput) {
        personaInput.value = "";
        currentPersona = "";
        btnPlay.disabled = true;
      }
      setPersonaHighlight(true);
    }




    // 隐藏结束面板 & 重置评分
    if (roundEndPanel) roundEndPanel.classList.add('hidden');
    currentPersonaFidelity = null;
    personaFidelityRadios.forEach(r => r.checked = false);

    updateHeader();
    btnRestart.disabled = true;
    btnNext.disabled = true;
    stepsLeftSpan.textContent = '—';
    canvas.width = 800; canvas.height = 800;

    // ✅ 新增：进入下一轮后，滚动到页面顶部
    requestAnimationFrame(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });


  } else {

    // 4轮完 → 问卷
    if (roundEndPanel) roundEndPanel.classList.add('hidden');
    personaFidelityRadios.forEach(r => r.checked = false);
    currentPersonaFidelity = null;

    qsTitle.textContent = `Task ${currentTask} Questionnaire`;
    show(pageQs);
    requestAnimationFrame(() => window.scrollTo({ top:0, behavior:'smooth' }));




  }
}



personaInput.addEventListener('input', ()=>{
  currentPersona = personaInput.value.trim();
  btnPlay.disabled = (currentPersona === "");

  if (currentPersona === "") {
    // 高亮：黄色边框
    personaInput.style.border = "2px solid #facc15";
    personaInput.style.boxShadow = "0 0 4px rgba(250,204,21,0.6)";
  } else {
    // 恢复正常样式
    personaInput.style.border = "1px solid #e5e7eb";
    personaInput.style.boxShadow = "none";
  }
});



function sanitizeState(state) {
  if (!state) return state;
  // 深拷贝，避免改动原对象（渲染还需要 map）
  const copy = JSON.parse(JSON.stringify(state));
  delete copy.layout;   // ← 新增
  delete copy.pomap;    // ← 新增
  delete copy.map;      // 如不想去掉 map，就注释掉这一行
  return copy;
}



function deepStrip(obj, banned = new Set(['layout','pomap','map'])) {
  if (Array.isArray(obj)) return obj.map(x => deepStrip(x, banned));
  if (obj && typeof obj === 'object') {
    const out = {};
    for (const [k, v] of Object.entries(obj)) {
      if (banned.has(k)) continue;
      out[k] = deepStrip(v, banned);
    }
    return out;
  }
  return obj;
}


/***********************
 * Download full log
 ***********************/
// function downloadLog() {
//   const blob = new Blob([JSON.stringify(logData, null, 2)], {type: 'application/json'});
//   const url = URL.createObjectURL(blob);
//   const a = document.createElement('a');
//   a.href = url;
//   a.download = `userstudy_log_${Date.now()}.json`;
//   a.click();
//   URL.revokeObjectURL(url);
// }
// // 让全局可访问（End 页按钮直接调用）
// window.downloadLog = downloadLog;


// ===== 提交日志并获取完成码 =====
const btnFinish = document.getElementById('btnFinish');

async function finishAndGetCode() {
  try {
    // 你也可以在这里补充客户端时间戳/版本等
    // const payload = { log: logData };
    // 最后兜底：把 layout/pomap/map 从整个日志结构里剔除
    // const cleanedLog = deepStrip(logData, new Set(['layout','pomap','map']));
    const cleanedLog = compactLog(logData);        // ← 用精简版
    const payload = { log: cleanedLog };

    const res = await postJSON(currentServer + '/submit_log', payload);

    if (!res || !res.success || !res.completion_code) {
      throw new Error(res && res.error ? res.error : 'No completion code returned');
    }

    const code = res.completion_code;
    const hintEl = document.getElementById('completionHint');
    const codeEl = document.getElementById('completionCode');

    if (hintEl) hintEl.textContent = "Submission successful. Please copy your completion code:";
    if (codeEl) codeEl.textContent = code;

    // 防止重复提交
    if (btnFinish) {
      btnFinish.disabled = true;
      btnFinish.textContent = "Submitted";
    }

    // 保险弹窗一份
    alert(`Success! Your completion code is: ${code}`);

    // （可选）仍然本地备份一份日志供你调试
    // const blob = new Blob([JSON.stringify(logData, null, 2)], {type: 'application/json'});
    // const url = URL.createObjectURL(blob);
    // const a = document.createElement('a');
    // a.href = url;
    // a.download = `userstudy_log_${Date.now()}.json`;
    // a.click();
    // URL.revokeObjectURL(url);

  } catch (e) {
    console.error(e);
    alert('Upload failed: ' + e.message);
  }
}

// 事件绑定
btnFinish?.addEventListener('click', finishAndGetCode);



/***********************
 * Page bindings
 ***********************/
btnToConsent?.addEventListener('click', ()=>{
  logData.prolificId = (prolificId?.value || "").trim();
  logData.age = age?.value || "";
  logData.gender = gender?.value || "";
  show(pageConsent);
});

btnToInstruction?.addEventListener('click', ()=>{
  show(pageInstruction1);
});

btnToInstruction2?.addEventListener('click', ()=>{
  // 兜底：确保达标
  if (practiceCumulativeReward < PRACTICE_PASS_SCORE) {
    alert('Practice score not enough. Finish more lettuce salad to proceed.');
    return;
  }
  // 离开练习页，关闭 practice 路由
  isPracticePlaying = false;
  practiceGameOver = true;

  // 跳到第二页（Study Instruction + attention check）
  show(pageInstruction2);
});



// Persona input toggles Play
if (personaInput) {
  const checkPersona = () => {
    currentPersona = (personaInput.value || "").trim();
    btnPlay.disabled = currentPersona.length === 0 || isPlaying;
  };
  personaInput.addEventListener('input', checkPersona);
  setTimeout(checkPersona, 0);
}


btnStartTask1?.addEventListener('click', async ()=> {
  evaluateAttention();
  if (!attentionPassed) {
    alert('Please answer the attention check correctly.');
    return;
  }

  // ✅ 分配三个 task 的 robot skins
  assignRobotSkins();

  // ✅ 分配三个 task 的 layout+model（关键！否则 config_id 为空）
  assignConfigsForParticipant();


  // 更新预览页 Task1 的机器人皮肤
  if (pretask1RobotImg && robotSkinTask1) {
    pretask1RobotImg.src = "static/images/" + robotSkinTask1;
  }

  show(pagePretask1);
  await startPretaskRound(1);
});





btnEnterTask1?.addEventListener('click', ()=>{
  // 关闭预览键盘路由
  isPretask1Playing = false;

  // 初始化 Task 1 页面 UI（你原有的“进入任务”初始化逻辑）
  currentTask = 1;
  currentRound = 1;
  currentPersona = "";
  if (personaInput) { personaInput.value = ""; }

  // Task 1：显示输入框，隐藏选择框
  if (personaInputWrap)  personaInputWrap.classList.remove('hidden');
  if (personaSelectWrap) personaSelectWrap.classList.add('hidden');
  setPersonaHighlight(true);


  ({layoutId: currentLayoutId, modelId: currentModelId} = parseConfigId(currentConfigId));

  updateHeader();
  btnPlay.disabled = true;
  btnRestart.disabled = true;
  btnNext.disabled = true;
  stepsLeftSpan.textContent = '—';

  // 清空画布
  clearMainCanvas();


  // legend 里的机器人图
  const robotIcon = document.getElementById('robotIcon');
  if (robotIcon && robotSkinTask1) {
    robotIcon.src = "static/images/" + robotSkinTask1;
  }

  show(pageTask);
  requestAnimationFrame(()=> window.scrollTo({top:0, behavior:'smooth'}));
});



btnPlay?.addEventListener('click', startRound);

btnRestart?.addEventListener('click', ()=>{
  if (!isPlaying && !gameOver) return;
  startRound();
});

btnNext?.addEventListener('click', endRoundAndNext);



qsNext.addEventListener('click', ()=>{
  const questionnaire = {};
  const questions = ["q_understandability","q_understandability2","q_adaptivity","q_willingness","q_satisfaction","q_agenttrust","q_trust","q_attention"];
  for (const q of questions) {
    const val = [...document.getElementsByName(q)].find(r=>r.checked)?.value;
    if (!val) { alert("Please answer all questions before continuing."); return; }
    questionnaire[q] = parseInt(val, 10);
  }
  if (currentTask === 1) {
    logData.questionnaires.task1 = questionnaire;
    // 显示 “去 Task 2” 的提示
    document.getElementById('postTask1Hint')?.classList.remove('hidden');
  } else if (currentTask === 2) {
    logData.questionnaires.task2 = questionnaire;
    // 显示 “去 Task 3” 的提示
    document.getElementById('postTask2Hint')?.classList.remove('hidden');
  } else if (currentTask === 3) {
    logData.questionnaires.task3 = questionnaire;
    // 显示 “去 Task 4” 的提示
    document.getElementById('postTask3Hint')?.classList.remove('hidden');
  } else {
    logData.questionnaires.task4 = questionnaire;
    show(pageEnd);
    console.log("LOG DATA:", logData);
  }
});



function initTask2PersonasFromHistory() {
  const uniq = Array.from(new Set(personaHistoryTask1.filter(p => p && p.trim())));
  personaOptionsTask2 = uniq.slice(0,4);
  personaUsedTask2.clear();
}
function initTask3PersonasFromHistory() {
  const uniq = Array.from(new Set(personaHistoryTask1.filter(p => p && p.trim())));
  personaOptionsTask3 = uniq.slice(0,4);
  personaUsedTask3.clear();
}
function initTask4PersonasFromHistory() {
  const uniq = Array.from(new Set(personaHistoryTask1.filter(p => p && p.trim())));
  personaOptionsTask4 = uniq.slice(0,4);
  personaUsedTask4.clear();
}


function renderPersonaSelectOptions() {
  if (!personaSelect) return;
  personaSelect.innerHTML = `<option value="" selected disabled>— Select a persona —</option>`;
  // const opts = (currentTask === 2) ? personaOptionsTask2 : personaOptionsTask3;
  // const used = (currentTask === 2) ? personaUsedTask2 : personaUsedTask3;
  const opts =
    (currentTask === 2) ? personaOptionsTask2 :
    (currentTask === 3) ? personaOptionsTask3 :
    personaOptionsTask4;

  const used =
    (currentTask === 2) ? personaUsedTask2 :
    (currentTask === 3) ? personaUsedTask3 :
    personaUsedTask4;


  opts.forEach((p, idx) => {
    if (!used.has(idx)) {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = p;
      personaSelect.appendChild(opt);
    }
  });
}

// 每次进入 Task 2 页面时切换 UI：隐藏输入框，显示下拉选择
function switchToTask2PersonaUI() {
  if (personaInputWrap)   personaInputWrap.classList.add('hidden');
  if (personaSelectWrap)  personaSelectWrap.classList.remove('hidden');

  // 禁用 Play 直到选择
  if (btnPlay) btnPlay.disabled = true;

  // 准备选项（第一次进入或每一轮开始时都可调用）
  renderPersonaSelectOptions();
}

function switchToSelectPersonaUIForTask(taskNo){
  if (personaInputWrap)  personaInputWrap.classList.add('hidden');
  if (personaSelectWrap) personaSelectWrap.classList.remove('hidden');
  if (btnPlay) btnPlay.disabled = true;

  if (taskNo === 2) initTask2PersonasFromHistory();
  if (taskNo === 3) initTask3PersonasFromHistory();
  if (taskNo === 4) initTask4PersonasFromHistory();
  renderPersonaSelectOptions();
}


const btnProceedTask2 = document.getElementById('btnProceedTask2');
if (btnProceedTask2) {
  btnProceedTask2.addEventListener('click', async ()=>{
    // 隐藏提示卡片（你之前加的）
    const hintEl = document.getElementById('postTask1Hint');
    if (hintEl) hintEl.classList.add('hidden');

    // 清空单选框
    const questions = ["q_understandability","q_understandability2","q_adaptivity","q_willingness","q_satisfaction","q_agenttrust","q_trust","q_attention"];
    
    questions.forEach(q=>{
      [...document.getElementsByName(q)].forEach(r=>r.checked=false);
    });

    
    // 设置 Task 2 的 config（你之前在切换 Task2 时代码里做的）
    currentTask = 2;
    currentRound = 1;
    currentPersona = "";
    if (personaInput) personaInput.value = "";
    currentConfigId = participantAssignment.task2.configId;
    ({layoutId: currentLayoutId, modelId: currentModelId} = parseConfigId(currentConfigId));
    // 机器人图片
    if (pretask2RobotImg && robotSkinTask2) {
      pretask2RobotImg.src = "static/images/" + robotSkinTask2;
    }

    // 进入预览页，加载真实 Task 2 config
    show(pagePretask2);
    await startPretaskRound(2);
  });
}




const btnProceedTask3 = document.getElementById('btnProceedTask3');
if (btnProceedTask3) {
  btnProceedTask3.addEventListener('click', async ()=>{
    document.getElementById('postTask2Hint')?.classList.add('hidden');

    // 清空问卷选择
    const qs = ["q_understandability","q_understandability2","q_adaptivity","q_willingness","q_satisfaction","q_agenttrust","q_trust","q_attention"];
    qs.forEach(q=>{ [...document.getElementsByName(q)].forEach(r=>r.checked=false); });

    currentTask = 3;
    currentRound = 1;
    currentPersona = "";
    if (personaInput) personaInput.value = "";

    currentConfigId = participantAssignment.task3.configId;
    ({layoutId: currentLayoutId, modelId: currentModelId} = parseConfigId(currentConfigId));


    if (pretask3RobotImg && robotSkinTask3) {
      pretask3RobotImg.src = "static/images/" + robotSkinTask3;
    }

    show(pagePretask3);
    await startPretaskRound(3);
  });
}




const btnProceedTask4 = document.getElementById('btnProceedTask4');
if (btnProceedTask4) {
  btnProceedTask4.addEventListener('click', async ()=>{
    document.getElementById('postTask3Hint')?.classList.add('hidden');

    // 清空问卷选择
    const qs = ["q_understandability","q_understandability2","q_adaptivity","q_willingness","q_satisfaction","q_agenttrust","q_trust","q_attention"];
    qs.forEach(q=>{ [...document.getElementsByName(q)].forEach(r=>r.checked=false); });

    currentTask = 4;
    currentRound = 1;
    currentPersona = "";
    if (personaInput) personaInput.value = "";

    currentConfigId = participantAssignment.task4.configId;
    ({layoutId: currentLayoutId, modelId: currentModelId} = parseConfigId(currentConfigId));


    if (pretask4RobotImg && robotSkinTask4) {
      pretask4RobotImg.src = "static/images/" + robotSkinTask4;
    }

    show(pagePretask4);
    await startPretaskRound(4);
  });
}



personaSelect?.addEventListener('change', () => {
  const idx = personaSelect.value;
  if (!idx) { btnPlay.disabled = true; return; }
  // const opts = (currentTask === 2) ? personaOptionsTask2 : personaOptionsTask3;
  const opts =
    (currentTask === 2) ? personaOptionsTask2 :
    (currentTask === 3) ? personaOptionsTask3 :
    personaOptionsTask4;

  currentPersona = opts[parseInt(idx,10)] || "";
  btnPlay.disabled = currentPersona.trim().length === 0 || isPlaying;
});



btnEnterTask2?.addEventListener('click', ()=>{
  isPretask2Playing = false;

  // 初始化 Task 2 页面 UI（你原有的 Task 2 初始化逻辑）
  // setPersonaHighlight(true); // 提醒先填 persona

  // 切换到 Task2 persona 选择模式
  initTask2PersonasFromHistory();
  // 切到 Task2 persona 选择模式（从 Task1 personas 中选择）
  switchToSelectPersonaUIForTask(2);


  updateHeader();
  btnPlay.disabled = true;
  btnRestart.disabled = true;
  btnNext.disabled = true;
  stepsLeftSpan.textContent = '—';

  // 清空画布
  clearMainCanvas();

  // legend 里更新机器人
  const robotIcon2 = document.getElementById('robotIcon');
  if (robotIcon2 && robotSkinTask2) {
    robotIcon2.src = "static/images/" + robotSkinTask2;
  }

  show(pageTask);
  requestAnimationFrame(()=> window.scrollTo({top:0, behavior:'smooth'}));
});


btnEnterTask3?.addEventListener('click', ()=>{
  isPretask3Playing = false;

  currentTask = 3;
  currentRound = 1;
  currentPersona = "";

  switchToSelectPersonaUIForTask(3);

  updateHeader();
  btnPlay.disabled = true;
  btnRestart.disabled = true;
  btnNext.disabled = true;
  stepsLeftSpan.textContent = '—';
  clearMainCanvas();

  // legend 里更新机器人
  const robotIcon3 = document.getElementById('robotIcon');
  if (robotIcon3 && robotSkinTask3) {
    robotIcon3.src = "static/images/" + robotSkinTask3;
  }

  show(pageTask);
  requestAnimationFrame(()=> window.scrollTo({top:0, behavior:'smooth'}));
});



btnEnterTask4?.addEventListener('click', ()=>{
  isPretask4Playing = false;

  currentTask = 4;
  currentRound = 1;
  currentPersona = "";

  switchToSelectPersonaUIForTask(4);

  updateHeader();
  btnPlay.disabled = true;
  btnRestart.disabled = true;
  btnNext.disabled = true;
  stepsLeftSpan.textContent = '—';
  clearMainCanvas();

  // legend 里更新机器人
  const robotIcon4 = document.getElementById('robotIcon');
  if (robotIcon4 && robotSkinTask4) {
    robotIcon4.src = "static/images/" + robotSkinTask4;
  }

  show(pageTask);
  requestAnimationFrame(()=> window.scrollTo({top:0, behavior:'smooth'}));
});



/***********************
 * Practice gating UI
 ***********************/
// === Attention check refs & state ===
const attentionCheckRadios = document.getElementsByName('attentionCheck');
const attentionCheckHint = document.getElementById('attentionCheckHint');
let attentionPassed = false; // 是否答对 attention check（选 B: trustworthiness）

// 练习达标阈值你已有：PRACTICE_PASS_SCORE
// practiceCumulativeReward 你已有

// 计算 attention 是否通过
function evaluateAttention() {
  const selected = [...attentionCheckRadios].find(r => r.checked)?.value;
  attentionPassed = (selected === 'trustworthiness'); // 正确答案
  if (attentionCheckHint) {
    attentionCheckHint.style.display = attentionPassed ? 'none' : (selected ? 'block' : 'none');
  }
  // 控制 Start Task 1
  if (btnStartTask1) btnStartTask1.disabled = !attentionPassed;
}


personaFidelityRadios.forEach(r => {
  r.addEventListener('change', () => {
    currentPersonaFidelity = parseInt(r.value, 10);
    btnNext.disabled = isNaN(currentPersonaFidelity);
  });
});




// 统一更新 Start Task 1 的门禁（分数 + attention 都要满足）
function updateStartTaskGate() {
  // 先计算 attention
  const pass = practiceCumulativeReward >= PRACTICE_PASS_SCORE;
  // 控制“Proceed to Study Instruction”按钮
  if (btnToInstruction2) btnToInstruction2.disabled = !pass;

  // 提示文本
  const hint = document.getElementById('practiceHint');
  if (hint) {
    hint.textContent = pass
      ? "Great! You can proceed to the Study Instruction."
      : "Practice score not enough. Finish more lettuce salad to proceed.";
  }
}

// 监听 attention 选项变化
// attentionCheckRadios.forEach(r => r.addEventListener('change', updateStartTaskGate));

attentionCheckRadios.forEach(r => r.addEventListener('change', evaluateAttention));



/***********************
 * Boot
 ***********************/
(async function boot(){
  try {
    await ensureSession();                // ⬅️ 先拿会话
  } catch (e) {
    console.error(e);
    alert('Failed to initialize session: ' + e.message);
    return;
  }
  updateStartTaskGate();
  preloadImages(()=>{ /* ready */ });
  show(pageIntro);
})();

