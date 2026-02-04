// static/js/view.js

const images = {};
// Map tile IDs to file names
const TILE_MAP = {
    0: "space.png", 1: "counter.png", 3: "FreshTomato.png", 4: "FreshLettuce.png",
    5: "plate.png", 6: "cutboard.png", 7: "delivery.png", 8: "FreshOnion.png",
    9: "dirtyplate.png", 10: "BadLettuce.png"
};

// 1. Image Loading
function preloadImages(callback) {
    const names = [
        "space.png", "counter.png", "FreshTomato.png", "ChoppedTomato.png",
        "FreshLettuce.png", "ChoppedLettuce.png", "plate.png", "cutboard.png",
        "delivery.png", "FreshOnion.png", "ChoppedOnion.png", "dirtyplate.png",
        "BadLettuce.png", "agent-red.png", "agent-blue.png", "agent-robot.png"
    ];
    let loaded = 0;
    names.forEach(n => {
        images[n] = new Image();
        images[n].src = `static/images/${n}`;
        images[n].onload = () => { if (++loaded === names.length) callback(); };
    });
}

// 2. Name Resolver (Fixes "lettuce" vs "FreshLettuce.png")
function resolveImage(name) {
    if (!name) return null;
    const n = name.toLowerCase().replace(/_/g, "");
    
    // Explicit mappings
    if (n === 'lettuce') return images['FreshLettuce.png'];
    if (n === 'onion') return images['FreshOnion.png'];
    if (n === 'tomato') return images['FreshTomato.png'];
    
    // Chopped mappings
    if (n.includes('choppedlettuce')) return images['ChoppedLettuce.png'];
    if (n.includes('choppedonion')) return images['ChoppedOnion.png'];
    if (n.includes('choppedtomato')) return images['ChoppedTomato.png'];

    // Fallback
    for (let key in images) {
        if (key.toLowerCase().replace(".png","") === n) return images[key];
    }
    return null;
}

// 3. Main Draw Function (Adapted from YOUR Original Code)
function drawGame(state, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !state || !state.map) return;
    const ctx = canvas.getContext('2d');

    // === Helper: Draw Plate with Content Scaled ===
    function drawPlateWithContent(x, y, w, h, plateName, contentName) {
        const plateImg = resolveImage(plateName);
        if (plateImg) ctx.drawImage(plateImg, x, y, w, h);

        if (contentName) {
            const contentImg = resolveImage(contentName);
            if (contentImg) {
                const CONTENT_SCALE = 0.65; // Scale down food on plate
                const cw = w * CONTENT_SCALE;
                const ch = h * CONTENT_SCALE;
                const cx = x + (w - cw) / 2;
                const cy = y + (h - ch) / 2;
                ctx.drawImage(contentImg, cx, cy, cw, ch);
            }
        }
    }

    // === Cleanup & Geometry ===
    const cw = canvas.width;
    const ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);

    const cell = Math.floor(Math.min(cw / state.ylen, ch / state.xlen));
    const drawW = cell * state.ylen;
    const drawH = cell * state.xlen;
    const offX = Math.floor((cw - drawW) / 2);
    const offY = Math.floor((ch - drawH) / 2);

    // === Map ===
    for (let x = 0; x < state.xlen; x++) {
        for (let y = 0; y < state.ylen; y++) {
            const tileName = TILE_MAP[state.map[x][y]] || "space.png";
            const img = images[tileName];
            if (img) ctx.drawImage(img, offX + y*cell, offY + x*cell, cell, cell);
        }
    }

    // === Items on Counter ===
    const holdingCells = new Set((state.agents||[]).map(a => `${a.x},${a.y}`));
    (state.items||[]).forEach(item => {
        const key = `${item.x},${item.y}`;
        if (holdingCells.has(key)) return;

        const baseX = offX + item.y * cell;
        const baseY = offY + item.x * cell;

        // 1. Draw Counter Background (Fixes transparency)
        if (images["counter.png"]) ctx.drawImage(images["counter.png"], baseX, baseY, cell, cell);

        // 2. Draw Item
        if (item.type === "plate" || item.type === "dirtyplate") {
            drawPlateWithContent(baseX, baseY, cell, cell, item.type, item.containing || null);
        } else {
            // Other items (Cutting Board, Pot, Ingredients)
            const baseImg = resolveImage(item.type);
            if (baseImg) ctx.drawImage(baseImg, baseX, baseY, cell, cell);

            // Draw Content (e.g. Lettuce on Cutting Board)
            if (item.containing) {
                const containedImg = resolveImage(item.containing);
                if (containedImg) ctx.drawImage(containedImg, baseX, baseY, cell, cell);
            }
            // Draw Holding (if applicable for items)
            if (item.holding) {
                const holdingImg = resolveImage(item.holding);
                if (holdingImg) ctx.drawImage(holdingImg, baseX, baseY, cell, cell);
            }
        }
    });

    // === Agents ===
    (state.agents||[]).forEach(agent => {
        // Agent Image
        let agentImg = null;
        if (agent.color === "robot") {
            // Using your skin logic logic
            agentImg = images["agent-robot.png"]; 
        } else {
            agentImg = images[`agent-${agent.color}.png`];
        }
        
        if (agentImg) {
            ctx.drawImage(agentImg, offX + agent.y*cell, offY + agent.x*cell, cell, cell);
        }

        // Held Item
        if (agent.holding) {
            const holdName = agent.holding;
            const holdImg = resolveImage(holdName);
            
            if (holdImg) {
                // Position: Bottom Right
                const SCALE_IN_HAND = 0.5;
                const w = cell * SCALE_IN_HAND;
                const h = cell * SCALE_IN_HAND;
                const x = offX + agent.y*cell + (cell - w);
                const y = offY + agent.x*cell + (cell - h);

                if (holdName === "plate" || holdName === "dirtyplate") {
                    // Logic for plate in hand (uses helper)
                    // Note: Ensure your backend sends 'holding_containing', otherwise this is null
                    const contentName = agent.holding_containing || null; 
                    drawPlateWithContent(x, y, w, h, holdName, contentName);
                } else {
                    // Normal item in hand
                    ctx.drawImage(holdImg, x, y, w, h);
                }
            }
        }
    });
}