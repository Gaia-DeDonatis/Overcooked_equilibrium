// static/js/practice.js

// --- PRACTICE ROUND SETUP ONLY ---

async function startPracticeRound() {
    // 1. Force the global state to Practice Mode
    STATE.phase = 0; 
    STATE.configId = 'layout_practice';
    STATE.practiceScore = 0;
    STATE.isPlaying = false;
    STATE.gameOver = false;

    try {
        // 2. Tell the server to reset for practice
        const data = await api('/reset', { config_id: 'layout_practice' });
        
        if (data.state) {
            // 3. Unlock the keyboard listener in controller.js
            STATE.isPlaying = true;
            
            // 4. Draw to the PRACTICE canvas
            drawGame(data.state, 'gameCanvas_practice');

            // 5. Update UI
            document.getElementById('practiceHint').innerText = "Deliver 1 Salad (200 pts) to proceed.";
            document.getElementById('to-instruction-2').disabled = true;
        }
    } catch (err) {
        console.error("Practice Reset Error:", err);
    }
}

// Bind Button
document.getElementById('btnTryPractice').onclick = startPracticeRound;