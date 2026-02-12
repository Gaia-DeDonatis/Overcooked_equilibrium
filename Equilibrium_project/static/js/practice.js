// static/js/practice.js

// --- PRACTICE ROUND SETUP ONLY ---

async function startPracticeRound() {
    console.log("Starting practice round...");
    
    // 1. Force the global state to Practice Mode
    STATE.phase = 0; 
    STATE.configId = 'layout_practice';
    STATE.practiceScore = 0;
    STATE.isPlaying = false;
    STATE.gameOver = false;

    try {
        // 2. Tell the server to reset for practice
        console.log("Calling /reset with config_id: layout_practice");
        const data = await api('/reset', { config_id: 'layout_practice' });
        
        console.log("Reset response:", data);
        
        if (!data.success) {
            console.error("Backend error:", data.error);
            alert(`Practice failed to load: ${data.error || 'Unknown error'}`);
            return;
        }
        
        if (data.state) {
            // 3. Unlock the keyboard listener in controller.js
            STATE.isPlaying = true;
            
            // 4. Draw to the PRACTICE canvas
            drawGame(data.state, 'gameCanvas_practice');

            // 5. Update UI
            document.getElementById('practiceHint').innerText = "Deliver 1 Salad (200 pts) to proceed.";
            document.getElementById('to-instruction-2').disabled = true;
            
            console.log("Practice round loaded successfully!");
        } else {
            console.error("No state returned from backend");
            alert("Practice failed: No game state received");
        }
    } catch (err) {
        console.error("Practice Reset Error:", err);
        alert(`Network error: ${err.message}`);
    }
}

// Bind Button
const btnTryPractice = document.getElementById('btnTryPractice');
if(btnTryPractice) {
    btnTryPractice.onclick = startPracticeRound;
}