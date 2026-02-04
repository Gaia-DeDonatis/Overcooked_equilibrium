// static/js/practice.js

// --- PRACTICE ROUND LOGIC ---

// 1. Start Practice Game
async function startPracticeRound() {
    STATE.configId = 'layout_practice';
    STATE.phase = 0; 
    STATE.practiceScore = 0;
    STATE.isPlaying = false;
    STATE.gameOver = false;

    try {
        const data = await api('/reset', { config_id: 'layout_practice' });
        
        if (data.state) {
            // ✅ Enter Practice Mode
            STATE.isPlaying = true;
            
            // Draw First Frame
            drawGame(data.state, 'gameCanvas_practice');

            // Set Score
            STATE.practiceScore = data.cumulative_reward || 0;
            
            // Reset UI
            document.getElementById('practiceHint').innerText = "Deliver 1 Salad (200 pts) to proceed.";
            document.getElementById('to-instruction-2').disabled = true;
        }
    } catch (err) {
        console.error(err);
        alert('Practice reset failed: ' + err.message);
    }
}

// Bind Button
document.getElementById('btnTryPractice').onclick = startPracticeRound;


// 2. Practice Input Handler
document.addEventListener('keydown', async (e) => {
    // A. Filters
    if(['ArrowUp','ArrowDown','ArrowLeft','ArrowRight'].indexOf(e.key) === -1) return;
    if(!STATE.isPlaying || STATE.gameOver) return;
    if(STATE.phase !== 0) return; // Exit if not practice
    
    e.preventDefault();

    try {
        // B. Send Key
        const data = await api('/key_event', { key: e.key, config_id: STATE.configId });
        
        // C. Draw
        drawGame(data.state, 'gameCanvas_practice');
        
        // D. Check Win Condition
        const prevScore = STATE.practiceScore;
        const currScore = data.cumulative_reward || 0;
        STATE.practiceScore = currScore;

        // WIN Logic
        if(currScore >= 200 && currScore > prevScore) {
            STATE.isPlaying = false;
            STATE.gameOver = true;
            
            document.getElementById('practiceHint').innerText = "Great job! Click 'Next' to continue.";
            document.getElementById('to-instruction-2').disabled = false;
            
            setTimeout(() => {
                alert("Practice Complete! You delivered the salad.");
            }, 100);
        }

    } catch(e) {
        console.error("Practice Input Error", e);
    }
});