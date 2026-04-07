// static/js/practice.js

// --- PRACTICE ROUND SETUP ONLY ---

async function practiceApi(endpoint, data = {}) {
    let practiceSessionId = sessionStorage.getItem('practice_session_id');

    if (!practiceSessionId) {
        const res = await fetch(`${SERVER_URL}/new_session`, { method: 'POST' });
        const d = await res.json();
        practiceSessionId = d.session_id;
        sessionStorage.setItem('practice_session_id', practiceSessionId);
    }

    const res = await fetch(`${SERVER_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...data, session_id: practiceSessionId })
    });

    return await res.json();
}

function clearPracticeSession() {
    sessionStorage.removeItem('practice_session_id');
}

async function startPracticeRound() {
    console.log("Starting practice round...");
    
    // 1. Force the global state to Practice Mode
    STATE.phase = 0; 
    STATE.configId = 'layout_practice';
    STATE.practiceDishes = 0;
    STATE.isPlaying = false;
    STATE.gameOver = false;

    try {
        // 2. Tell the server to reset for practice
        console.log("Calling /reset with config_id: layout_practice");
        const data = await practiceApi('/reset', { 
            config_id: 'layout_practice', 
            prolificId: STATE.prolificId             
         });
        
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
            focusGameSurface();

            // 5. Update UI
            document.getElementById('practiceHint').innerText = "Deliver 1 complete dish to proceed.";
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