# Pilot Test Setup Guide (ngrok + local server)

Use this every time you want to run the pilot test and let someone access your experiment from another device.

## Step 1 — Open the project in VS Code
Open your project folder:

`C:\Users\dedong1\work\Overcooked_equilibrium`

## Step 2 — Open Terminal 1
In VS Code, open a terminal.

Run:

```bash
python backend_new.py
```

Leave this terminal open.

### Check
Open this in your browser on your own computer:

`http://127.0.0.1:5000`

If the experiment page opens, the backend is running correctly.

---

## Step 3 — Open Terminal 2
Open a second terminal in VS Code.

Run:

```bash
ngrok http 5000
```

Leave this terminal open too.

### Check
You should see something like:

`Forwarding https://something.ngrok-free.dev -> http://localhost:5000`

Copy the HTTPS link.

---

## Step 4 — Test the public link yourself
Before sending it to anyone, open the ngrok HTTPS link on:
- your own browser
- your phone if possible

If it works there, the tunnel is active.

---

## Step 5 — Send the link
Send the HTTPS ngrok link to the participant.

Example:

`https://something.ngrok-free.dev`

Tell them to open it in Safari or Chrome directly.

---

## Step 6 — Important things during the pilot
Keep BOTH terminals open:
- Terminal 1: `python backend_new.py`
- Terminal 2: `ngrok http 5000`

If you close either one, the experiment stops working.

Also:
- do not let your computer sleep
- do not disconnect from the internet
- do not restart ngrok during the session
- use a unique participant ID for each person

---

## Step 7 — If the participant cannot open the page
### First check locally
Open on your computer:

`http://127.0.0.1:5000`

If this does not work, the backend crashed.

### Then check ngrok
Open on your computer:

`http://127.0.0.1:4040`

This is the ngrok inspector.

If the participant refreshes the page, you should see a request there.

### What it means
- If you see a request in `4040`, the participant reached ngrok.
- If you do not see a request, the problem is on their browser/network side.

### Ask the participant to try:
- copy the link and paste it into Safari/Chrome directly
- do not open only inside WhatsApp
- try incognito/private mode
- disable VPN
- try mobile data instead of Wi‑Fi

---

## Step 8 — Where the data is saved
The participant data is saved on your own computer inside your project folder, under the submissions folder.

So yes, if the participant completes the experiment while your backend is running, their data is collected on your machine.

---

## Step 9 — After the pilot
When the participant is done:
1. wait until submission finishes
2. stop ngrok with `Ctrl + C`
3. stop the backend with `Ctrl + C`

---

## Quick checklist
### Before sending the link
- [ ] `backend_new.py` is running
- [ ] `http://127.0.0.1:5000` works
- [ ] `ngrok http 5000` is running
- [ ] the HTTPS ngrok link works for you
- [ ] both terminals stay open

### During the test
- [ ] computer stays awake
- [ ] internet stays connected
- [ ] participant uses a unique ID

### If something fails
- [ ] check `127.0.0.1:5000`
- [ ] check `127.0.0.1:4040`
- [ ] ask participant to try another browser or mobile data

---

## One-time setup reminder
You only need to add the auth token once:

```bash
ngrok config add-authtoken YOUR_TOKEN
```

Do not share your auth token with anyone.
