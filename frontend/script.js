const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const ctx = overlay.getContext("2d");
const alarm = document.getElementById("alarm-sound");

const userText = document.getElementById("detected-user");
const systemStatus = document.getElementById("system-status");
const livenessText = document.getElementById("liveness-status");
const alertList = document.getElementById("alert-list");

let alertPlaying = false;

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
    video.onloadedmetadata = () => {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        startSendingFrames();
    }
});

// Send frames to backend every 300 ms
function startSendingFrames() {
    setInterval(async () => {

        ctx.drawImage(video, 0, 0, overlay.width, overlay.height);
        const frameData = overlay.toDataURL("image/jpeg");

        const res = await fetch("https://your-backend-url.onrender.com/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame: frameData })
        });

        const data = await res.json();
        updateUI(data);

    }, 300);
}

function updateUI(data) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // Draw bounding boxes
    data.boxes.forEach(box => {
        ctx.strokeStyle = box.color;
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x, box.y, box.w, box.h);
    });

    userText.textContent = data.user || "Unknown";
    systemStatus.textContent = data.status;
    livenessText.textContent = data.liveness ? "PASS" : "Checking...";

    alertList.innerHTML = "";
    data.alerts.forEach(alert => {
        const li = document.createElement("li");
        li.textContent = alert;
        alertList.appendChild(li);
    });

    if (data.alerts.length > 0 && !alertPlaying) {
        alarm.play();
        alertPlaying = true;
    }

    if (data.alerts.length === 0 && alertPlaying) {
        alarm.pause();
        alarm.currentTime = 0;
        alertPlaying = false;
    }
}
