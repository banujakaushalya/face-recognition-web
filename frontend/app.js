// Global variables for camera streams
let mainStream = null;
let registerStream = null;
let capturedPhoto = null;
let streamingInterval = null;

// Rate limiting configuration (in milliseconds)
// Adjust this value to control frame sending speed:
// 1000ms = 1 FPS (slower, less backend load)
// 1500ms = 0.67 FPS (very slow, minimum backend load)
// 500ms = 2 FPS (original speed, more backend load)
const FRAME_SEND_INTERVAL = 1500; // Reduced to 0.67 FPS to minimize backend bombardment

// Stop all streaming and camera activity
function stopAllStreaming() {
  // Stop streaming interval
  if (streamingInterval) {
    clearInterval(streamingInterval);
    streamingInterval = null;
  }

  // Stop main camera stream
  if (mainStream) {
    mainStream.getTracks().forEach((track) => track.stop());
    mainStream = null;
  }

  // Hide video elements
  const video = document.getElementById("liveVideo");
  const placeholder = document.getElementById("video-placeholder");
  const resultBox = document.getElementById("result-box");

  if (video) video.classList.add("hidden");
  if (placeholder) placeholder.classList.remove("hidden");
  if (resultBox) resultBox.classList.add("hidden");
}

// Load content and manage section visibility
function loadContent(section) {
  // Stop all streaming first
  stopAllStreaming();

  // Hide all sections
  document.getElementById("main-content").classList.add("hidden");
  document.getElementById("register-content").classList.add("hidden");
  document.getElementById("admin-content").classList.add("hidden");

  // Show selected section and initialize if home
  switch (section) {
    case "home":
      document.getElementById("main-content").classList.remove("hidden");
      initCamera();
      break;
    case "register":
      document.getElementById("register-content").classList.remove("hidden");
      break;
    case "admin":
      document.getElementById("admin-content").classList.remove("hidden");
      break;
  }
}

// Initialize main camera for face recognition and start streaming
async function initCamera() {
  try {
    const video = document.getElementById("liveVideo");
    const placeholder = document.getElementById("video-placeholder");
    const resultBox = document.getElementById("result-box");

    // Stop existing stream if any
    if (mainStream) {
      mainStream.getTracks().forEach((track) => track.stop());
    }

    // Clear result box
    resultBox.classList.add("hidden");

    // Request camera access
    mainStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    video.srcObject = mainStream;
    video.classList.remove("hidden");
    placeholder.classList.add("hidden");

    console.log("Camera initialized successfully");

    // Start streaming frames to backend
    startStreamingToBackend();
  } catch (error) {
    console.error("Error accessing camera:", error);
    alert("Unable to access camera. Please check permissions.");
  }
}

// Stream video frames to backend continuously
function startStreamingToBackend() {
  const video = document.getElementById("liveVideo");
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");

  // Clear any existing interval
  if (streamingInterval) {
    clearInterval(streamingInterval);
  }

  // Stream frames to backend at configured rate
  streamingInterval = setInterval(() => {
    if (!video.srcObject) return;

    // Set canvas dimensions to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    ctx.drawImage(video, 0, 0);

    // Convert canvas to blob and send to backend
    canvas.toBlob(
      (blob) => {
        sendLiveFrameToBackend(blob);
      },
      "image/jpeg",
      0.85,
    );
  }, FRAME_SEND_INTERVAL);
}

// Send live frame to backend
async function sendLiveFrameToBackend(blob) {
  try {
    const formData = new FormData();
    formData.append("frame", blob, "frame.jpg");

    const response = await fetch("http://localhost:8000/recognize-face", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();
    console.log("Recognition result:", result);

    // Display result if face is recognized or rejected
    if (
      result.status === "approved" ||
      result.status === "rejected" ||
      result.status === "no_face"
    ) {
      displayResult(result);
    }
  } catch (error) {
    console.error("Error sending frame to backend:", error);
  }
}

// Display result in the result box
function displayResult(result) {
  const resultBox = document.getElementById("result-box");
  const resultStatus = document.getElementById("result-status");
  const resultText = document.getElementById("result-text");
  const resultDetails = document.getElementById("result-details");

  // STOP STREAMING IMMEDIATELY when result is received (approved, rejected, or no_face)
  // User must click "Reset" button to restart streaming
  if (streamingInterval) {
    clearInterval(streamingInterval);
    streamingInterval = null;
    console.log("Streaming stopped - waiting for reset button");
  }

  // Clear previous status classes
  resultStatus.className = "mb-4";

  if (result.status === "approved") {
    resultStatus.classList.add("text-green-600");
    resultText.textContent = "✓ Access Approved";
    resultDetails.textContent = `Welcome, ${result.name || "User"}!`;
  } else if (result.status === "rejected") {
    resultStatus.classList.add("text-red-600");
    resultText.textContent = "✗ Access Denied";
    resultDetails.textContent =
      result.message || "Face not recognized in database";
  } else if (result.status === "no_face") {
    resultStatus.classList.add("text-yellow-600");
    resultText.textContent = "⚠ No Face Detected";
    resultDetails.textContent = "Please position your face in front of camera";
  }

  resultBox.classList.remove("hidden");
}

// Reset main camera and clear results
// This restarts the streaming process after a result is shown
function resetCamera() {
  const video = document.getElementById("liveVideo");
  const placeholder = document.getElementById("video-placeholder");
  const resultBox = document.getElementById("result-box");

  // Stop streaming
  if (streamingInterval) {
    clearInterval(streamingInterval);
    streamingInterval = null;
  }

  if (mainStream) {
    mainStream.getTracks().forEach((track) => track.stop());
    mainStream = null;
  }

  video.classList.add("hidden");
  placeholder.classList.remove("hidden");
  resultBox.classList.add("hidden");

  console.log("Camera reset - restarting streaming");

  // Reinitialize camera and RESTART STREAMING
  initCamera();
}

// Initialize camera for registration
async function initRegisterCamera() {
  try {
    const video = document.getElementById("regVideo");
    const placeholder = document.getElementById("photo-placeholder");
    const controls = document.getElementById("camera-controls");

    // Stop existing stream if any
    if (registerStream) {
      registerStream.getTracks().forEach((track) => track.stop());
    }

    // Request camera access
    registerStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });

    video.srcObject = registerStream;
    video.classList.remove("hidden");
    placeholder.classList.add("hidden");
    controls.classList.remove("hidden");

    console.log("Registration camera initialized");
  } catch (error) {
    console.error("Error accessing camera:", error);
    alert("Unable to access camera. Please check permissions.");
  }
}

// Capture photo from registration camera
function captureRegisterPhoto() {
  const video = document.getElementById("regVideo");
  const canvas = document.getElementById("photoCanvas");

  if (!video.srcObject) {
    alert("Camera is not initialized.");
    return;
  }

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  // Convert canvas to image data
  capturedPhoto = canvas.toDataURL("image/jpeg");
  canvas.classList.remove("hidden");

  console.log("Photo captured for registration");
}

// Stop registration camera
function stopRegisterCamera() {
  const video = document.getElementById("regVideo");
  const canvas = document.getElementById("photoCanvas");
  const placeholder = document.getElementById("photo-placeholder");
  const controls = document.getElementById("camera-controls");

  if (registerStream) {
    registerStream.getTracks().forEach((track) => track.stop());
    registerStream = null;
  }

  video.classList.add("hidden");
  canvas.classList.add("hidden");
  placeholder.classList.remove("hidden");
  controls.classList.add("hidden");

  console.log("Registration camera stopped");
}

// Handle registration form submission
document.addEventListener("DOMContentLoaded", function () {
  const registerForm = document.getElementById("registerForm");

  if (registerForm) {
    registerForm.addEventListener("submit", async function (e) {
      e.preventDefault();

      if (!capturedPhoto) {
        alert("Please capture a photo first");
        return;
      }

      // Get form data
      const data = {
        fullName: document.getElementById("username").value,
        email: document.getElementById("email").value,
        address1: document.getElementById("address1").value,
        address2: document.getElementById("address2").value,
        phone: document.getElementById("phone").value,
        photo: capturedPhoto,
      };

      try {
        const response = await fetch("http://localhost:8000/register", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
          alert("Registration successful!");
          registerForm.reset();
          capturedPhoto = null;
          stopRegisterCamera();
          document.getElementById("photoCanvas").classList.add("hidden");
        } else {
          alert(`Error: ${result.message}`);
        }
      } catch (error) {
        console.error("Error during registration:", error);
        alert("Error during registration. Please try again.");
      }
    });
  }

  // Show home section by default
  loadContent("home");
});
