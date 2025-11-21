document.addEventListener("DOMContentLoaded", function () {
  const toggle = document.getElementById("darkToggle");
  toggle.addEventListener("click", () => {
    document.body.classList.toggle("dark");
  });

  // floating mascot
  const mascot = document.createElement("div");
  mascot.innerHTML = "🤖 EcoBot";
  mascot.style.position = "fixed";
  mascot.style.bottom = "100px";
  mascot.style.right = "40px";
  mascot.style.background = "#4caf50";
  mascot.style.color = "white";
  mascot.style.padding = "10px 20px";
  mascot.style.borderRadius = "20px";
  mascot.style.boxShadow = "0 4px 10px rgba(0,0,0,0.2)";
  mascot.style.animation = "float 3s ease-in-out infinite";
  document.body.appendChild(mascot);
  
 // 🌟 EcoBot Celebration with Random Messages
document.addEventListener('wasteReported', () => {
  const messages = [
    "Woohoo! You’re saving the planet 🌍💪",
    "Green points unlocked! ♻️",
    "Mother Earth is smiling right now 🌱",
    "You rock! Thanks for keeping things clean 🌿",
    "EcoBot approves this report ✅",
    "That’s another win for Team Earth 🌎👏",
    "Clean streets, happy hearts 💚"
  ];

  // Pick a random message
  const randomMessage = messages[Math.floor(Math.random() * messages.length)];

  // Make EcoBot happy
  ecoBot.classList.add('celebrate');
  ecoBot.src = '/static/img/ecobot_happy.png';
  showEcoTip(randomMessage);

  // After 2.5s, return EcoBot to normal
  setTimeout(() => {
    ecoBot.classList.remove('celebrate');
    ecoBot.src = '/static/img/ecobot.png';
  }, 2500);
});



@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

// 🌿 EcoBot Mascot with Personality
const ecoTips = [
  "Reduce, Reuse, Recycle ♻️",
  "Small actions = Big change!",
  "Keep Kenya clean 🇰🇪",
  "Every report makes Earth happier 🌍",
  "Together, we can make a green future 💚"
];

// Create EcoBot
const ecoBot = document.createElement('img');
ecoBot.src = '/static/img/ecobot.png';
ecoBot.id = 'ecoBot';
document.body.appendChild(ecoBot);

let ecoBubble;

// Function to show eco tips
function showEcoTip(text) {
  if (ecoBubble) ecoBubble.remove();
  ecoBubble = document.createElement('div');
  ecoBubble.className = 'ecoBubble';
  ecoBubble.textContent = text;
  document.body.appendChild(ecoBubble);
  setTimeout(() => ecoBubble.remove(), 4000);
}

// Random eco tips
setInterval(() => {
  showEcoTip(ecoTips[Math.floor(Math.random() * ecoTips.length)]);
  blinkEcoBot();
}, 12000);

// On click — happy reaction
ecoBot.addEventListener('click', () => {
  ecoBot.classList.add('happyPulse');
  ecoBot.src = '/static/img/ecobot_happy.png';
  showEcoTip("Yay! You just helped the planet 🌿💪");
  setTimeout(() => {
    ecoBot.src = '/static/img/ecobot.png';
    ecoBot.classList.remove('happyPulse');
  }, 1000);
});

// Simulate blinking every few seconds (if you only have one image)
function blinkEcoBot() {
  ecoBot.style.filter = 'brightness(0.7)';
  setTimeout(() => ecoBot.style.filter = 'brightness(1)', 200);
}

function showEcoTip(message) {
  const tip = document.createElement('div');
  tip.textContent = message;
  tip.style.position = 'fixed';
  tip.style.bottom = '100px';
  tip.style.left = '50%';
  tip.style.transform = 'translateX(-50%)';
  tip.style.background = '#00e676';
  tip.style.color = '#fff';
  tip.style.padding = '10px 20px';
  tip.style.borderRadius = '20px';
  tip.style.fontWeight = 'bold';
  tip.style.boxShadow = '0 0 15px rgba(0,0,0,0.2)';
  tip.style.zIndex = '9999';
  tip.style.transition = 'opacity 0.5s';
  document.body.appendChild(tip);

  setTimeout(() => tip.style.opacity = '0', 2000);
  setTimeout(() => tip.remove(), 2500);
}


// Dark Mode Toggle
document.addEventListener("DOMContentLoaded", function () {
  const toggle = document.getElementById("darkModeToggle");
  const body = document.body;

  // Check saved mode in localStorage
  if (localStorage.getItem("theme") === "dark") {
    body.classList.add("dark-mode");
    if (toggle) toggle.textContent = "☀️ Light Mode";
  }

  if (toggle) {
    toggle.addEventListener("click", function () {
      body.classList.toggle("dark-mode");

      if (body.classList.contains("dark-mode")) {
        localStorage.setItem("theme", "dark");
        toggle.textContent = "☀️ Light Mode";
      } else {
        localStorage.setItem("theme", "light");
        toggle.textContent = "🌙 Dark Mode";
      }
    });
  }
});
