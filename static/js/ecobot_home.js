// 🌍 Floating EcoBot Messages
const ecoBot = document.getElementById('ecoBotHome');
const messageBox = document.getElementById('ecoBotMessage');

const ecoFacts = [
  "Did you know? Recycling one aluminum can saves enough energy to run a TV for 3 hours! 📺♻️",
  "Plastic bottles take up to 450 years to decompose! 🚯",
  "You can save 8,000 liters of water a year just by turning off the tap while brushing! 🚿",
  "Trees absorb CO₂ — plant one today! 🌳",
  "Every small act counts toward a cleaner planet 💚",
  "Keep calm and recycle on ♻️",
  "EcoBot loves clean spaces — you rock! 🌱"
];

// Show a random message
function showRandomMessage() {
  const randomFact = ecoFacts[Math.floor(Math.random() * ecoFacts.length)];
  messageBox.textContent = randomFact;
  messageBox.style.opacity = '1';
  setTimeout(() => messageBox.style.opacity = '0', 5000);
}

// Show one message every 30 seconds
setInterval(showRandomMessage, 30000);

// Show a greeting when page loads
setTimeout(showRandomMessage, 2000);

// Bonus: When clicked, EcoBot waves excitedly
ecoBot.addEventListener('click', () => {
  ecoBot.style.transform = 'scale(1.2) rotate(10deg)';
  showRandomMessage();
  setTimeout(() => {
    ecoBot.style.transform = '';
  }, 800);
});
