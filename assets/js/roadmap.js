// 5 walking frames
const walkFrames = [
  "/assets/images/sprite/walk_1.png",
  "/assets/images/sprite/walk_2.png",
  "/assets/images/sprite/walk_3.png",
  "/assets/images/sprite/walk_4.png",
  "/assets/images/sprite/walk_5.png",
];
const idleFrame = "/assets/images/sprite/idle.png";
// Roadmap milestone data
const roadmapData = [
  {
    x: 100,
    y: 400,
    title: "Started College",
    role: "Student",
    location: "BITS Pilani",
    logo: "/assets/images/bits_logo.png",
  },
  {
    x: 300,
    y: 300,
    title: "Interned at Amazon",
    role: "SDE Intern - Alexa",
    location: "Bangalore",
    logo: "/assets/images/amazon_logo.png",
  },
  {
    x: 600,
    y: 350,
    title: "Research Project",
    role: "Semantic Segmentation",
    location: "BITS + Mentor",
    logo: "/assets/images/research.png",
  },
];

// Decorative elements (trees, stones, mountains)
const decorations = [
  { type: "tree1", x: 180, y: 380 },
  { type: "tree2", x: 450, y: 300 },
  { type: "stone", x: 250, y: 340 },
  { type: "mountain", x: 550, y: 200 },
];

// Show milestone popup
function showPopup(step) {
  const popup = document.getElementById("popup");
  document.getElementById("popup-logo").src = step.logo;
  document.getElementById("popup-title").textContent = step.title;
  document.getElementById("popup-role").textContent = step.role;
  document.getElementById("popup-location").textContent = step.location;
  popup.style.left = `${step.x}px`;
  popup.style.top = `${step.y - 120}px`;
  popup.classList.remove("hidden");
}

export function initRoadmap() {
  const roadmapContainer = document.getElementById("roadmap-container");
  const charSprite = document.createElement("img");
  charSprite.id = "character";
  charSprite.src = "/assets/images/character_idle.png";
  charSprite.classList.add("character");
  charSprite.style.position = "absolute";
  roadmapContainer.appendChild(charSprite);

  let current = 0; // Current milestone index
  let isMoving = false; // Declare isMoving here
  let walkInterval;

  // Add decorative elements
  decorations.forEach(({ type, x, y }) => {
    const deco = document.createElement("img");
    deco.src = `/assets/images/${type}.png`;
    deco.classList.add("deco");
    deco.style.position = "absolute";
    deco.style.left = `${x}px`;
    deco.style.top = `${y}px`;
    roadmapContainer.appendChild(deco);
  });

  // Add milestones
  roadmapData.forEach((milestone, i) => {
    const block = document.createElement("img");
    block.src = "/assets/images/milestone.png";
    block.classList.add("milestone");
    block.style.position = "absolute";
    block.style.left = `${milestone.x}px`;
    block.style.top = `${milestone.y}px`;
    block.dataset.index = i;
    roadmapContainer.appendChild(block);
  });

  // Animate character movement
  function animateCharacter(from, to, duration = 1000) {
    if (isMoving) return;
    isMoving = true;

    let startTime = null;
    let frameIndex = 0;

    walkInterval = setInterval(() => {
      charSprite.src = walkFrames[frameIndex % walkFrames.length];
      frameIndex++;
    }, 100);

    const deltaX = to.x - from.x;
    const deltaY = to.y - from.y;

    const animate = (timestamp) => {
      if (!startTime) startTime = timestamp;
      const elapsed = timestamp - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const currentX = from.x + deltaX * progress;
      const currentY = from.y + deltaY * progress;

      charSprite.style.left = `${currentX}px`;
      charSprite.style.top = `${currentY}px`;

      // Scroll camera to follow character
      roadmapContainer.scrollTo({
        left: currentX - window.innerWidth / 2 + 50,
        top: currentY - window.innerHeight / 2 + 50,
        behavior: "smooth",
      });

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        clearInterval(walkInterval);
        charSprite.src = idleFrame;
        isMoving = false;
        showPopup(to);
      }
    };

    requestAnimationFrame(animate);
  }

  // Go to a specific milestone
  function goToStep(index) {
    if (index < 0 || index >= roadmapData.length || isMoving) return;
    const from = roadmapData[current];
    const to = roadmapData[index];
    current = index;
    animateCharacter(from, to);
  }

  // Key listeners
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") {
      goToStep(current + 1);
      console.log("Right Arrow Pressed");
    }
    if (e.key === "ArrowLeft") goToStep(current - 1);
  });

  // Start position
  charSprite.style.left = `${roadmapData[0].x}px`;
  charSprite.style.top = `${roadmapData[0].y}px`;
  goToStep(0);
}
