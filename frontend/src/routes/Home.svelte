<script lang="ts">
  let logoEl: SVGSVGElement;

  function handleMouseMove(event: MouseEvent) {
    const { width, height, left, top } = logoEl.getBoundingClientRect();
    const x = event.clientX - left;
    const y = event.clientY - top;

    const rotateX = ((y - height / 2) / height) * -15;
    const rotateY = ((x - width / 2) / width) * 15;

    logoEl.style.transform = `rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.05)`;
  }

  function resetTilt() {
    logoEl.style.transform = `rotateX(0deg) rotateY(0deg) scale(1)`;
  }
</script>

<div class="home">
  <div class="logo-container">
    <svg
      bind:this={logoEl}
      on:mousemove={handleMouseMove}
      on:mouseleave={resetTilt}
      role="log"
      version="1.2"
      xmlns="http://www.w3.org/2000/svg"
      class="logo"
      viewBox="0 0 700 700"
    >
      <title>pharmalyticalogo</title>
      <style>
        .s0 {
          fill: #2ca28e;
        }
        .s1 {
          fill: #0e3a5e;
        }
      </style>
      <g>
        <g>
          <g id="stage">
            <g id="layer1 1">
              <g id="Clip-Path: Group 1">
                <g id="Group 1">
                  <g id="Group 1">
                    <g id="Clip-Path: Group 1">
                      <g id="Group 1">
                        <path
                          id="Path 1"
                          class="s0"
                          d="m341.7 86.5c34.1-3.3 74 11.1 100.5 32.4 29.4 23.6 48.1 58 52.1 95.5 1.7 16.5 0.7 33.9 0.7 50.6v85.8 83.2c0.1 20.5 1.6 42.1-2.3 62.2-4.7 23.5-15.2 45.5-30.4 64.1-26.7 32.2-62.2 49.8-103.6 53.6-38.4 1.7-74.6-10.1-103.9-35.4-28.1-24.2-47.5-59.9-50.2-97.2-1.1-16-0.3-32.5-0.3-48.5v-89.9-79.3c-0.1-20.4-1.4-41.5 2.7-61.6 4.8-24.6 16.1-47.5 32.7-66.3 26.1-29.9 62.6-46.6 102-49.2zm-85.4 67.1c-13 15.2-21.8 33.4-25.8 52.9-4 19.8-2.6 41.3-2.6 61.4l0.1 80.9-0.1 84.6c0 15.6-0.7 31.8 0.6 47.4 2.7 31.1 19.7 61.2 43.7 81 24.9 20.5 55 29 86.9 27.6 34.4-4 63.8-18.4 85.7-45.8 12.6-15.7 21.1-34.4 24.7-54.2 3.1-17.6 1.8-36.5 1.8-54.3v-89.7l0.1-82.7c0-15.3 0.9-31.3-0.9-46.4-3.7-31.3-19.8-59.9-44.6-79.3-23.8-18.6-53.6-27.9-83.8-26.1-33.5 2.9-63.9 16.8-85.8 42.7z"
                        />
                        <path
                          id="Path 1"
                          class="s1"
                          d="m394.5 222.6h39v155.5l-38.9 0.1z"
                        />
                        <path
                          id="Path 1"
                          class="s1"
                          d="m330.3 273.1l38.7 0.1v105l-38.7-0.1z"
                        />
                        <path
                          id="Path 1"
                          class="s1"
                          d="m265.2 319.3h39.4q-0.1 29.4 0.1 58.8c-13.1 0.3-26.4 0-39.6 0.1z"
                        />
                      </g>
                    </g>
                  </g>
                </g>
              </g>
            </g>
          </g>
        </g>
      </g>
    </svg>
  </div>

  <h1>Welcome to Pharmalytica AI Dashboard</h1>
  <p>Select a feature from the top navigation to begin.</p>
  <div class="about-section">
    <h2>About Pharmalytica</h2>
    <p>
      Pharmalytica is a cutting-edge analytics platform built to empower
      pharmaceutical professionals with actionable insights derived from complex
      medical and sales data. Our goal is to simplify decision-making using
      AI-driven tools, interactive dashboards, and natural language interfaces.
    </p>
    <p>
      This project combines advanced data visualization, real-time AI
      assistance, and intelligent data processing to support pharmacies,
      manufacturers, and healthcare institutions in optimizing their operations
      and understanding key market trends.
    </p>
  </div>
</div>

<style>
  .about-section {
    max-width: 800px;
    margin: 2rem auto 0;
    text-align: center;
    font-size: 1.1rem;
    line-height: 1.6;
    color: #d1d5db; /* Tailwind's gray-300; works well on dark backgrounds */
  }

  .about-section h2 {
    font-size: 1.75rem;
    margin-bottom: 1rem;
    color: #f9fafb; /* Tailwind's gray-50; for headings */
  }
  .about-section {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 0.8s ease forwards;
    animation-delay: 0.5s;
  }

  @keyframes fadeInUp {
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .logo-container {
    text-align: center;
    margin-bottom: 2rem;
    perspective: 150px;
    display: inline-block;
  }

  .logo {
    width: 250px;
    height: auto;
    display: block;
    margin: 0 auto;
    transition: transform 0.2s ease;
    transform-style: preserve-3d;
    will-change: transform;
  }

  .logo path {
    transition: filter 0.3s ease;
    filter: none;
  }

  .logo:hover .s1 {
    filter: drop-shadow(0 0 20px #0e3a5e);
  }

  .logo:hover .s0 {
    filter: drop-shadow(0 0 20px #2ca28e);
  }
</style>
