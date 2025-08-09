<script lang="ts">
  import Home from "./routes/Home.svelte";
  import ChatBot from "./routes/ChatBot.svelte";
  import Graphs from "./routes/Graphs.svelte";
  import { Bot, ChartLine, Home as HomeIcon } from "@lucide/svelte";

  let currentRoute: string = window.location.pathname;

  function navigate(path: string) {
    if (path !== currentRoute) {
      window.history.pushState({}, "", path);
      currentRoute = path;
    }
  }

  window.addEventListener("popstate", () => {
    currentRoute = window.location.pathname;
  });

  function getGradientClass(route: string): string {
    if (route === "/chatbot") return "chatbot-gradient";
    if (route === "/graphs") return "graphs-gradient";
    return "home-gradient";
  }

  $: currentGradient = getGradientClass(currentRoute);
</script>

<div class="app">
  <div class="gradient-bg {currentGradient}"></div>

  <nav>
    <div class="nav-container">
      <button
        on:click={() => navigate("/")}
        class:active={currentRoute === "/"}
      >
        <HomeIcon size={18} /> Home
      </button>
      <button
        on:click={() => navigate("/chatbot")}
        class:active={currentRoute === "/chatbot"}
      >
        <Bot size={18} /> ChatBot
      </button>
      <button
        on:click={() => navigate("/graphs")}
        class:active={currentRoute === "/graphs"}
      >
        <ChartLine size={18} /> AI/ML Graphs
      </button>
    </div>
  </nav>

  <main>
    {#if currentRoute === "/"}
      <Home />
    {:else if currentRoute === "/chatbot"}
      <ChatBot />
    {:else if currentRoute === "/graphs"}
      <Graphs />
    {:else}
      <h2>404: Page not found</h2>
    {/if}
  </main>
</div>

<style>
  :global(*) {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  :global(body) {
    margin: 0;
    font-family: sans-serif;
    overflow-x: hidden;
  }

  :global(html) {
    margin: 0;
    padding: 0;
  }

  .app {
    position: relative;
    min-height: 80vh;
    overflow-y: visible;
  }

  .gradient-bg {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-size: 200% 200%;
    animation: radialPulse 8s ease-in-out infinite;
    z-index: -1;
    transition: background 1.2s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .home-gradient {
    background: radial-gradient(
      ellipse at 30% 20%,
      #1e293b 0%,
      /* slate-800 */ #0f172a 50%,
      /* slate-900 */ #020617 100% /* almost black */
    );
  }

  .chatbot-gradient {
    background: radial-gradient(
      ellipse at 70% 80%,
      #3b0764 0%,
      /* deep purple */ #1e1b4b 50%,
      /* indigo-950 */ #0f172a 100% /* slate-900 */
    );
  }

  .graphs-gradient {
    background: radial-gradient(
      ellipse at 50% 50%,
      #0f766e 0%,
      /* teal-800 */ #164e63 50%,
      /* cyan-900 */ #082f49 100% /* deeper blue-gray */
    );
  }

  @keyframes radialPulse {
    0%,
    100% {
      background-size: 200% 200%;
      background-position: 0% 50%;
    }
    25% {
      background-size: 250% 250%;
      background-position: 100% 0%;
    }
    50% {
      background-size: 300% 300%;
      background-position: 50% 100%;
    }
    75% {
      background-size: 250% 250%;
      background-position: 0% 0%;
    }
  }

  nav {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    background-color: rgba(17, 17, 17, 0.9);
    backdrop-filter: blur(10px);
    z-index: 1000;
    box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
  }

  .nav-container {
    max-width: 1000px;
    margin: 0 auto;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 2rem;
    padding: 1rem;
  }

  nav button {
    color: white;
    font-weight: 500;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: none;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    transition: all 0.3s ease;
  }

  nav button:hover {
    background-color: rgba(255, 255, 255, 0.1);
    transform: translateY(-2px);
  }

  nav button.active {
    background-color: rgba(255, 255, 255, 0.2);
    font-weight: bold;
  }

  main {
    padding: 6rem 2rem 2rem;
    max-width: 1200px;
    margin: 0 auto;
    position: relative;
    z-index: 1;
  }
</style>
