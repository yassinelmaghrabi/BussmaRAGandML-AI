<script lang="ts">
  import { API_BASE_URL } from "../lib/apiConfig";
  import { onMount } from "svelte";

  interface Message {
    id: string;
    text: string;
    isUser: boolean;
    timestamp: Date;
  }

  let messages: Message[] = [];
  let inputValue = "";
  let isLoading = false;
  let chatContainer: HTMLElement;
  let inputElement: HTMLInputElement;

  // Auto-scroll to bottom when new messages are added
  $: if (messages.length && chatContainer) {
    setTimeout(() => {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 10);
  }

  onMount(() => {
    // Focus input on mount
    inputElement?.focus();

    // Add welcome message
    messages = [
      {
        id: generateId(),
        text: "Hello! I'm your AI assistant. How can I help you today?",
        isUser: false,
        timestamp: new Date(),
      },
    ];
  });

  function generateId(): string {
    return Date.now().toString() + Math.random().toString(36).substr(2, 9);
  }

  async function sendMessage() {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: generateId(),
      text: inputValue.trim(),
      isUser: true,
      timestamp: new Date(),
    };

    // Add user message
    messages = [...messages, userMessage];
    const currentInput = inputValue;
    inputValue = "";
    isLoading = true;

    try {
      const response = await fetch(`${API_BASE_URL}/query-rag`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: currentInput }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const aiMessage: Message = {
        id: generateId(),
        text: data.response || "I'm sorry, I couldn't process your request.",
        isUser: false,
        timestamp: new Date(),
      };

      messages = [...messages, aiMessage];
    } catch (error) {
      console.error("Error sending message:", error);

      const errorMessage: Message = {
        id: generateId(),
        text: "Sorry, I'm having trouble connecting right now. Please try again later.",
        isUser: false,
        timestamp: new Date(),
      };

      messages = [...messages, errorMessage];
    } finally {
      isLoading = false;
      inputElement?.focus();
    }
  }

  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function clearChat() {
    messages = [
      {
        id: generateId(),
        text: "Chat cleared. How can I help you?",
        isUser: false,
        timestamp: new Date(),
      },
    ];
  }

  function formatTime(date: Date): string {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
</script>

<div class="chat-container">
  <header class="chat-header">
    <div class="header-content">
      <h1>ChatBot</h1>
      <p>AI Assistant</p>
    </div>
    <button class="clear-btn" on:click={clearChat} title="Clear chat">
      <svg
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        stroke-width="2"
      >
        <path
          d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6h14zM10 11v6M14 11v6"
        />
      </svg>
    </button>
  </header>

  <div class="messages-container" bind:this={chatContainer}>
    {#each messages as message (message.id)}
      <div class="message {message.isUser ? 'user' : 'assistant'}">
        <div class="message-content">
          <div class="message-text">
            {message.text}
          </div>
          <div class="message-time">
            {formatTime(message.timestamp)}
          </div>
        </div>
      </div>
    {/each}

    {#if isLoading}
      <div class="message assistant">
        <div class="message-content">
          <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <div class="input-container">
    <div class="input-wrapper">
      <textarea
        bind:this={inputElement}
        bind:value={inputValue}
        on:keydown={handleKeyPress}
        placeholder="Type your message..."
        rows="1"
        disabled={isLoading}
      ></textarea>
      <button
        class="send-btn"
        on:click={sendMessage}
        disabled={!inputValue.trim() || isLoading}
        title="Send message"
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
        </svg>
      </button>
    </div>
  </div>
</div>

<style>
  .chat-container {
    display: flex;
    flex-direction: column;
    height: 90vh;
    min-height: 500px;
    max-height: 900px;
    min-width: 1000px;
    max-width: 1600px;
    margin: 2rem auto;
    background: #f8f9fa;
    border: 1px solid #e1e5e9;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  }

  .chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom: 1px solid #e1e5e9;
  }

  .header-content h1 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
  }

  .header-content p {
    margin: 0;
    font-size: 0.9rem;
    opacity: 0.9;
  }

  .clear-btn {
    background: rgba(255, 255, 255, 0.2);
    border: none;
    color: white;
    padding: 0.5rem;
    border-radius: 6px;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .clear-btn:hover {
    background: rgba(255, 255, 255, 0.3);
  }

  .messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    scroll-behavior: smooth;
  }

  .message {
    display: flex;
    animation: fadeIn 0.3s ease-in-out;
  }

  .message.user {
    justify-content: flex-end;
  }

  .message.assistant {
    justify-content: flex-start;
  }

  .message-content {
    max-width: 70%;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
  }

  .message-text {
    padding: 0.75rem 1rem;
    border-radius: 18px;
    word-wrap: break-word;
    line-height: 1.4;
  }

  .user .message-text {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-bottom-right-radius: 6px;
  }

  .assistant .message-text {
    background: white;
    color: #333;
    border: 1px solid #e1e5e9;
    border-bottom-left-radius: 6px;
  }

  .message-time {
    font-size: 0.75rem;
    color: #666;
    padding: 0 0.5rem;
  }

  .user .message-time {
    text-align: right;
  }

  .assistant .message-time {
    text-align: left;
  }

  .typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    padding: 1rem;
    background: white;
    border: 1px solid #e1e5e9;
    border-radius: 18px;
    border-bottom-left-radius: 6px;
  }

  .typing-indicator span {
    width: 8px;
    height: 8px;
    background: #666;
    border-radius: 50%;
    animation: typing 1.4s infinite ease-in-out;
  }

  .typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
  }

  .typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
  }

  .input-container {
    padding: 1rem 1.5rem;
    background: white;
    border-top: 1px solid #e1e5e9;
  }

  .input-wrapper {
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
  }

  textarea {
    flex: 1;
    resize: none;
    border: 1px solid #e1e5e9;
    border-radius: 20px;
    padding: 0.75rem 1rem;
    font-family: inherit;
    font-size: 0.95rem;
    outline: none;
    transition:
      border-color 0.2s,
      box-shadow 0.2s;
    max-height: 120px;
    overflow-y: auto;
  }

  textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
  }

  textarea:disabled {
    background-color: #f5f5f5;
    cursor: not-allowed;
  }

  .send-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 50%;
    width: 44px;
    height: 44px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition:
      transform 0.2s,
      box-shadow 0.2s;
  }

  .send-btn:hover:not(:disabled) {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
  }

  .send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes typing {
    0%,
    60%,
    100% {
      transform: translateY(0);
    }
    30% {
      transform: translateY(-10px);
    }
  }

  /* Responsive design */
  @media (max-width: 768px) {
    .chat-container {
      height: 70vh;
      min-height: 400px;
      margin: 1rem;
      border-radius: 8px;
    }

    .message-content {
      max-width: 85%;
    }

    .chat-header {
      padding: 1rem;
    }

    .input-container {
      padding: 1rem;
    }
  }

  /* Custom scrollbar */
  .messages-container::-webkit-scrollbar {
    width: 6px;
  }

  .messages-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
  }

  .messages-container::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
  }

  .messages-container::-webkit-scrollbar-thumb:hover {
    background: #a8a8a8;
  }
</style>
