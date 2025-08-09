// src/lib/apiConfig.ts
export const API_HOST = import.meta.env.VITE_API_HOST || "localhost";
export const API_PORT = import.meta.env.VITE_API_PORT || "8000";

// This is the full base URL
export const API_BASE_URL = `http://${API_HOST}:${API_PORT}`;

