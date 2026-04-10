import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_MIDDLEWARE_API_URL,
});

export default api;