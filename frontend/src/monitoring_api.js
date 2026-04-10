import axios from 'axios';

const monitoring_api = axios.create({
  baseURL: import.meta.env.VITE_MONITORING_API_URL});

export default monitoring_api;