// API client for Lift OS Core services

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import Cookies from 'js-cookie';
import { toast } from 'react-hot-toast';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance
const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = Cookies.get('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error) => {
    if (error.response?.status === 401) {
      // Unauthorized - clear token and redirect to login
      Cookies.remove('auth_token');
      window.location.href = '/login';
      toast.error('Session expired. Please log in again.');
    } else if (error.response?.status === 403) {
      toast.error('You do not have permission to perform this action.');
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Please try again later.');
    } else if (error.response?.data?.message) {
      toast.error(error.response.data.message);
    } else if (error.message) {
      toast.error(error.message);
    }
    
    return Promise.reject(error);
  }
);

// Auth API
export const authAPI = {
  login: (email: string, password: string) =>
    api.post('/api/v1/auth/login', { email, password }),
  
  register: (userData: {
    name: string;
    email: string;
    password: string;
    organization_name: string;
  }) =>
    api.post('/api/v1/auth/register', userData),
  
  logout: () =>
    api.post('/api/v1/auth/logout'),
  
  me: () =>
    api.get('/api/v1/auth/me'),
  
  refreshToken: () =>
    api.post('/api/v1/auth/refresh'),
  
  forgotPassword: (email: string) =>
    api.post('/api/v1/auth/forgot-password', { email }),
  
  resetPassword: (token: string, password: string) =>
    api.post('/api/v1/auth/reset-password', { token, password }),
};

// Memory API
export const memoryAPI = {
  search: (searchRequest: {
    query: string;
    search_type: string;
    limit: number;
    filters?: any;
    memory_type?: string;
  }) =>
    api.post('/api/v1/memory/search', searchRequest),
  
  store: (storeRequest: {
    content: string;
    metadata?: any;
    memory_type: string;
  }) =>
    api.post('/api/v1/memory/store', storeRequest),
  
  getInsights: () =>
    api.get('/api/v1/memory/insights'),
  
  getContexts: () =>
    api.get('/api/v1/memory/contexts'),
  
  createContext: (contextData: {
    context_type: string;
    domain: string;
    settings?: any;
  }) =>
    api.post('/api/v1/memory/contexts', contextData),
};

// Billing API
export const billingAPI = {
  getPlans: () =>
    api.get('/api/v1/billing/plans'),
  
  getSubscription: (organizationId: string) =>
    api.get(`/api/v1/billing/subscriptions/${organizationId}`),
  
  createSubscription: (subscriptionData: {
    organization_id: string;
    plan_id: string;
    billing_cycle: string;
    payment_method_id?: string;
  }) =>
    api.post('/api/v1/billing/subscriptions', subscriptionData),
  
  updateSubscription: (organizationId: string, updateData: {
    plan_id?: string;
    billing_cycle?: string;
  }) =>
    api.put(`/api/v1/billing/subscriptions/${organizationId}`, updateData),
  
  getUsage: (organizationId: string, params?: {
    start_date?: string;
    end_date?: string;
  }) =>
    api.get(`/api/v1/billing/usage/${organizationId}`, { params }),
  
  getAnalytics: (organizationId: string) =>
    api.get(`/api/v1/billing/analytics/${organizationId}`),
};

// Registry API
export const registryAPI = {
  getModules: () =>
    api.get('/api/v1/registry/modules'),
  
  getModule: (moduleId: string) =>
    api.get(`/api/v1/registry/modules/${moduleId}`),
  
  registerModule: (moduleData: any) =>
    api.post('/api/v1/registry/modules', moduleData),
  
  updateModule: (moduleId: string, updateData: any) =>
    api.put(`/api/v1/registry/modules/${moduleId}`, updateData),
  
  deleteModule: (moduleId: string) =>
    api.delete(`/api/v1/registry/modules/${moduleId}`),
  
  getModuleHealth: (moduleId: string) =>
    api.get(`/api/v1/registry/modules/${moduleId}/health`),
};

// Observability API
export const observabilityAPI = {
  getSystemOverview: () =>
    api.get('/api/v1/observability/overview'),
  
  getServicesHealth: () =>
    api.get('/api/v1/observability/health/services'),
  
  getMetrics: (queryParams?: {
    metric_name?: string;
    start_time?: string;
    end_time?: string;
    labels?: any;
    organization_id?: string;
    service_name?: string;
  }) =>
    api.post('/api/v1/observability/metrics/query', queryParams),
  
  recordMetric: (metricData: {
    name: string;
    value: number;
    labels?: any;
    organization_id?: string;
    service_name?: string;
  }) =>
    api.post('/api/v1/observability/metrics', metricData),
  
  getLogs: (serviceName: string, params?: {
    level?: string;
    limit?: number;
  }) =>
    api.get(`/api/v1/observability/logs/${serviceName}`, { params }),
  
  recordLog: (logData: {
    level: string;
    message: string;
    service_name: string;
    organization_id?: string;
    correlation_id?: string;
    metadata?: any;
  }) =>
    api.post('/api/v1/observability/logs', logData),
  
  getAlerts: (params?: {
    status?: string;
    severity?: string;
    organization_id?: string;
  }) =>
    api.get('/api/v1/observability/alerts', { params }),
  
  createAlert: (alertData: {
    name: string;
    description: string;
    severity: string;
    organization_id?: string;
    service_name?: string;
    metadata?: any;
  }) =>
    api.post('/api/v1/observability/alerts', alertData),
  
  resolveAlert: (alertId: string) =>
    api.put(`/api/v1/observability/alerts/${alertId}/resolve`),
};

// Organizations API
export const organizationsAPI = {
  getCurrent: () =>
    api.get('/api/v1/organizations/current'),
  
  update: (organizationId: string, updateData: {
    name?: string;
    domain?: string;
    settings?: any;
  }) =>
    api.put(`/api/v1/organizations/${organizationId}`, updateData),
  
  getUsers: (organizationId: string) =>
    api.get(`/api/v1/organizations/${organizationId}/users`),
  
  inviteUser: (organizationId: string, inviteData: {
    email: string;
    roles: string[];
  }) =>
    api.post(`/api/v1/organizations/${organizationId}/invite`, inviteData),
  
  removeUser: (organizationId: string, userId: string) =>
    api.delete(`/api/v1/organizations/${organizationId}/users/${userId}`),
  
  updateUserRoles: (organizationId: string, userId: string, roles: string[]) =>
    api.put(`/api/v1/organizations/${organizationId}/users/${userId}/roles`, { roles }),
};

// Users API
export const usersAPI = {
  updateProfile: (updateData: {
    name?: string;
    email?: string;
  }) =>
    api.put('/api/v1/users/profile', updateData),
  
  changePassword: (passwordData: {
    current_password: string;
    new_password: string;
  }) =>
    api.put('/api/v1/users/password', passwordData),
  
  getPreferences: () =>
    api.get('/api/v1/users/preferences'),
  
  updatePreferences: (preferences: any) =>
    api.put('/api/v1/users/preferences', preferences),
  
  deleteAccount: () =>
    api.delete('/api/v1/users/account'),
};

// Health check
export const healthAPI = {
  check: () =>
    api.get('/health'),
  
  checkService: (serviceName: string) =>
    api.get(`/health/${serviceName}`),
};

export default api;