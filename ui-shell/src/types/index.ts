// Core types for Lift OS UI Shell

export interface User {
  id: string;
  email: string;
  name: string;
  org_id: string;
  roles: UserRole[];
  is_active: boolean;
  last_login?: string;
  metadata: Record<string, any>;
  created_at: string;
  updated_at?: string;
}

export enum UserRole {
  ADMIN = 'admin',
  USER = 'user',
  ANALYST = 'analyst',
  DEVELOPER = 'developer',
  VIEWER = 'viewer',
}

export interface Organization {
  id: string;
  name: string;
  domain?: string;
  subscription_tier: SubscriptionTier;
  settings: Record<string, any>;
  is_active: boolean;
  created_at: string;
  updated_at?: string;
}

export enum SubscriptionTier {
  FREE = 'free',
  BASIC = 'basic',
  PRO = 'pro',
  ENTERPRISE = 'enterprise',
}

export interface Module {
  id: string;
  name: string;
  version: string;
  base_url: string;
  health_endpoint: string;
  api_prefix: string;
  status: ModuleStatus;
  features: string[];
  permissions: string[];
  memory_requirements: Record<string, any>;
  ui_components: UIComponent[];
  metadata: Record<string, any>;
  created_at: string;
  updated_at?: string;
}

export enum ModuleStatus {
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  MAINTENANCE = 'maintenance',
  ERROR = 'error',
}

export interface UIComponent {
  name: string;
  path: string;
  permissions: string[];
}

export interface MemorySearchRequest {
  query: string;
  search_type: 'neural' | 'conceptual' | 'knowledge' | 'hybrid';
  limit: number;
  filters?: Record<string, any>;
  memory_type?: string;
}

export interface MemorySearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, any>;
  memory_type: string;
  timestamp: string;
}

export interface MemoryInsights {
  total_memories: number;
  dominant_concepts: string[];
  knowledge_density: number;
  temporal_patterns: Record<string, any>;
  semantic_clusters: Array<Record<string, any>>;
  memory_types: Record<string, number>;
}

export interface BillingPlan {
  plan_id: string;
  name: string;
  description: string;
  price_monthly: number;
  price_yearly: number;
  features: Record<string, any>;
  limits: Record<string, any>;
  is_active: boolean;
}

export interface Subscription {
  subscription_id: string;
  organization_id: string;
  plan_id: string;
  billing_cycle: 'monthly' | 'yearly';
  status: string;
  current_period_start: string;
  current_period_end: string;
  created_at: string;
  plan: BillingPlan;
}

export interface UsageRecord {
  usage_id: string;
  organization_id: string;
  service_name: string;
  usage_type: string;
  quantity: number;
  timestamp: string;
  metadata: Record<string, any>;
}

export interface SystemHealth {
  system_health: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  active_services: number;
  total_alerts: number;
  critical_alerts: number;
  avg_response_time: number;
  uptime_percentage: number;
  last_updated: string;
}

export interface ServiceHealth {
  service_name: string;
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  response_time_ms: number;
  details?: Record<string, any>;
}

export interface Alert {
  alert_id: string;
  name: string;
  description: string;
  severity: 'critical' | 'warning' | 'info';
  status: 'firing' | 'resolved';
  timestamp: string;
  organization_id?: string;
  service_name?: string;
  metadata?: Record<string, any>;
}

export interface APIResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  errors?: string[];
  metadata?: Record<string, any>;
}

export interface PaginatedResponse<T = any> {
  items: T[];
  total: number;
  page: number;
  size: number;
  pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface AuthState {
  user: User | null;
  organization: Organization | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface NavigationItem {
  name: string;
  href: string;
  icon?: any;
  current?: boolean;
  children?: NavigationItem[];
  permissions?: string[];
}

export interface DashboardStats {
  total_users: number;
  active_modules: number;
  memory_usage: number;
  api_calls_today: number;
  system_health: string;
  recent_activities: Activity[];
}

export interface Activity {
  id: string;
  type: string;
  description: string;
  timestamp: string;
  user?: string;
  metadata?: Record<string, any>;
}

export interface NotificationSettings {
  email_notifications: boolean;
  push_notifications: boolean;
  alert_notifications: boolean;
  billing_notifications: boolean;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  timezone: string;
  notifications: NotificationSettings;
  dashboard_layout: string[];
}

// Form types
export interface LoginForm {
  email: string;
  password: string;
  remember_me?: boolean;
}

export interface RegisterForm {
  name: string;
  email: string;
  password: string;
  confirm_password: string;
  organization_name: string;
  terms_accepted: boolean;
}

export interface MemorySearchForm {
  query: string;
  search_type: 'neural' | 'conceptual' | 'knowledge' | 'hybrid';
  memory_type?: string;
  limit: number;
}

export interface ModuleRegistrationForm {
  module_id: string;
  name: string;
  version: string;
  base_url: string;
  health_endpoint: string;
  api_prefix: string;
  features: string[];
  permissions: string[];
}

// Chart data types
export interface ChartDataPoint {
  name: string;
  value: number;
  timestamp?: string;
}

export interface TimeSeriesData {
  timestamp: string;
  value: number;
  label?: string;
}

export interface MetricData {
  name: string;
  current_value: number;
  previous_value?: number;
  change_percentage?: number;
  trend: 'up' | 'down' | 'stable';
  data_points: TimeSeriesData[];
}